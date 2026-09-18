"""Tests for multilingual document upload and Q&A pipeline."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.utils.language import detect_language, check_embedding_multilingual_support

client = TestClient(app)


def test_language_detection():
    """Verify language detection across non-Latin scripts and Latin languages."""
    assert detect_language("வணக்கம், இந்த ஆவணத்தின் உள்ளடக்கம் என்ன?") == "Tamil"
    assert detect_language("यह दस्तावेज़ किस बारे में है?") == "Hindi"
    assert detect_language("ما هي المعلومات الواردة في هذا المستند؟") == "Arabic"
    assert detect_language("Bonjour, quel est le contenu de ce document?") == "French"
    assert detect_language("¿Cuál es el contenido de este documento?") == "Spanish"
    assert detect_language("Was ist der Inhalt dieses Dokuments?") == "German"
    assert detect_language("What is the main topic of this document?") == "English"


def test_embedding_capability_warning():
    """Verify English-centric embedding model generates explicit warning without crashing."""
    result = check_embedding_multilingual_support("all-MiniLM-L6-v2")
    assert result["is_multilingual_native"] is False
    assert result["warning"] is not None
    assert "English-centric" in result["warning"]


@patch("app.services.rag.RAGService._call_openrouter")
def test_multilingual_document_qa(mock_llm):
    """Test document upload in French and Q&A in French and English."""
    mock_llm.return_value = "Le taux d'intérêt pour le compte d'épargne Zanaco est de 5.5%."

    # 1. Upload French document
    french_doc = "Compte d'Épargne Zanaco: Taux d'intérêt annuel est de 5.5%. Solde minimum requis est de 200 ZMW.".encode("utf-8")
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("epargne_zanaco.txt", french_doc, "text/plain")}
    )
    assert upload_res.status_code == 200
    session_id = upload_res.json()["session_id"]

    # 2. Query in French (matching language)
    ask_res_fr = client.post(
        "/chat/ask",
        json={
            "session_id": session_id,
            "question": "Quel est le taux d'intérêt du compte d'épargne?"
        }
    )
    assert ask_res_fr.status_code == 200
    assert ask_res_fr.json()["answer_source"] == "user_document"
    assert "5.5%" in ask_res_fr.json()["answer"]

    # 3. Verify mock call received system prompt specifying French
    _, kwargs = mock_llm.call_args
    assert "French" in kwargs.get("system_prompt", "")


@patch("app.services.rag.RAGService._call_openrouter")
def test_cross_lingual_document_qa(mock_llm):
    """Test cross-lingual Q&A: French document queried in English."""
    mock_llm.return_value = "The annual interest rate for the Zanaco savings account is 5.5%."

    # 1. Upload French document
    french_doc = "Compte d'Épargne Zanaco: Taux d'intérêt annuel est de 5.5%.".encode("utf-8")
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("doc_fr.txt", french_doc, "text/plain")}
    )
    assert upload_res.status_code == 200
    session_id = upload_res.json()["session_id"]

    # 2. Query in English on French document
    ask_res_en = client.post(
        "/chat/ask",
        json={
            "session_id": session_id,
            "question": "What is the savings account interest rate?"
        }
    )
    assert ask_res_en.status_code == 200
    assert ask_res_en.json()["answer_source"] == "user_document"

    # 3. Verify system prompt instructed answer in English
    _, kwargs = mock_llm.call_args
    assert "English" in kwargs.get("system_prompt", "")
