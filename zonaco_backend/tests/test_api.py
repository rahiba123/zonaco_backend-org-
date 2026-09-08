"""Automated test suite for User Document Upload and RAG Q&A Backend."""

import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.rag import RAGResponse
from app.state_machine import SessionState, StateMachine

client = TestClient(app)


def test_session_start():
    """Test starting a session returns welcome menu."""
    start_res = client.post("/session/start", json={"user_id": "test_user_123"})
    assert start_res.status_code == 201
    start_data = start_res.json()
    assert "session_id" in start_data
    assert start_data["current_state"] == "MAIN_MENU"


@patch("app.services.rag.RAGService._call_openrouter")
def test_document_summary_and_explicit_query(mock_llm):
    """Test asking for summary of uploaded document routes correctly to user document RAG."""
    mock_llm.return_value = "This document provides an overview of Oracle FLEXCUBE Core Services (CS)."

    # 1. Upload sample document
    txt_content = b"Oracle FLEXCUBE Universal Banking 12.0.3 Core Services CS overview and maintenance parameters."
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("flexcube_cs.txt", txt_content, "text/plain")}
    )
    assert upload_res.status_code == 200
    session_id = upload_res.json()["session_id"]

    # 2. Ask summary question
    ask_res = client.post(
        "/chat/ask",
        json={
            "session_id": session_id,
            "question": "explain the summary of this document"
        }
    )
    assert ask_res.status_code == 200
    ask_data = ask_res.json()
    assert ask_data["answer_source"] == "user_document"
    assert "Oracle FLEXCUBE" in ask_data["answer"]


def test_upload_csv_document():
    """Test uploading a CSV document."""
    csv_content = b"Account Type,Interest Rate,Minimum Balance\nSavings,4.5%,100 ZMW\nFixed Deposit,8.0%,1000 ZMW"
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("rates.csv", csv_content, "text/csv")}
    )
    assert upload_res.status_code == 200
    assert upload_res.json()["filename"] == "rates.csv"
    assert upload_res.json()["chunks_indexed"] > 0


def test_upload_excel_document():
    """Test uploading an Excel (.xlsx) document."""
    import openpyxl
    from io import BytesIO

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Loan Products"
    ws.append(["Product", "Interest Rate", "Max Tenure"])
    ws.append(["Personal Loan", "18%", "36 Months"])
    ws.append(["Home Loan", "12%", "240 Months"])
    buffer = BytesIO()
    wb.save(buffer)
    xlsx_bytes = buffer.getvalue()

    upload_res = client.post(
        "/documents/upload",
        files={"file": ("loan_products.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    )
    assert upload_res.status_code == 200
    assert upload_res.json()["filename"] == "loan_products.xlsx"
    assert upload_res.json()["chunks_indexed"] > 0


def test_upload_markdown_document():
    """Test uploading a Markdown (.md) document."""
    md_content = b"# Zanaco Express Services\n\n- Cash deposit\n- Cash withdrawal\n- Utility bill payment"
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("express.md", md_content, "text/markdown")}
    )
    assert upload_res.status_code == 200
    assert upload_res.json()["filename"] == "express.md"
    assert upload_res.json()["chunks_indexed"] > 0


def test_unsupported_document_type():
    """Test uploading an unsupported file type returns 400."""
    exe_content = b"MZ\x90\x00\x03\x00\x00\x00"
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("malicious.exe", exe_content, "application/octet-stream")}
    )
    assert upload_res.status_code == 400
    assert upload_res.json()["error"] is True


def test_document_status_and_deletion():
    """Test document status check and deletion endpoints."""
    txt_content = b"Sample text content for status check and deletion test."
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("temp.txt", txt_content, "text/plain")}
    )
    assert upload_res.status_code == 200
    session_id = upload_res.json()["session_id"]

    # Check status
    status_res = client.get(f"/documents/{session_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["has_document"] is True

    # Delete documents
    del_res = client.delete(f"/documents/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["chunks_deleted"] > 0

    # Status check after delete
    status_after = client.get(f"/documents/{session_id}/status")
    assert status_after.json()["has_document"] is False
