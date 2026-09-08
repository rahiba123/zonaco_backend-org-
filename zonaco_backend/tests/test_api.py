"""Automated test suite for Zanaco FAQ Chatbot Backend."""

import os
from unittest.mock import AsyncMock, patch
try:
    import pytest
except ImportError:
    class _PytestMock:
        def raises(self, exc):
            import contextlib
            @contextlib.contextmanager
            def _cm():
                try:
                    yield
                except exc:
                    pass
                else:
                    raise AssertionError(f"Expected exception {exc} was not raised.")
            return _cm()
    pytest = _PytestMock()
from fastapi.testclient import TestClient
from app.main import app
from app.services.excel_parser import ExcelFAQParser
from app.services.rag import RAGResponse
from app.state_machine import SessionState, StateMachine
from app.utils.exceptions import InvalidStateTransitionException

client = TestClient(app)


def test_state_machine_valid_transitions():
    """Test standard state machine transitions adhere to conversation flow."""
    # WELCOME -> MAIN_MENU
    next_state = StateMachine.validate_and_transition(
        SessionState.WELCOME, SessionState.MAIN_MENU, "start_menu"
    )
    assert next_state == SessionState.MAIN_MENU

    # MAIN_MENU -> CATEGORY_SELECT
    next_state = StateMachine.validate_and_transition(
        SessionState.MAIN_MENU, SessionState.CATEGORY_SELECT, "choose_categories"
    )
    assert next_state == SessionState.CATEGORY_SELECT

    # CATEGORY_SELECT -> ANSWERED
    next_state = StateMachine.validate_and_transition(
        SessionState.CATEGORY_SELECT, SessionState.ANSWERED, "ask_question"
    )
    assert next_state == SessionState.ANSWERED

    # ANSWERED -> SATISFACTION_CHECK
    next_state = StateMachine.validate_and_transition(
        SessionState.ANSWERED, SessionState.SATISFACTION_CHECK, "prompt_satisfaction"
    )
    assert next_state == SessionState.SATISFACTION_CHECK

    # SATISFACTION_CHECK -> MORE_QUESTIONS_CHECK
    next_state = StateMachine.validate_and_transition(
        SessionState.SATISFACTION_CHECK, SessionState.MORE_QUESTIONS_CHECK, "satisfaction_yes"
    )
    assert next_state == SessionState.MORE_QUESTIONS_CHECK

    # MORE_QUESTIONS_CHECK -> RATING
    next_state = StateMachine.validate_and_transition(
        SessionState.MORE_QUESTIONS_CHECK, SessionState.RATING, "more_questions_no"
    )
    assert next_state == SessionState.RATING

    # RATING -> END
    next_state = StateMachine.validate_and_transition(
        SessionState.RATING, SessionState.END, "submit_rating"
    )
    assert next_state == SessionState.END


def test_state_machine_invalid_transitions():
    """Test that illegal state jumps raise InvalidStateTransitionException."""
    with pytest.raises(InvalidStateTransitionException):
        StateMachine.validate_and_transition(
            SessionState.WELCOME, SessionState.END, "illegal_jump"
        )

    with pytest.raises(InvalidStateTransitionException):
        StateMachine.validate_and_transition(
            SessionState.END, SessionState.MAIN_MENU, "cannot_leave_terminal_end"
        )


def test_excel_parser_sheets_and_links():
    """Test Excel parser extracts rows and URLs accurately."""
    parser = ExcelFAQParser()
    if os.path.exists(parser.file_path):
        entries = parser.parse_all()
        assert len(entries) > 0

        categories = set(e.category for e in entries)
        assert "CARD SERVICES" in categories or "GENERAL BANKING" in categories

        # Test link extraction logic
        test_url = "For more information visit https://www.zanaco.co.zm/branches or call."
        extracted = ExcelFAQParser.extract_urls(test_url)
        assert "https://www.zanaco.co.zm/branches" in extracted


def test_health_endpoint():
    """Test GET /health returns operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert data["vector_store_initialized"] is True


def test_privacy_policy_endpoint():
    """Test GET /privacy-policy returns official policy text."""
    response = client.get("/privacy-policy")
    assert response.status_code == 200
    data = response.json()
    assert "Zanaco" in data["title"]
    assert len(data["links"]) > 0


def test_categories_and_questions_endpoints():
    """Test GET /categories and GET /categories/{category}/questions."""
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) > 0

    first_cat = data["categories"][0]["name"]
    q_res = client.get(f"/categories/{first_cat}/questions")
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["category"] == first_cat
    assert len(q_data["questions"]) > 0


def test_invalid_state_transition_http_error():
    """Test calling /chat/satisfaction before answering any question returns HTTP 400."""
    start_res = client.post("/session/start", json={})
    session_id = start_res.json()["session_id"]

    # Calling /chat/satisfaction from MAIN_MENU should fail with 400
    sat_res = client.post("/chat/satisfaction", json={"session_id": session_id, "satisfied": True})
    assert sat_res.status_code == 400
    data = sat_res.json()
    assert data["error"] is True
    assert "Cannot perform 'submit_satisfaction' in state 'MAIN_MENU'" in data["message"]


@patch("app.services.rag.RAGService.answer_question")
def test_full_conversation_lifecycle(mock_rag_answer):
    """Test complete flow: start -> ask -> satisfaction -> more questions -> rate -> history."""
    # Mock RAG response
    mock_rag_answer.return_value = RAGResponse(
        answer="To apply for a debit card, visit your nearest Zanaco branch or apply via the app.",
        matched_faq_intent="Card Application",
        links=["https://www.zanaco.co.zm/cards"],
        confidence_score=0.92,
        should_offer_escalation=False
    )

    # 1. Start Session
    start_res = client.post("/session/start", json={"user_id": "test_user_123"})
    assert start_res.status_code == 201
    start_data = start_res.json()
    session_id = start_data["session_id"]
    assert start_data["current_state"] == "MAIN_MENU"

    # 2. Ask Question
    ask_res = client.post(
        "/chat/ask",
        json={
            "session_id": session_id,
            "category": "CARD SERVICES",
            "question": "How do I apply for a debit card?"
        }
    )
    assert ask_res.status_code == 200
    ask_data = ask_res.json()
    assert ask_data["current_state"] == "ANSWERED"
    assert ask_data["confidence_score"] == 0.92
    assert "https://www.zanaco.co.zm/cards" in ask_data["links"]

    # 3. Submit Satisfaction (YES)
    sat_res = client.post("/chat/satisfaction", json={"session_id": session_id, "satisfied": True})
    assert sat_res.status_code == 200
    sat_data = sat_res.json()
    assert sat_data["current_state"] == "MORE_QUESTIONS_CHECK"

    # 4. Submit More Questions (NO -> moves to RATING because satisfied is True)
    more_res = client.post("/chat/more-questions", json={"session_id": session_id, "more": False})
    assert more_res.status_code == 200
    more_data = more_res.json()
    assert more_data["current_state"] == "RATING"

    # 5. Submit Rating & Feedback
    rate_res = client.post(
        "/chat/rate",
        json={"session_id": session_id, "rating": 5, "feedback_text": "Very quick and helpful!"}
    )
    assert rate_res.status_code == 200
    rate_data = rate_res.json()
    assert rate_data["current_state"] == "END"

    # 6. Fetch Complete History
    hist_res = client.get(f"/session/{session_id}/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["session_id"] == session_id
    assert hist_data["current_state"] == "END"
    assert hist_data["is_satisfied"] is True
    assert hist_data["rating"] == 5
    assert hist_data["feedback"] == "Very quick and helpful!"
    assert len(hist_data["messages"]) >= 5


def test_agent_escalation_flow():
    """Test manual live agent escalation."""
    # 1. Start Session
    start_res = client.post("/session/start", json={})
    session_id = start_res.json()["session_id"]

    # 2. Escalate directly from MAIN_MENU
    esc_res = client.post("/chat/escalate", json={"session_id": session_id})
    assert esc_res.status_code == 200
    esc_data = esc_res.json()
    assert esc_data["current_state"] == "LIVE_AGENT"
    assert "agent_queue_ticket" in esc_data
    assert "ZNCO-ESC-" in esc_data["agent_queue_ticket"]


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
    assert ask_data["should_offer_escalation"] is False


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


