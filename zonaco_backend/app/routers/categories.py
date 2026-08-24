"""Router for FAQ category browsing, predefined question queries, and privacy policy."""

from fastapi import APIRouter, Depends
from app.dependencies import get_excel_parser_dep
from app.schemas.categories import (
    CategoryItem,
    CategoryListResponse,
    PrivacyPolicyResponse,
    QuestionItem,
    QuestionListResponse,
)
from app.services.excel_parser import ExcelFAQParser
from app.utils.exceptions import CategoryNotFoundException

router = APIRouter(tags=["Knowledge Base Browsing"])


@router.get(
    "/categories",
    response_model=CategoryListResponse,
    summary="List FAQ categories",
    description="Returns the available top-level FAQ categories and their question counts."
)
async def list_categories(
    parser: ExcelFAQParser = Depends(get_excel_parser_dep)
) -> CategoryListResponse:
    """Retrieve all FAQ categories from Excel workbook."""
    overview = parser.get_categories_overview()
    items = [
        CategoryItem(
            name=cat["name"],
            question_count=cat["question_count"],
            description=cat["description"]
        )
        for cat in overview
    ]
    return CategoryListResponse(categories=items)


@router.get(
    "/categories/{category}/questions",
    response_model=QuestionListResponse,
    summary="List questions in category",
    description="Direct lookup of all predefined FAQ questions under a specific category."
)
async def list_questions_by_category(
    category: str,
    parser: ExcelFAQParser = Depends(get_excel_parser_dep)
) -> QuestionListResponse:
    """Retrieve browsable FAQ questions for a given category."""
    questions = parser.get_questions_by_category(category)
    if not questions:
        raise CategoryNotFoundException(category)

    question_items = [
        QuestionItem(
            id=entry.doc_id,
            question=entry.question,
            intent=entry.faq_intent,
            type_tag=entry.topic_type,
            links=entry.links
        )
        for entry in questions
    ]
    return QuestionListResponse(category=category.upper(), questions=question_items)


@router.get(
    "/privacy-policy",
    response_model=PrivacyPolicyResponse,
    summary="Zanaco privacy policy",
    description="Returns official Zanaco Customer Privacy and Data Protection statement."
)
async def get_privacy_policy() -> PrivacyPolicyResponse:
    """Return static Zanaco data protection and privacy policy."""
    policy_content = (
        "Zambia National Commercial Bank Plc (Zanaco) is committed to safeguarding your privacy and protecting "
        "your personal and financial data. We collect and process your information solely to provide banking services, "
        "comply with legal and regulatory obligations under the Data Protection Act No. 3 of 2021, and enhance your "
        "customer support experience. We do not sell your personal information to third parties, and all interactions "
        "are secured with industry-standard encryption protocols."
    )
    return PrivacyPolicyResponse(
        title="Zanaco Customer Privacy & Data Protection Policy",
        content=policy_content,
        links=["https://www.zanaco.co.zm/privacy-policy"]
    )
