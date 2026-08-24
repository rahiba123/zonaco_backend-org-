"""Schemas for category browsing, predefined questions, and policy content."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CategoryItem(BaseModel):
    """Category descriptor."""
    name: str = Field(..., description="Category title matching Excel sheet name")
    question_count: int = Field(..., description="Total FAQ entries available in this category")
    description: Optional[str] = Field(default=None, description="Brief description of the category scope")


class CategoryListResponse(BaseModel):
    """Response containing available FAQ categories."""
    categories: List[CategoryItem] = Field(..., description="List of top-level categories")


class QuestionItem(BaseModel):
    """Individual predefined FAQ question for browsing."""
    id: str = Field(..., description="Unique question reference ID")
    question: str = Field(..., description="Question title / prompt")
    intent: Optional[str] = Field(default=None, description="Associated FAQ intent tag")
    type_tag: Optional[str] = Field(default=None, description="Internal topic type tag")
    links: Optional[List[str]] = Field(default=None, description="Extracted reference links")


class QuestionListResponse(BaseModel):
    """Response containing browsable questions for a specific category."""
    category: str = Field(..., description="Category name")
    questions: List[QuestionItem] = Field(..., description="List of questions in this category")


class PrivacyPolicyResponse(BaseModel):
    """Zanaco Data Privacy Policy content."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Policy body text")
    links: List[str] = Field(default=[], description="Official privacy links")
