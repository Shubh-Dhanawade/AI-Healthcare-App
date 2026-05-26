"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────
# Auth Schemas
# ─────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ─────────────────────────────────────────
# Document Schemas
# ─────────────────────────────────────────

class ExtractedFieldSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    field_name: str
    field_value: Optional[str] = None
    field_category: Optional[str] = None


class SummarySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    summary_text: str
    coverage_summary: Optional[str] = None
    exclusions_summary: Optional[str] = None
    waiting_period_summary: Optional[str] = None
    premium_summary: Optional[str] = None
    model_used: str
    created_at: datetime


class RiskAnalysisSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    clause_text: str
    risk_type: str
    severity: str
    explanation: Optional[str] = None
    recommendation: Optional[str] = None
    created_at: datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    status: str
    extraction_method: Optional[str] = None
    renewal_date: Optional[datetime] = None
    premium_due_date: Optional[datetime] = None
    safety_score: int = 100
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    extracted_text: Optional[str] = None
    summary: Optional[SummarySchema] = None
    extracted_fields: List[ExtractedFieldSchema] = []
    risk_analyses: List[RiskAnalysisSchema] = []


# ─────────────────────────────────────────
# AI Service Schemas
# ─────────────────────────────────────────

class SummarizeRequest(BaseModel):
    document_id: str


class ExtractFieldsRequest(BaseModel):
    document_id: str


class RiskAnalysisRequest(BaseModel):
    document_id: str


class ExtractedFieldsResponse(BaseModel):
    document_id: str
    fields: List[ExtractedFieldSchema]


class RiskAnalysisResponse(BaseModel):
    document_id: str
    risks: List[RiskAnalysisSchema]
    overall_risk_level: str


class SummarizeResponse(BaseModel):
    document_id: str
    summary: SummarySchema



# ─────────────────────────────────────────
# RAG Query Schemas
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    evaluate: Optional[bool] = False



class EvaluationSchema(BaseModel):
    faithfulness: float
    faithfulness_reasoning: Optional[str] = None
    answer_relevance: float
    answer_relevance_reasoning: Optional[str] = None
    context_relevance: float
    latency: float


class QueryResponse(BaseModel):
    document_id: str
    answer: str
    context: List[str]
    evaluation: EvaluationSchema


# ─────────────────────────────────────────
# Comparison Schemas
# ─────────────────────────────────────────

class CompareRequest(BaseModel):
    document_ids: List[str]


class FeatureWinnerSchema(BaseModel):
    feature: str
    winner: str
    reason: str


class ComparisonSynthesisSchema(BaseModel):
    synthesis: str
    best_for: str
    verdict: str
    feature_winners: List[FeatureWinnerSchema] = []


class CompareResponse(BaseModel):
    documents: List[DocumentDetailResponse]
    comparison_synthesis: ComparisonSynthesisSchema


# ─────────────────────────────────────────
# Chat, Translation & Checklist Schemas
# ─────────────────────────────────────────

class ChatMessageSchema(BaseModel):
    role: str
    content: str


class ChatQueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    history: Optional[List[ChatMessageSchema]] = None


class ChatQueryResponse(BaseModel):
    response: str


class TranslateRequest(BaseModel):
    text: str
    target_language: str


class TranslateResponse(BaseModel):
    translated_text: str


class ClaimsChecklistRequest(BaseModel):
    document_id: str
    treatment_type: str


class ChecklistItemSchema(BaseModel):
    document_name: str
    importance: str
    description: str


class ClaimsChecklistResponse(BaseModel):
    checklist: List[ChecklistItemSchema]
    claim_steps: List[str]
    estimated_approval_days: str


