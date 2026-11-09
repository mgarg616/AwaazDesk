"""
AwaazDesk - Pydantic Request & Response Data Models
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ComplaintCreateRequest(BaseModel):
    citizen_name: str = Field(..., description="Full name of the citizen")
    citizen_phone: Optional[str] = Field(None, description="Contact phone number")
    citizen_email: Optional[str] = Field(None, description="Contact email address")
    language: str = Field(default="hinglish", description="Language code: hi, en, or hinglish")
    submission_channel: str = Field(default="text", description="Submission channel: voice or text")
    raw_text: str = Field(..., min_length=5, description="Full complaint text (either typed or transcribed)")
    transcription_confidence: Optional[float] = Field(default=1.0, description="STT confidence score if voice channel")
    original_audio_path: Optional[str] = Field(None, description="Relative path to stored audio file if voice")

class ComplaintUpdateRequest(BaseModel):
    status: Optional[str] = Field(None, description="New status")
    assigned_department: Optional[str] = Field(None, description="Department handling grievance")
    admin_notes: Optional[str] = Field(None, description="Resolution or follow-up remarks")
    updated_by: Optional[str] = Field(default="Admin Officer", description="Actor performing modification")
    verified_category: Optional[str] = Field(None, description="Human-verified grievance category by authorized officer")
    needs_manual_review: Optional[bool] = Field(None, description="Manual review triage flag")

def mask_citizen_name(name: Optional[str]) -> str:
    """Masks a citizen name to prevent identity exposure, e.g. 'Rajesh Kumar Sharma' -> 'R**** S****'."""
    if not name:
        return "Citizen"
    parts = [p.strip() for p in name.strip().split() if p.strip()]
    if not parts:
        return "Citizen"
    if len(parts) == 1:
        w = parts[0]
        return f"{w[0]}****"
    return f"{parts[0][0]}**** {parts[-1][0]}****"

def mask_phone_number(phone: Optional[str]) -> Optional[str]:
    """Masks a citizen phone number, e.g. '+91 98*****10'."""
    if not phone:
        return None
    phone = phone.strip()
    if len(phone) <= 4:
        return "****"
    visible_prefix = phone[:5]
    visible_suffix = phone[-2:]
    return f"{visible_prefix}*****{visible_suffix}"

def mask_email_address(email: Optional[str]) -> Optional[str]:
    """Masks a citizen email address, e.g. 'r***h@example.com'."""
    if not email or "@" not in email:
        return None
    parts = email.strip().split("@", 1)
    local, domain = parts[0], parts[1]
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[0] + "***" + local[-1]
    return f"{masked_local}@{domain}"

class ComplaintResponse(BaseModel):
    id: int
    ticket_id: str
    citizen_name: str
    citizen_phone: Optional[str] = None
    citizen_email: Optional[str] = None
    language: str
    submission_channel: str
    original_audio_path: Optional[str] = None
    raw_text: str
    transcription_confidence: float
    category_ml: str
    category_confidence: float
    category_top_probabilities: Optional[Dict[str, float]] = None
    urgency_level_rule: str
    urgency_score: float
    urgency_triggers: Optional[List[str]] = None
    sentiment_polarity: str
    sentiment_score: float
    summary: str
    status: str
    assigned_department: Optional[str] = None
    admin_notes: Optional[str] = None
    created_at: str
    updated_at: str
    verified_category: Optional[str] = None
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    needs_manual_review: bool = False

class ComplaintPublicResponse(BaseModel):
    """
    Strict Allowlist Public Response Schema.
    Guarantees that raw private complaint text, internal audio storage paths,
    and citizen contact PII can never accidentally leak to anonymous callers.
    """
    ticket_id: str
    citizen_name: str
    language: str
    submission_channel: str
    category_ml: str
    category_confidence: float
    urgency_level_rule: str
    urgency_score: float
    summary: str
    status: str
    assigned_department: Optional[str] = None
    admin_notes: Optional[str] = None
    created_at: str
    updated_at: str
    verified_category: Optional[str] = None
    needs_manual_review: bool = False

    # Explicitly redacted in all public views:
    citizen_phone: Optional[str] = None
    citizen_email: Optional[str] = None
    raw_text: Optional[str] = None
    original_audio_path: Optional[str] = None

class TimelinePublicEvent(BaseModel):
    """Timeline event sanitized for public ticket tracking."""
    id: int
    ticket_id: str
    action: str
    performed_by: str
    details: Optional[str] = None
    timestamp: str



class TranscriptionResponse(BaseModel):
    text: str
    confidence: Optional[float] = None   # Gnani v3 may omit confidence; never None-crash
    detected_language: Optional[str] = None  # BCP-47 detected lang returned by Gnani live
    language: str
    provider: str
    is_placeholder: bool
    diagnostic_message: str

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=2, description="Text to synthesize into speech")
    language: str = Field(default="hi", description="Language code: hi, en, or hinglish")
    voice_gender: Optional[str] = Field(default="female", description="Voice persona: female or male")

class SynthesisResponse(BaseModel):
    audio_base64: str
    audio_format: str = "audio/wav"
    provider: str
    is_placeholder: bool
    diagnostic_message: str

class AnalysisPreviewRequest(BaseModel):
    text: str = Field(..., min_length=5)
    language: Optional[str] = "hinglish"

class AnalysisPreviewResponse(BaseModel):
    category_ml: str
    category_confidence: float
    category_top_probabilities: Dict[str, float]
    urgency_level_rule: str
    urgency_score: float
    urgency_triggers: List[str]
    sentiment_polarity: str
    sentiment_score: float
    summary: str
    model_attribution: Dict[str, str]

class RetrainResponse(BaseModel):
    success: bool
    samples_trained: int
    accuracy_score: float
    categories: List[str]
    message: str

# -------------------------------------------------------------------------
# Authentication & User Management Schemas
# -------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Account password")

class UserProfileResponse(BaseModel):
    id: Optional[int] = None
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    is_active: bool = True
    created_at: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse

class CitizenRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: Optional[str] = None
    email: Optional[str] = None

class AdminCreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: Optional[str] = None
    email: Optional[str] = None
    role: str = Field(default="officer", description="Role: admin, officer, or citizen")

class LogoutResponse(BaseModel):
    success: bool
    message: str


