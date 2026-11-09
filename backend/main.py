"""
AwaazDesk - FastAPI Backend Server
Provides comprehensive REST APIs for Grievance Ingestion, ML Classification,
Urgency/Sentiment Analysis, Gnani STT & TTS Integrations, Admin Operations, and Analytics.
"""
import sys
import os
import random
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    BACKEND_HOST,
    BACKEND_PORT,
    UPLOADS_DIR,
    GRIEVANCE_CATEGORIES,
    URGENCY_LEVELS,
    STATUS_OPTIONS,
    SUPPORTED_LANGUAGES
)
from backend.database import (
    init_db,
    create_complaint,
    get_complaint_by_ticket,
    list_complaints,
    update_complaint,
    get_ticket_timeline,
    get_analytics_summary,
    get_user_by_username,
    create_user
)
from backend.models.schemas import (
    ComplaintCreateRequest,
    ComplaintUpdateRequest,
    ComplaintResponse,
    ComplaintPublicResponse,
    mask_phone_number,
    mask_email_address,
    mask_citizen_name,
    TimelinePublicEvent,
    TranscriptionResponse,
    SynthesisRequest,
    SynthesisResponse,
    AnalysisPreviewRequest,
    AnalysisPreviewResponse,
    RetrainResponse,
    LoginRequest,
    TokenResponse,
    LogoutResponse,
    UserProfileResponse,
    CitizenRegisterRequest,
    AdminCreateUserRequest
)
from backend.security.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_user_optional,
    require_admin,
    require_officer_or_admin,
    check_login_rate_limit,
    record_failed_login,
    reset_failed_logins
)
from backend.services.ml_classifier import ml_classifier
from backend.services.urgency_engine import evaluate_urgency
from backend.services.sentiment_engine import analyze_sentiment
from backend.services.summarizer import generate_summary
from backend.services.gnani_stt import gnani_stt
from backend.services.gnani_tts import gnani_tts
from backend.services.gnani_evon import gnani_evon
from backend.training.train_classifier import train_and_export_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("awaazdesk.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database and ensures ML model is ready upon server startup."""
    logger.info("Initializing AwaazDesk database...")
    init_db(seed_sample_data=True)
    if not ml_classifier.is_ready():
        logger.info("Training initial ML model pipeline...")
        try:
            train_and_export_model()
            ml_classifier.load_artifacts()
        except Exception as e:
            logger.error("Failed to train initial ML model: %s", e)
    yield

# Initialize FastAPI App
app = FastAPI(
    title="AwaazDesk API",
    description="AI Voice Grievance Intelligence System - Backend Service",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------
# 1. Health & Diagnostics
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "system": "AwaazDesk - AI Voice Grievance Intelligence System",
        "status": "Operational",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/api/health")
def health_check():
    """Returns real-time status of backend subsystems, ML classifier, and Gnani APIs."""
    stt_status = gnani_stt.get_status_info()
    tts_status = gnani_tts.get_status_info()
    evon_status = gnani_evon.get_status_info()
    return {
        "status": "Healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "Connected (SQLite)",
        "ml_classifier": {
            "ready": ml_classifier.is_ready(),
            "algorithm": "Scikit-Learn TF-IDF + Logistic Regression",
            "attribution": "[ML Model: Scikit-learn TF-IDF + LogisticRegression]"
        },
        "urgency_engine": {
            "ready": True,
            "engine": "Multi-factor Rule-Based Heuristic Engine",
            "attribution": "[Rule-Based Heuristic Engine]"
        },
        "sentiment_engine": {
            "ready": True,
            "engine": "Multilingual Lexical Polarity Engine",
            "attribution": "[Lexical Sentiment Engine]"
        },
        "speech_services": {
            "stt": stt_status,
            "tts": tts_status,
            "evon": evon_status
        },
        "gnani_ai_stack": {
            "prisma_stt": stt_status,
            "timbre_tts": tts_status,
            "evon_indic_llm": evon_status
        }
    }

# ---------------------------------------------------------------------
# 1B. Authentication & Role-Based Access Control
# ---------------------------------------------------------------------
@app.post("/api/auth/login", response_model=TokenResponse)
def login_for_access_token(req: LoginRequest):
    """
    Authenticates username and password against SQLite database with Bcrypt.
    Enforces brute-force rate-limiting and issues a signed JWT access token.
    """
    check_login_rate_limit(req.username.strip())

    user = get_user_by_username(req.username.strip())
    if not user or not verify_password(req.password, user["hashed_password"]):
        record_failed_login(req.username.strip())
        logger.warning("Failed login attempt for username: %s", req.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.get("is_active", 1):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive.",
        )

    # Authentication succeeded; reset any prior failed attempts
    reset_failed_logins(req.username.strip())

    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    user_profile = UserProfileResponse(
        id=user["id"],
        username=user["username"],
        email=user.get("email"),
        full_name=user.get("full_name"),
        role=user["role"],
        is_active=bool(user.get("is_active", 1)),
        created_at=user.get("created_at")
    )
    return TokenResponse(access_token=token, token_type="bearer", user=user_profile)

@app.post("/api/auth/logout", response_model=LogoutResponse)
def logout_user(current_user: dict = Depends(get_current_user)):
    """
    Revokes the current JWT session server-side.
    Adds the token's unique jti to the database revocation blacklist.
    """
    payload = current_user.get("_token_payload", {})
    jti = payload.get("jti")
    exp = payload.get("exp", 0)
    if jti:
        from backend.database import revoke_token
        revoke_token(jti, exp)
        logger.info("Token jti '%s' revoked on logout for user '%s'.", jti, current_user.get("username"))
    return LogoutResponse(success=True, message="Session successfully logged out and revoked.")

@app.get("/api/auth/me", response_model=UserProfileResponse)
def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Returns the profile of the currently authenticated user."""
    return UserProfileResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user.get("email"),
        full_name=current_user.get("full_name"),
        role=current_user["role"],
        is_active=bool(current_user.get("is_active", 1)),
        created_at=current_user.get("created_at")
    )


@app.post("/api/auth/register-citizen", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def register_citizen(req: CitizenRegisterRequest):
    """Allows public registration for citizen accounts (assigned role: 'citizen')."""
    existing = get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken.")
    hashed = hash_password(req.password)
    user = create_user(
        username=req.username,
        email=req.email,
        full_name=req.full_name,
        hashed_password=hashed,
        role="citizen"
    )
    if not user:
        raise HTTPException(status_code=500, detail="Failed to create citizen account.")
    return UserProfileResponse(
        id=user["id"],
        username=user["username"],
        email=user.get("email"),
        full_name=user.get("full_name"),
        role=user["role"],
        is_active=bool(user.get("is_active", 1)),
        created_at=user.get("created_at")
    )

# ---------------------------------------------------------------------
# 2. Voice Services (STT & TTS)
# ---------------------------------------------------------------------
@app.post("/api/voice/transcribe", response_model=TranscriptionResponse)
def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    language: str = Form("hinglish")
):
    """
    Transcribes audio using dedicated Gnani Prisma STT integration.
    If Gnani credentials are not configured, transparently runs in integration placeholder mode.
    """
    try:
        content = file.file.read()
        # Save audio file to uploads directory
        file_ext = Path(file.filename).suffix or ".wav"
        unique_filename = f"audio_{uuid.uuid4().hex[:10]}{file_ext}"
        saved_path = UPLOADS_DIR / unique_filename
        with open(saved_path, "wb") as f:
            f.write(content)

        # Execute transcription via Gnani STT service
        result = gnani_stt.transcribe_audio(
            audio_bytes=content,
            filename=file.filename,
            language_code=language
        )
        return TranscriptionResponse(**result)
    except Exception as e:
        logger.exception("Error in voice transcription endpoint: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Voice transcription service temporarily unavailable. Please retry or enter grievance as text."
        )

@app.post("/api/voice/synthesize", response_model=SynthesisResponse)
def synthesize_speech_endpoint(req: SynthesisRequest):
    """
    Synthesizes speech using dedicated Gnani Timbre TTS integration.
    If Gnani credentials are not configured, transparently returns an audible chime placeholder.
    """
    try:
        result = gnani_tts.synthesize_speech(
            text=req.text,
            language_code=req.language,
            voice_gender=req.voice_gender or "female"
        )
        return SynthesisResponse(**result)
    except Exception as e:
        logger.exception("Error in speech synthesis endpoint: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Speech synthesis service temporarily unavailable. Please retry."
        )

# ---------------------------------------------------------------------
# 3. Live Intelligence Preview (Pre-submission)
# ---------------------------------------------------------------------
@app.post("/api/complaints/preview", response_model=AnalysisPreviewResponse)
def preview_complaint_intelligence(req: AnalysisPreviewRequest):
    """
    Analyzes text in real-time, providing immediate ML classification,
    rule-based urgency, sentiment, and summary with clear model attribution.
    """
    ml_res = ml_classifier.predict_category(req.text)
    urgency_res = evaluate_urgency(req.text)
    sentiment_res = analyze_sentiment(req.text)
    summary_res = generate_summary(req.text)

    return AnalysisPreviewResponse(
        category_ml=ml_res["category_ml"],
        category_confidence=ml_res["category_confidence"],
        category_top_probabilities=ml_res["category_top_probabilities"],
        urgency_level_rule=urgency_res["urgency_level_rule"],
        urgency_score=urgency_res["urgency_score"],
        urgency_triggers=urgency_res["urgency_triggers"],
        sentiment_polarity=sentiment_res["sentiment_polarity"],
        sentiment_score=sentiment_res["sentiment_score"],
        summary=summary_res["summary"],
        model_attribution={
            "category": ml_res["attribution"],
            "urgency": urgency_res["attribution"],
            "sentiment": sentiment_res["attribution"],
            "summary": summary_res["attribution"]
        }
    )

@app.post("/api/evon/analyze")
def evon_analyze_grievance(req: AnalysisPreviewRequest):
    """
    Analyzes citizen grievance using Gnani Evon v3.3 sovereign Indic LLM (Artha stack).
    Extracts deep Indic nuance, root-cause domain categorization, and officer recommendations.
    """
    return gnani_evon.analyze_grievance(req.text, language=req.language or "hinglish")

# ---------------------------------------------------------------------
# 4. Complaints Ingestion & Tracking
# ---------------------------------------------------------------------
@app.post("/api/complaints", response_model=ComplaintResponse, status_code=status.HTTP_201_CREATED)
def submit_complaint(req: ComplaintCreateRequest):
    """
    Submits a grievance, runs ML classification, Rule-based urgency scoring,
    Sentiment analysis, and Extractive summarization, then persists to SQLite.
    """
    # 1. Run Intelligence Pipeline
    ml_res = ml_classifier.predict_category(req.raw_text)
    urgency_res = evaluate_urgency(req.raw_text)
    sentiment_res = analyze_sentiment(req.raw_text)
    summary_res = generate_summary(req.raw_text)

    # 2. Check experimental confidence threshold (0.35) for human triage
    from backend.config import ML_CONFIDENCE_THRESHOLD
    is_low_confidence = bool(ml_res["category_confidence"] < ML_CONFIDENCE_THRESHOLD)

    # 3. Generate Unique Ticket ID (collision-proof)
    while True:
        random_num = random.randint(1000, 9999)
        ticket_id = f"AWA-2026-{random_num}"
        if not get_complaint_by_ticket(ticket_id):
            break

    # Auto department mapping
    dept_map = {
        "Civic Infrastructure & Sanitation": "Municipal Corporation / PWD",
        "Electricity & Power Supply": "Electricity Distribution Board (DISCOM)",
        "Water Supply & Drainage": "Water & Sewerage Supply Board",
        "Public Transport & Railways": "Public Transport Division / Indian Railways",
        "Banking & Financial Services": "Banking Ombudsman & Customer Grievances",
        "Telecommunications & Internet": "Telecom Regulatory & Operator Helpdesk",
        "Healthcare & Civil Supplies": "District Health Department / Food & Civil Supplies",
        "Other Grievances": "General Public Grievance Cell"
    }
    assigned_dept = dept_map.get(ml_res["category_ml"], "General Public Grievance Cell")

    complaint_data = {
        "ticket_id": ticket_id,
        "citizen_name": req.citizen_name.strip(),
        "citizen_phone": req.citizen_phone.strip() if req.citizen_phone else None,
        "citizen_email": req.citizen_email.strip() if req.citizen_email else None,
        "language": req.language,
        "submission_channel": req.submission_channel,
        "original_audio_path": req.original_audio_path,
        "raw_text": req.raw_text.strip(),
        "transcription_confidence": req.transcription_confidence or 1.0,
        "category_ml": ml_res["category_ml"],
        "category_confidence": ml_res["category_confidence"],
        "category_top_probabilities": json.dumps(ml_res["category_top_probabilities"]),
        "urgency_level_rule": urgency_res["urgency_level_rule"],
        "urgency_score": urgency_res["urgency_score"],
        "urgency_triggers": json.dumps(urgency_res["urgency_triggers"]),
        "sentiment_polarity": sentiment_res["sentiment_polarity"],
        "sentiment_score": sentiment_res["sentiment_score"],
        "summary": summary_res["summary"],
        "status": "Open",
        "assigned_department": assigned_dept,
        "admin_notes": None,
        "verified_category": None,
        "verified_by": None,
        "verified_at": None,
        "needs_manual_review": 1 if is_low_confidence else 0
    }

    create_complaint(complaint_data)
    created = get_complaint_by_ticket(ticket_id)
    if not created:
        raise HTTPException(status_code=500, detail="Failed to retrieve created complaint.")

    return ComplaintResponse(**created)

@app.get("/api/complaints")
def get_complaints(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    urgency: Optional[str] = Query(None),
    language: Optional[str] = Query(None),
    search_query: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Retrieves list of complaints with comprehensive filtering.
    Privacy Protection: Masked citizen PII is returned for public/anonymous callers.
    Full PII is returned only for authenticated administrative/officer personnel.
    """
    clean_category = category if isinstance(category, str) else None
    clean_status = status if isinstance(status, str) else None
    clean_urgency = urgency if isinstance(urgency, str) else None
    clean_language = language if isinstance(language, str) else None
    clean_search = search_query if isinstance(search_query, str) else None
    clean_limit = limit if isinstance(limit, int) else 100
    clean_offset = offset if isinstance(offset, int) else 0

    results = list_complaints(
        category=clean_category,
        status=clean_status,
        urgency=clean_urgency,
        language=clean_language,
        search_query=clean_search,
        limit=clean_limit,
        offset=clean_offset
    )

    is_admin_or_officer = False
    if isinstance(current_user, dict) and current_user.get("role") in ["admin", "officer", "superadmin"]:
        is_admin_or_officer = True

    if is_admin_or_officer:
        return [ComplaintResponse(**item) for item in results]

    # Return strict allowlist public schema for anonymous citizens
    # Guarantees that raw_text, internal audio paths, phone, and email are never exposed
    masked_list = []
    for item in results:
        masked_list.append(ComplaintPublicResponse(
            ticket_id=item["ticket_id"],
            citizen_name=mask_citizen_name(item.get("citizen_name")),
            language=item.get("language", "hi"),
            submission_channel=item.get("submission_channel", "text"),
            category_ml=item.get("category_ml", "Other Grievances"),
            category_confidence=round(float(item.get("category_confidence", 0.0)), 2),
            urgency_level_rule=item.get("urgency_level_rule", "Low"),
            urgency_score=round(float(item.get("urgency_score", 0.0)), 2),
            summary=item.get("summary", ""),
            status=item.get("status", "Open"),
            assigned_department=item.get("assigned_department"),
            admin_notes=item.get("admin_notes"),
            created_at=item.get("created_at", ""),
            updated_at=item.get("updated_at", ""),
            verified_category=item.get("verified_category"),
            needs_manual_review=bool(item.get("needs_manual_review", 0))
        ))
    return masked_list

@app.get("/api/complaints/{ticket_id}")
def get_complaint_details(
    ticket_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Retrieves full details for a complaint including timeline audit events.
    Privacy Protection:
    - Anonymous public callers: Contact PII, raw text narrative, and internal audio storage paths are redacted.
    - Timeline audit trail masks the citizen's full legal name to 'Citizen'.
    - Authorized administrative personnel receive complete unmasked operational data.
    """
    complaint = get_complaint_by_ticket(ticket_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")

    timeline = get_ticket_timeline(ticket_id)

    is_admin_or_officer = False
    if isinstance(current_user, dict) and current_user.get("role") in ["admin", "officer", "superadmin"]:
        is_admin_or_officer = True

    if is_admin_or_officer:
        complaint_resp = ComplaintResponse(**complaint)
        timeline_resp = timeline
    else:
        # Strict allowlist schema for public tracking
        complaint_resp = ComplaintPublicResponse(
            ticket_id=complaint["ticket_id"],
            citizen_name=mask_citizen_name(complaint.get("citizen_name")),
            language=complaint.get("language", "hi"),
            submission_channel=complaint.get("submission_channel", "text"),
            category_ml=complaint.get("category_ml", "Other Grievances"),
            category_confidence=round(float(complaint.get("category_confidence", 0.0)), 2),
            urgency_level_rule=complaint.get("urgency_level_rule", "Low"),
            urgency_score=round(float(complaint.get("urgency_score", 0.0)), 2),
            summary=complaint.get("summary", ""),
            status=complaint.get("status", "Open"),
            assigned_department=complaint.get("assigned_department"),
            admin_notes=complaint.get("admin_notes"),
            created_at=complaint.get("created_at", ""),
            updated_at=complaint.get("updated_at", ""),
            verified_category=complaint.get("verified_category"),
            needs_manual_review=bool(complaint.get("needs_manual_review", 0))
        )
        # Sanitize timeline to prevent exposing full citizen name in performed_by
        sanitized_timeline = []
        for ev in timeline:
            ev_dict = dict(ev)
            performed = ev_dict.get("performed_by", "")
            if performed.startswith("Citizen"):
                ev_dict["performed_by"] = "Citizen"
            sanitized_timeline.append(TimelinePublicEvent(**ev_dict))
        timeline_resp = [ev.model_dump() if hasattr(ev, "model_dump") else ev.dict() for ev in sanitized_timeline]

    return {
        "complaint": complaint_resp,
        "timeline": timeline_resp
    }

@app.patch("/api/complaints/{ticket_id}")
def update_complaint_status(
    ticket_id: str,
    updates: ComplaintUpdateRequest,
    current_officer: dict = Depends(require_officer_or_admin)
):
    """
    Updates status, department assignment, admin notes, or human-verified category.
    Protected endpoint: Enforces least privilege (requires Officer or Admin credentials).
    """
    complaint = get_complaint_by_ticket(ticket_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")

    payload = {}
    if updates.status is not None:
        if updates.status not in STATUS_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {STATUS_OPTIONS}")
        payload["status"] = updates.status
    if updates.assigned_department is not None:
        payload["assigned_department"] = updates.assigned_department
    if updates.admin_notes is not None:
        payload["admin_notes"] = updates.admin_notes

    # Human-Verified Category (Officer / Admin Only)
    if updates.verified_category is not None:
        if updates.verified_category not in GRIEVANCE_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid verified category. Must be one of {GRIEVANCE_CATEGORIES}"
            )
        payload["verified_category"] = updates.verified_category
        verifying_actor = (
            current_officer.get("full_name") or current_officer.get("username") or "Officer"
            if isinstance(current_officer, dict) else "Officer"
        )
        payload["verified_by"] = verifying_actor
        payload["verified_at"] = datetime.now().isoformat()
        # Clearing the manual review flag once verified, unless explicitly overridden
        if updates.needs_manual_review is None:
            payload["needs_manual_review"] = 0

    if updates.needs_manual_review is not None:
        payload["needs_manual_review"] = 1 if updates.needs_manual_review else 0

    # Determine actor attribution
    actor = updates.updated_by
    if not actor or actor == "Admin Officer":
        if isinstance(current_officer, dict):
            actor = current_officer.get("full_name") or current_officer.get("username") or "Officer"
        else:
            actor = "Officer"

    success = update_complaint(ticket_id, payload, actor=actor)
    if not success:
        raise HTTPException(status_code=400, detail="No valid update fields provided.")

    updated_data = get_complaint_by_ticket(ticket_id)
    timeline = get_ticket_timeline(ticket_id)
    return {
        "message": f"Ticket {ticket_id} successfully updated.",
        "complaint": ComplaintResponse(**updated_data),
        "timeline": timeline
    }

# ---------------------------------------------------------------------
# 5. Analytics & Metrics
# ---------------------------------------------------------------------
@app.get("/api/analytics")
def get_analytics():
    """Provides aggregated grievance statistics and distribution data for Plotly frontend charts."""
    return get_analytics_summary()

# ---------------------------------------------------------------------
# 6. ML Model Retraining & Evaluation
# ---------------------------------------------------------------------
@app.get("/api/ml/evaluation")
def get_ml_evaluation_report():
    """Returns the empirical evaluation report, classification report, and confusion matrix."""
    report_file = PROJECT_ROOT / "backend" / "models" / "evaluation_report.json"
    if not report_file.exists():
        from backend.training.evaluate_pipeline import run_pipeline_evaluation
        return run_pipeline_evaluation()
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/ml/retrain", response_model=RetrainResponse)
def trigger_retrain(current_admin: dict = Depends(require_admin)):
    """
    Retrains the TF-IDF + Logistic Regression category classifier with current database entries.
    Protected endpoint: Requires valid administrative or officer Bearer token.
    """
    try:
        metrics = train_and_export_model()
        ml_classifier.load_artifacts()
        return RetrainResponse(
            success=True,
            samples_trained=metrics["samples"],
            accuracy_score=round(metrics["train_accuracy"], 4),
            categories=metrics["categories"],
            message=f"Model retrained on {metrics['samples']} samples with 3-fold CV accuracy {round(metrics['cv_accuracy'], 4)}."
        )
    except Exception as e:
        logger.exception("Error retraining ML model: %s", e)
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
