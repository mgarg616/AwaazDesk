"""
AwaazDesk - Configuration & Environment Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Server Settings
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Database Path
DB_REL_PATH = os.getenv("SQLITE_DB_PATH", "data/awaazdesk.db")
DB_PATH = BASE_DIR / DB_REL_PATH
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Uploads Path
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ML Models Directory & Artifact Paths
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Production Artifacts (Full-data refit for live inference)
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

# 2. Evaluation Artifacts (Fitted strictly on X_train N=90; test set never seen)
EVALUATION_MODEL_PATH = MODELS_DIR / "evaluation_model.joblib"
EVALUATION_VECTORIZER_PATH = MODELS_DIR / "evaluation_vectorizer.joblib"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"

# Shared ML Pipeline Hyperparameters (Consistent across Evaluation and Production)
ML_MAX_FEATURES = 5000
ML_NGRAM_RANGE = (1, 2)
ML_SUBLINEAR_TF = True
ML_TOKEN_PATTERN = r"(?u)[\w\u0900-\u097F]+"
ML_LOGISTIC_REG_C = 6.0
ML_LOGISTIC_REG_MAX_ITER = 1000
ML_LOGISTIC_REG_CLASS_WEIGHT = "balanced"
ML_RANDOM_STATE = 42

# -------------------------------------------------------------------------
# Gnani.ai API Settings
# Official base URL (api.vachana.ai) verified from Gnani developer docs.
# A single API key (GNANI_API_KEY) with STT+TTS scopes covers both services.
# GNANI_PRISMA_TOKEN is preserved for .env back-compat but is NOT used in
# the corrected request (official auth is X-API-Key-ID header only).
# -------------------------------------------------------------------------

# Single shared API key for both STT and TTS (same key, two scopes).
# Individual overrides (GNANI_PRISMA_API_KEY / GNANI_TIMBRE_API_KEY) are
# still supported so users who set them separately don't need to change .env.
_GNANI_API_KEY_SHARED = os.getenv("GNANI_API_KEY", "").strip()

# Gnani.ai Prisma STT Settings
GNANI_PRISMA_API_KEY = (
    os.getenv("GNANI_PRISMA_API_KEY", "").strip()
    or _GNANI_API_KEY_SHARED
)
GNANI_PRISMA_TOKEN = os.getenv("GNANI_PRISMA_TOKEN", "").strip()   # Legacy – not used in v3 auth
GNANI_PRISMA_ENDPOINT = os.getenv(
    "GNANI_PRISMA_ENDPOINT",
    "https://api.vachana.ai/stt/v3"           # Verified endpoint (Prisma v3)
).strip()

# Gnani.ai Timbre TTS Settings
GNANI_TIMBRE_API_KEY = (
    os.getenv("GNANI_TIMBRE_API_KEY", "").strip()
    or _GNANI_API_KEY_SHARED
)
GNANI_TIMBRE_ENDPOINT = os.getenv(
    "GNANI_TIMBRE_ENDPOINT",
    "https://api.vachana.ai/api/v1/tts/inference"  # Verified endpoint (Timbre v3)
).strip()

# Gnani.ai Evon (Artha Sovereign Indic LLM Stack) Settings
GNANI_EVON_API_KEY = (
    os.getenv("GNANI_EVON_API_KEY", "").strip()
    or _GNANI_API_KEY_SHARED
)
GNANI_EVON_ENDPOINT = os.getenv(
    "GNANI_EVON_ENDPOINT",
    "https://api.vachana.ai/v1/chat/completions"  # Standard / vLLM OpenAI-compatible endpoint
).strip()
GNANI_EVON_MODEL = os.getenv(
    "GNANI_EVON_MODEL",
    "gnani-evon-v3.3"
).strip()

# Supported Languages
# gnani_code is the BCP-47 locale passed to TTS; STT uses indic-auto for auto-detection.
SUPPORTED_LANGUAGES = {
    "hi": {"name": "Hindi", "gnani_code": "hi-IN", "display": "हिन्दी (Hindi)"},
    "en": {"name": "English", "gnani_code": "en-IN", "display": "English (India)"},
    "hinglish": {"name": "Hinglish", "gnani_code": "hi-IN", "display": "Hinglish (Code-Mixed)"}
}

# Grievance Categories
GRIEVANCE_CATEGORIES = [
    "Civic Infrastructure & Sanitation",
    "Electricity & Power Supply",
    "Water Supply & Drainage",
    "Public Transport & Railways",
    "Banking & Financial Services",
    "Telecommunications & Internet",
    "Healthcare & Civil Supplies",
    "Other Grievances"
]

# ML Confidence Threshold for Human-in-the-Loop Triage
# Experimental initial threshold (0.35) for routing to officer manual review.
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.35"))

# Urgency Levels
URGENCY_LEVELS = ["Critical", "High", "Medium", "Low"]

# Status Options
STATUS_OPTIONS = [
    "Open",
    "In Progress",
    "Under Investigation",
    "Resolved",
    "Closed"
]

# -------------------------------------------------------------------------
# Security & JWT Authentication Settings
# -------------------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256").strip()
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))  # Default: 60 minutes

def validate_jwt_secret(key: str = None) -> str:
    """
    Enforces that JWT_SECRET_KEY is present and meets minimum 32-character entropy.
    Fails safely with a descriptive ValueError if missing, empty, or weak.
    """
    secret = (key if key is not None else JWT_SECRET_KEY).strip()
    if not secret:
        raise ValueError(
            "CRITICAL SECURITY CONFIGURATION ERROR: JWT_SECRET_KEY is missing or empty. "
            "Configure a cryptographically random secret of at least 32 characters in your .env file."
        )
    if len(secret) < 32:
        raise ValueError(
            f"CRITICAL SECURITY CONFIGURATION ERROR: JWT_SECRET_KEY is too weak ({len(secret)} characters). "
            "Must be at least 32 characters long."
        )
    return secret

# Optional Bootstrap Admin (Used ONLY if explicitly provided in environment on fresh setup)
ADMIN_INITIAL_USERNAME = os.getenv("ADMIN_INITIAL_USERNAME", "").strip()
ADMIN_INITIAL_PASSWORD = os.getenv("ADMIN_INITIAL_PASSWORD", "").strip()


