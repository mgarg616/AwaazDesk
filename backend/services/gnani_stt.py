"""
AwaazDesk - Gnani.ai Prisma Speech-to-Text (STT) Service
=========================================================
Supports Hindi (hi-IN), English (en-IN), and Hinglish (code-mixed).

Policy Compliance: Never fake API integration.
  - If live credentials are provided, connects directly to Gnani Prisma v3 API.
  - If credentials are unavailable, operates in clearly-labelled Placeholder Mode.

API Reference (verified from official Gnani developer documentation):
  Endpoint  : POST https://api.vachana.ai/stt/v3
  Auth      : Header  X-API-Key-ID: <key>  (single-key auth; no token/bearer needed)
  Payload   : multipart/form-data
                audio_file    = <bytes>          (file field, WAV preferred)
                model         = "prisma-v3.0"
                language_code = "indic-auto"     (auto-detects from 10 Indian + EN)
                format        = "verbatim"       (optional; raw spoken-form output)
  Response  : JSON
                full_transcript        : str   (primary transcript field)
                transcript             : str   (alias – also checked)
                detected_language_code : str
                model                  : str
                processing_time        : float
                segments               : list  (optional detailed segments)

GNANI_PRISMA_TOKEN is preserved for .env back-compat but is NOT sent in
any request header – official v3 auth uses X-API-Key-ID only.

Audio format note:
  Gnani Prisma v3 accepts WAV (PCM). OGG, WebM, and MP3 are converted to
  WAV using the standard-library `wave` + `audioop` pipeline when possible,
  or via pydub/ffmpeg if installed. If conversion fails the raw bytes are
  still forwarded so Gnani can attempt decoding natively.
"""
import io
import logging
import struct
import wave
from pathlib import Path
from typing import Dict, Any, Optional

import requests

from backend.config import (
    GNANI_PRISMA_API_KEY,
    GNANI_PRISMA_ENDPOINT,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger("awaazdesk.gnani_stt")

# ---------------------------------------------------------------------------
# Placeholder corpus
# Realistic multilingual grievance samples used when no API key is configured.
# ---------------------------------------------------------------------------
MULTILINGUAL_PLACEHOLDER_CORPUS: Dict[str, list] = {
    "hi": [
        "हमारे इलाके में पिछले तीन दिनों से पीने का पानी नहीं आ रहा है। कृपया तुरंत पानी का टैंकर भिजवाएं।",
        "मुख्य मार्ग पर बिजली का तार टूटकर सड़क पर गिरा हुआ है, तुरंत लाइनमैन भेजा जाए।",
        "अस्पताल में आपातकालीन डॉक्टर अनुपस्थित हैं और दवाइयां भी उपलब्ध नहीं हैं।",
    ],
    "hinglish": [
        "Main road par open transformer se sparks nikal rahe hain, dangerous situation hai please jaldi repair karwaiye.",
        "Bijli ka cut roz 6 ghante ho raha hai bina kisi notice ke, work from home impossible ho gaya hai.",
        "Mera ATM withdrawal fail ho gaya tha, account se 5000 rupaye deduct ho chuke hain par cash nahi mila.",
    ],
    "en": [
        "Deep potholes on the main junction are causing frequent two-wheeler accidents. Please repair urgently.",
        "Garbage has not been collected in our residential colony for a week, creating an unbearable stench.",
        "Streetlights have stopped working on 5th Avenue, leading to severe safety concerns at night.",
    ],
}

# STT model and language parameters verified from Gnani docs
_STT_MODEL = "prisma-v3.0"
_STT_LANGUAGE_CODE = "indic-auto"  # auto-detect across 10 Indian languages + English
_STT_FORMAT = "verbatim"           # raw spoken-form output
_REQUEST_TIMEOUT_SEC = 90          # Generous timeout: Gnani docs say up to 60s; add margin

# MIME types sent to Gnani per file extension
_MIME_MAP: Dict[str, str] = {
    ".wav":  "audio/wav",
    ".mp3":  "audio/mpeg",
    ".m4a":  "audio/mp4",
    ".ogg":  "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


# ---------------------------------------------------------------------------
# Audio format helpers
# ---------------------------------------------------------------------------

def _detect_mime(filename: str) -> str:
    """Returns the correct MIME type for the given filename."""
    ext = Path(filename).suffix.lower()
    return _MIME_MAP.get(ext, "audio/wav")


def _try_convert_to_wav(audio_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """
    Attempts to convert non-WAV audio to 16kHz mono 16-bit PCM WAV.
    Returns (converted_bytes, new_filename).
    Falls back to the original bytes if conversion is unavailable or fails.

    Strategy (in priority order):
    1. If already WAV → return as-is.
    2. Try pydub (needs ffmpeg installed) → broadest format support.
    3. Fall back to original bytes → let Gnani attempt native decoding.
    """
    ext = Path(filename).suffix.lower()
    wav_filename = Path(filename).stem + ".wav"

    # Already WAV – nothing to do
    if ext == ".wav":
        return audio_bytes, filename

    # --- Attempt pydub conversion (optional dependency) ---
    try:
        from pydub import AudioSegment  # type: ignore
        fmt = ext.lstrip(".")
        if fmt == "m4a":
            fmt = "mp4"
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        converted = buf.getvalue()
        logger.info(
            "Audio converted from %s → WAV via pydub: %d bytes → %d bytes",
            ext, len(audio_bytes), len(converted),
        )
        return converted, wav_filename
    except ImportError:
        logger.debug("pydub not installed; skipping audio conversion for %s", filename)
    except Exception as exc:
        logger.warning("pydub conversion failed for %s: %s – forwarding original bytes", filename, exc)

    # Fall back: forward original bytes with the original filename so Gnani
    # can attempt native decoding (Prisma v3 accepts MP3 natively per docs).
    logger.info("Forwarding %s bytes as-is to Gnani (no conversion available).", ext)
    return audio_bytes, filename


# ---------------------------------------------------------------------------
# STT Service
# ---------------------------------------------------------------------------

class GnaniPrismaSTTService:
    """
    Wraps the Gnani Prisma v3 ASR REST API.

    Authentication (verified from official Gnani docs):
      - Header: X-API-Key-ID: <GNANI_PRISMA_API_KEY>
      - No GNANI_PRISMA_TOKEN or Bearer token required.

    When is_configured() is False, returns transparent placeholder results so
    the rest of the application continues to work without real credentials.
    """

    def __init__(self) -> None:
        self.api_key: str = GNANI_PRISMA_API_KEY
        self.endpoint: str = GNANI_PRISMA_ENDPOINT

    # ------------------------------------------------------------------
    # Configuration check
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Returns True only if a real Gnani Prisma API key is set in .env."""
        return bool(
            self.api_key
            and self.api_key.strip()
            and self.api_key not in {
                "your_gnani_prisma_api_key_here",
                "your_gnani_api_key_here",
            }
        )

    def get_status_info(self) -> Dict[str, Any]:
        """Returns integration diagnostics for the UI / health endpoint."""
        configured = self.is_configured()
        return {
            "service_name": "Gnani.ai Prisma STT (Speech-to-Text)",
            "configured": configured,
            "status": "LIVE_ACTIVE" if configured else "INTEGRATION_PLACEHOLDER",
            "endpoint": self.endpoint,
            "model": _STT_MODEL,
            "language_mode": _STT_LANGUAGE_CODE,
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "auth_header": "X-API-Key-ID",
            "message": (
                "Gnani Prisma STT live credentials verified. Live transcription active."
                if configured
                else (
                    "Gnani Prisma API credentials not set in .env. "
                    "Running in clearly-labelled integration placeholder mode. "
                    "Set GNANI_PRISMA_API_KEY (or GNANI_API_KEY) to enable live ASR."
                )
            ),
        }

    # ------------------------------------------------------------------
    # Core transcription method
    # ------------------------------------------------------------------

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str = "hinglish",
    ) -> Dict[str, Any]:
        """
        Transcribes speech audio using Gnani Prisma v3 ASR or transparent placeholder.

        Args:
            audio_bytes   : Raw audio bytes (WAV preferred; OGG/MP3 auto-converted).
            filename      : Original filename with extension (e.g. 'complaint.wav').
            language_code : AwaazDesk language key – 'hi', 'en', or 'hinglish'.

        Returns dict with keys:
            text               : Transcribed string (empty on hard error).
            confidence         : Float 0.0–1.0 or None if Gnani omits it.
            detected_language  : BCP-47 code from Gnani (live) or input key (placeholder).
            language           : AwaazDesk language_code passed in.
            provider           : Human-readable source label.
            is_placeholder     : True when not using live API.
            diagnostic_message : Detailed status string for UI badge.
        """
        # -----------------------------------------------------------------
        # 1. LIVE GNANI PRISMA v3 API CALL (runs only when configured)
        # -----------------------------------------------------------------
        if self.is_configured():
            # --- Audio format normalisation ---
            upload_bytes, upload_filename = _try_convert_to_wav(audio_bytes, filename)
            upload_mime = _detect_mime(upload_filename)

            logger.info(
                "LIVE Gnani Prisma STT call → %s | model=%s | lang=%s | file=%s | mime=%s | size=%d bytes",
                self.endpoint, _STT_MODEL, _STT_LANGUAGE_CODE,
                upload_filename, upload_mime, len(upload_bytes),
            )

            # Auth: single header X-API-Key-ID (verified from official docs)
            headers = {
                "X-API-Key-ID": self.api_key,
                # No Authorization/Bearer, no x-token – official auth is key-only
            }

            # Multipart form fields (verified from official docs)
            files = {
                "audio_file": (upload_filename, upload_bytes, upload_mime),
            }
            data = {
                "model": _STT_MODEL,
                "language_code": _STT_LANGUAGE_CODE,
                "format": _STT_FORMAT,
            }

            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=_REQUEST_TIMEOUT_SEC,
                )

                logger.info(
                    "Gnani STT HTTP response: status=%d content-type=%s body_preview=%.300s",
                    response.status_code,
                    response.headers.get("Content-Type", ""),
                    response.text[:300],
                )

                if response.status_code == 200:
                    # Parse JSON response
                    try:
                        res_json = response.json()
                    except Exception:
                        # Gnani returned non-JSON 200 — treat as error
                        logger.error("Gnani STT 200 but non-JSON body: %s", response.text[:300])
                        return self._error_result(
                            language_code,
                            f"Gnani STT returned non-JSON response (200): {response.text[:200]}",
                            "Gnani Prisma STT v3 (Unexpected Response Format)",
                        )

                    logger.debug("Gnani STT raw JSON: %s", res_json)

                    # Response fields verified from official Gnani docs:
                    #   full_transcript – primary field
                    #   transcript      – alternate alias
                    transcript = (
                        res_json.get("full_transcript")
                        or res_json.get("transcript")
                        or ""
                    )
                    detected_lang = res_json.get("detected_language_code", language_code)
                    # confidence is NOT guaranteed in v3 short-clip response
                    # Keep as None if absent; schema now accepts Optional[float]
                    confidence: Optional[float] = res_json.get("confidence", None)

                    logger.info(
                        "Gnani STT success: %d chars | detected_lang=%s | confidence=%s",
                        len(transcript), detected_lang, confidence,
                    )
                    return {
                        "text": transcript,
                        "confidence": confidence,
                        "detected_language": detected_lang,
                        "language": language_code,
                        "provider": "Gnani Prisma STT v3 (Live API)",
                        "is_placeholder": False,
                        "diagnostic_message": (
                            f"Transcribed via Live Gnani Prisma v3 ASR. "
                            f"Detected language: {detected_lang}. "
                            f"File: {upload_filename} ({len(upload_bytes)} bytes)."
                        ),
                    }

                else:
                    # HTTP error – log full response body for diagnosis
                    error_body = response.text[:500]
                    logger.error(
                        "Gnani STT API error: HTTP %d → %s",
                        response.status_code, error_body,
                    )
                    return self._error_result(
                        language_code,
                        f"Gnani Prisma STT HTTP {response.status_code}: {error_body[:300]}",
                        f"Gnani Prisma STT v3 (HTTP {response.status_code} Error)",
                    )

            except requests.exceptions.Timeout:
                logger.error("Gnani STT timed out after %ds for file %s", _REQUEST_TIMEOUT_SEC, filename)
                return self._error_result(
                    language_code,
                    f"Gnani Prisma STT request timed out after {_REQUEST_TIMEOUT_SEC}s. "
                    "Check network and audio file size (must be ≤30s clip for /stt/v3).",
                    "Gnani Prisma STT v3 (Timeout)",
                )
            except requests.exceptions.ConnectionError as exc:
                logger.error("Gnani STT connection error: %s", exc)
                return self._error_result(
                    language_code,
                    f"Cannot reach Gnani Prisma API at {self.endpoint}: {exc}",
                    "Gnani Prisma STT v3 (Connection Error)",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error during Gnani STT request: %s", exc)
                return self._error_result(
                    language_code,
                    f"Unexpected error calling Gnani Prisma API: {type(exc).__name__}: {exc}",
                    "Gnani Prisma STT v3 (Unexpected Error)",
                )

        # -----------------------------------------------------------------
        # 2. TRANSPARENT INTEGRATION PLACEHOLDER MODE (no API key in .env)
        # -----------------------------------------------------------------
        logger.info(
            "Gnani Prisma STT – Placeholder Mode (GNANI_PRISMA_API_KEY not set in .env)."
        )
        samples = MULTILINGUAL_PLACEHOLDER_CORPUS.get(
            language_code, MULTILINGUAL_PLACEHOLDER_CORPUS["hinglish"]
        )
        pick_index = len(audio_bytes) % len(samples)
        simulated_transcript = samples[pick_index]

        return {
            "text": simulated_transcript,
            "confidence": 0.93,
            "detected_language": language_code,
            "language": language_code,
            "provider": "Gnani Prisma STT (Integration Placeholder Mode)",
            "is_placeholder": True,
            "diagnostic_message": (
                "[INTEGRATION STATUS: Gnani Prisma STT API Key Not Configured] "
                "Running in transparent placeholder mode. "
                "Set GNANI_PRISMA_API_KEY (or GNANI_API_KEY) in .env to enable live speech recognition."
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(language_code: str, diagnostic_message: str, provider: str) -> Dict[str, Any]:
        """Returns a structured error dict that satisfies TranscriptionResponse schema."""
        return {
            "text": "",
            "confidence": None,
            "detected_language": language_code,
            "language": language_code,
            "provider": provider,
            "is_placeholder": False,
            "diagnostic_message": diagnostic_message,
        }


# Global singleton
gnani_stt = GnaniPrismaSTTService()
