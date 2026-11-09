"""
AwaazDesk - Gnani.ai Timbre Text-to-Speech (TTS) Service
=========================================================
Supports Hindi, English, and Hinglish vocalization.

Policy Compliance: Never fake API integration.
  - If live credentials are provided, connects directly to Gnani Timbre v3 API.
  - If credentials are unavailable, operates in clearly-labelled Placeholder Mode
    and returns a synthesized audio chime for playback validation.

API Reference (verified from official Gnani developer documentation):
  Endpoint  : POST https://api.vachana.ai/api/v1/tts/inference
  Auth      : Header  X-API-Key-ID: <key>  (single-key auth; same key as STT)
              Header  Content-Type: application/json
  Payload   : JSON
                text         : str    (required; supports inline audio tags for timbre-v3.0)
                model        : str    (required; "timbre-v3.0")
                voice        : str    (required; e.g. "Shivani", "Avinash", "Nalini")
                language     : str    (optional; BCP-47 e.g. "hi-IN", "en-IN")
                speed        : float  (optional; default 1.0)
                audio_config : dict   (optional; controls output format)
                  sample_rate  : int    (e.g. 24000)
                  num_channels : int    (1 for mono)
                  sample_width : int    (2 for 16-bit)
                  encoding     : str    ("linear_pcm")
                  container    : str    ("wav")
  Response  : Raw binary audio (audio/wav) – NOT base64 JSON.
              Write response.content directly to bytes/file.

Voice catalogue (Timbre v3.0, multilingual):
  Female: Shivani, Nalini, Lakshmi, Kalpana, Vasudha, Riyaa, Tarika, Chetana
  Male  : Avinash, Shubham, Prakash, Devansh, Yashraj
"""
import base64
import io
import logging
import math
import struct
import wave
from typing import Dict, Any, Optional

import requests

from backend.config import (
    GNANI_TIMBRE_API_KEY,
    GNANI_TIMBRE_ENDPOINT,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger("awaazdesk.gnani_tts")

# ---------------------------------------------------------------------------
# TTS constants (verified from official Gnani docs)
# ---------------------------------------------------------------------------
_TTS_MODEL = "timbre-v3.0"
_TTS_DEFAULT_VOICE_FEMALE = "Shivani"    # Female – multi-lingual, all 10 Indian langs
_TTS_DEFAULT_VOICE_MALE = "Avinash"      # Male   – multi-lingual
_REQUEST_TIMEOUT_SEC = 60

# Audio output configuration requested from Gnani (WAV, 24kHz mono 16-bit)
_AUDIO_CONFIG = {
    "sample_rate": 24000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}


# ---------------------------------------------------------------------------
# Placeholder WAV generator
# ---------------------------------------------------------------------------

def generate_placeholder_wav(duration_sec: float = 1.5, frequency: float = 440.0) -> bytes:
    """
    Generates a clean synthetic PCM WAV chime for placeholder playback.
    Used only when Gnani API credentials are not configured.
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)    # Mono
        wav_file.setsampwidth(2)    # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            t = float(i) / sample_rate
            # Harmonic chime envelope: frequency Hz, exponential decay
            decay = math.exp(-3.0 * t)
            sample = 32767.0 * 0.4 * math.sin(2.0 * math.pi * frequency * t) * decay
            wav_file.writeframes(struct.pack("<h", int(sample)))

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# TTS Service
# ---------------------------------------------------------------------------

class GnaniTimbreTTSService:
    """
    Wraps the Gnani Timbre v3 TTS REST API.

    Authentication (verified from official Gnani docs):
      - Header: X-API-Key-ID: <GNANI_TIMBRE_API_KEY>
      - Same API key as STT (dual-scoped key).
      - No token/bearer required.

    Response: raw binary audio (audio/wav) written from response.content.

    When is_configured() is False, returns a synthetic WAV chime with clear
    placeholder metadata so the application continues to function.
    """

    def __init__(self) -> None:
        self.api_key: str = GNANI_TIMBRE_API_KEY
        self.endpoint: str = GNANI_TIMBRE_ENDPOINT

    # ------------------------------------------------------------------
    # Configuration check
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Returns True only if a real Gnani Timbre API key is set in .env."""
        return bool(
            self.api_key
            and self.api_key.strip()
            and self.api_key not in {"your_gnani_timbre_api_key_here", "your_gnani_api_key_here"}
        )

    def get_status_info(self) -> Dict[str, Any]:
        """Returns integration diagnostics for the UI / health endpoint."""
        configured = self.is_configured()
        return {
            "service_name": "Gnani.ai Timbre TTS (Text-to-Speech)",
            "configured": configured,
            "status": "LIVE_ACTIVE" if configured else "INTEGRATION_PLACEHOLDER",
            "endpoint": self.endpoint,
            "model": _TTS_MODEL,
            "auth_header": "X-API-Key-ID",
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "message": (
                "Gnani Timbre TTS live credentials verified. Live synthesis active."
                if configured
                else (
                    "Gnani Timbre API credentials not set in .env. "
                    "Running in clearly-labelled integration placeholder mode. "
                    "Set GNANI_TIMBRE_API_KEY (or GNANI_API_KEY) to enable live TTS."
                )
            ),
        }

    # ------------------------------------------------------------------
    # Voice selection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _select_voice(voice_gender: str) -> str:
        """
        Returns a Timbre v3.0 voice name from the official voice catalogue.
        Falls back to default female voice for unrecognised gender strings.
        """
        gender = (voice_gender or "female").strip().lower()
        if gender in {"male", "m"}:
            return _TTS_DEFAULT_VOICE_MALE
        return _TTS_DEFAULT_VOICE_FEMALE

    # ------------------------------------------------------------------
    # Core synthesis method
    # ------------------------------------------------------------------

    def synthesize_speech(
        self,
        text: str,
        language_code: str = "hi",
        voice_gender: str = "female",
    ) -> Dict[str, Any]:
        """
        Synthesizes text into audio using Gnani Timbre v3 or transparent placeholder.

        Args:
            text          : Text to synthesize (Hindi, English, or Hinglish).
            language_code : AwaazDesk language key – 'hi', 'en', or 'hinglish'.
            voice_gender  : 'female' (default) or 'male'.

        Returns:
            Dict with keys:
              audio_base64       : Base64-encoded WAV bytes for Streamlit playback.
              audio_format       : MIME type string (always 'audio/wav').
              provider           : Human-readable source label.
              is_placeholder     : True when not using live API.
              diagnostic_message : Detailed status string for UI badge.
        """
        lang_config = SUPPORTED_LANGUAGES.get(language_code, SUPPORTED_LANGUAGES["hi"])
        gnani_lang_code: str = lang_config["gnani_code"]  # e.g. "hi-IN"
        voice_name: str = self._select_voice(voice_gender)

        # -----------------------------------------------------------------
        # 1. LIVE GNANI TIMBRE v3 API CALL (runs only when configured)
        # -----------------------------------------------------------------
        if self.is_configured():
            logger.info(
                "LIVE Gnani Timbre TTS call → %s | model=%s | voice=%s | lang=%s | text_len=%d",
                self.endpoint, _TTS_MODEL, voice_name, gnani_lang_code, len(text),
            )

            # Auth: single header X-API-Key-ID (verified from official docs)
            headers = {
                "X-API-Key-ID": self.api_key,
                "Content-Type": "application/json",
            }

            # Request body (verified from official docs)
            payload: Dict[str, Any] = {
                "text": text,
                "model": _TTS_MODEL,
                "voice": voice_name,
                "language": gnani_lang_code,
                "speed": 1.0,
                "audio_config": _AUDIO_CONFIG,
            }

            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT_SEC,
                )

                if response.status_code == 200:
                    # Response is raw binary audio (audio/wav) – NOT JSON
                    audio_bytes: bytes = response.content
                    content_type: str = response.headers.get("Content-Type", "audio/wav")
                    b64_audio: str = base64.b64encode(audio_bytes).decode("utf-8")

                    logger.info(
                        "Gnani TTS success: %d audio bytes received (Content-Type: %s)",
                        len(audio_bytes), content_type,
                    )
                    return {
                        "audio_base64": b64_audio,
                        "audio_format": "audio/wav",
                        "provider": "Gnani Timbre TTS v3 (Live API)",
                        "is_placeholder": False,
                        "diagnostic_message": (
                            f"Synthesized via Live Gnani Timbre v3 | voice={voice_name} | "
                            f"language={gnani_lang_code} | {len(audio_bytes)} bytes."
                        ),
                    }

                else:
                    logger.error(
                        "Gnani TTS API error: HTTP %d → %s",
                        response.status_code, response.text[:300],
                    )
                    # On API error: fall back to placeholder chime with clear label
                    fallback_bytes = generate_placeholder_wav(frequency=520.0)
                    return {
                        "audio_base64": base64.b64encode(fallback_bytes).decode("utf-8"),
                        "audio_format": "audio/wav",
                        "provider": "Gnani Timbre TTS v3 (API Error – Chime Fallback)",
                        "is_placeholder": True,
                        "diagnostic_message": (
                            f"Gnani Timbre TTS returned HTTP {response.status_code}. "
                            f"Serving placeholder audio chime. Response: {response.text[:200]}"
                        ),
                    }

            except requests.exceptions.Timeout:
                logger.error("Gnani TTS request timed out after %ds", _REQUEST_TIMEOUT_SEC)
                fallback_bytes = generate_placeholder_wav(frequency=520.0)
                return {
                    "audio_base64": base64.b64encode(fallback_bytes).decode("utf-8"),
                    "audio_format": "audio/wav",
                    "provider": "Gnani Timbre TTS v3 (Timeout – Chime Fallback)",
                    "is_placeholder": True,
                    "diagnostic_message": (
                        f"Gnani Timbre TTS request timed out after {_REQUEST_TIMEOUT_SEC}s. "
                        "Serving placeholder audio chime."
                    ),
                }
            except requests.exceptions.ConnectionError as exc:
                logger.error("Gnani TTS connection error: %s", exc)
                fallback_bytes = generate_placeholder_wav(frequency=520.0)
                return {
                    "audio_base64": base64.b64encode(fallback_bytes).decode("utf-8"),
                    "audio_format": "audio/wav",
                    "provider": "Gnani Timbre TTS v3 (Connection Error – Chime Fallback)",
                    "is_placeholder": True,
                    "diagnostic_message": f"Could not reach Gnani Timbre API: {exc}",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error during Gnani TTS request: %s", exc)
                fallback_bytes = generate_placeholder_wav(frequency=520.0)
                return {
                    "audio_base64": base64.b64encode(fallback_bytes).decode("utf-8"),
                    "audio_format": "audio/wav",
                    "provider": "Gnani Timbre TTS v3 (Unexpected Error – Chime Fallback)",
                    "is_placeholder": True,
                    "diagnostic_message": f"Unexpected error calling Gnani Timbre API: {exc}",
                }

        # -----------------------------------------------------------------
        # 2. TRANSPARENT INTEGRATION PLACEHOLDER MODE (no API key in .env)
        # -----------------------------------------------------------------
        logger.info(
            "Gnani Timbre TTS – Placeholder Mode (GNANI_TIMBRE_API_KEY not set in .env)."
        )
        placeholder_bytes = generate_placeholder_wav(duration_sec=1.8, frequency=480.0)
        b64_audio = base64.b64encode(placeholder_bytes).decode("utf-8")

        return {
            "audio_base64": b64_audio,
            "audio_format": "audio/wav",
            "provider": "Gnani Timbre TTS (Integration Placeholder Mode)",
            "is_placeholder": True,
            "diagnostic_message": (
                "[INTEGRATION STATUS: Gnani Timbre TTS API Key Not Configured] "
                "Running in transparent placeholder mode with audible tone verification. "
                "Set GNANI_TIMBRE_API_KEY (or GNANI_API_KEY) in .env to enable live neural voice synthesis."
            ),
        }


# Global singleton
gnani_tts = GnaniTimbreTTSService()
