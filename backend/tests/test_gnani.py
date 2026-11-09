"""
AwaazDesk – Gnani STT + TTS Integration Tests
==============================================
Test categories (clearly distinguished):
  [MOCK]        – Uses unittest.mock to simulate Gnani API responses.
                  These tests verify request construction and response parsing
                  WITHOUT making real network calls.
  [PLACEHOLDER] – Verifies placeholder mode behaviour when no API key is set.
  [UNIT]        – Pure unit tests (WAV generator, voice selection, etc.)

NOTE: No live Gnani API calls are made in this test suite.
      Live integration must be tested manually after setting credentials in .env.
"""
import base64
import io
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest

# Reload singletons after mocking so changes in config are reflected
from backend.services.gnani_stt import GnaniPrismaSTTService, MULTILINGUAL_PLACEHOLDER_CORPUS
from backend.services.gnani_tts import GnaniTimbreTTSService, generate_placeholder_wav


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_wav(duration_ms: int = 100) -> bytes:
    """Returns a valid minimal WAV file for use as test audio input."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for _ in range(num_samples):
            wf.writeframes(struct.pack("<h", 0))
    return buf.getvalue()


def _make_mock_200_response(json_data: dict) -> MagicMock:
    """Returns a mock requests.Response with status_code=200 and given JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = json_data
    mock_resp.headers = {"Content-Type": "application/json"}
    return mock_resp


def _make_mock_200_binary_response(audio_bytes: bytes) -> MagicMock:
    """Returns a mock requests.Response with status_code=200 and binary content."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = audio_bytes
    mock_resp.headers = {"Content-Type": "audio/wav"}
    return mock_resp


def _make_mock_error_response(status_code: int, text: str = "Bad Request") -> MagicMock:
    """Returns a mock requests.Response with an error status code."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    mock_resp.content = text.encode()
    mock_resp.headers = {"Content-Type": "application/json"}
    return mock_resp


# ---------------------------------------------------------------------------
# [UNIT] WAV generator
# ---------------------------------------------------------------------------

class TestPlaceholderWavGenerator:
    """[UNIT] Tests for the synthetic WAV chime generator."""

    def test_wav_starts_with_riff_header(self):
        wav = generate_placeholder_wav(duration_sec=0.5, frequency=440.0)
        assert wav[:4] == b"RIFF", "WAV must start with RIFF header"

    def test_wav_has_wave_identifier(self):
        wav = generate_placeholder_wav(duration_sec=0.5)
        assert wav[8:12] == b"WAVE", "WAV must contain WAVE identifier at offset 8"

    def test_wav_minimum_length(self):
        wav = generate_placeholder_wav(duration_sec=0.2)
        assert len(wav) > 100, "WAV must be non-trivially long"

    def test_wav_different_frequencies(self):
        wav_440 = generate_placeholder_wav(frequency=440.0)
        wav_880 = generate_placeholder_wav(frequency=880.0)
        # Both are valid WAVs
        assert wav_440[:4] == b"RIFF"
        assert wav_880[:4] == b"RIFF"
        # They differ in content
        assert wav_440 != wav_880

    def test_wav_duration_affects_size(self):
        short = generate_placeholder_wav(duration_sec=0.5)
        long_ = generate_placeholder_wav(duration_sec=2.0)
        assert len(long_) > len(short), "Longer duration should produce more bytes"


# ---------------------------------------------------------------------------
# [PLACEHOLDER] STT – no API key configured
# ---------------------------------------------------------------------------

class TestGnaniSTTPlaceholderMode:
    """[PLACEHOLDER] STT behaviour when GNANI_PRISMA_API_KEY is not set."""

    def setup_method(self):
        self.stt = GnaniPrismaSTTService()
        # Ensure not configured for all placeholder tests
        self.stt.api_key = ""

    def test_is_configured_false_when_no_key(self):
        assert self.stt.is_configured() is False

    def test_is_configured_false_for_placeholder_literal(self):
        self.stt.api_key = "your_gnani_prisma_api_key_here"
        assert self.stt.is_configured() is False

    def test_placeholder_returns_required_keys(self):
        audio = _make_dummy_wav()
        result = self.stt.transcribe_audio(audio, "test.wav", "hi")
        for key in ("text", "confidence", "language", "provider", "is_placeholder", "diagnostic_message"):
            assert key in result, f"Missing key: {key}"

    def test_placeholder_is_placeholder_true(self):
        audio = _make_dummy_wav()
        result = self.stt.transcribe_audio(audio, "test.wav", "hi")
        assert result["is_placeholder"] is True

    def test_placeholder_text_nonempty(self):
        audio = _make_dummy_wav()
        result = self.stt.transcribe_audio(audio, "test.wav", "hi")
        assert len(result["text"]) > 0

    def test_placeholder_provider_label(self):
        audio = _make_dummy_wav()
        result = self.stt.transcribe_audio(audio, "test.wav", "hi")
        assert "Placeholder" in result["provider"]

    def test_placeholder_diagnostic_mentions_env_var(self):
        audio = _make_dummy_wav()
        result = self.stt.transcribe_audio(audio, "test.wav", "en")
        assert "GNANI_PRISMA_API_KEY" in result["diagnostic_message"] \
               or "GNANI_API_KEY" in result["diagnostic_message"]

    def test_placeholder_all_language_codes(self):
        audio = _make_dummy_wav()
        for lang in ("hi", "en", "hinglish"):
            result = self.stt.transcribe_audio(audio, "test.wav", lang)
            assert result["is_placeholder"] is True
            assert len(result["text"]) > 0, f"Empty transcript for lang={lang}"

    def test_placeholder_corpus_contains_all_languages(self):
        for lang in ("hi", "en", "hinglish"):
            assert lang in MULTILINGUAL_PLACEHOLDER_CORPUS
            assert len(MULTILINGUAL_PLACEHOLDER_CORPUS[lang]) >= 3

    def test_get_status_info_unconfigured(self):
        info = self.stt.get_status_info()
        assert info["configured"] is False
        assert info["status"] == "INTEGRATION_PLACEHOLDER"
        assert "X-API-Key-ID" in info["auth_header"]


# ---------------------------------------------------------------------------
# [MOCK] STT – live API path with mocked requests.post
# ---------------------------------------------------------------------------

class TestGnaniSTTLiveMocked:
    """[MOCK] STT live API path – mocked responses, no real network calls."""

    def setup_method(self):
        self.stt = GnaniPrismaSTTService()
        self.stt.api_key = "mock_test_key_do_not_use_for_real_calls"
        self.audio = _make_dummy_wav()

    def test_live_success_full_transcript_field(self):
        """[MOCK] Verifies full_transcript field is extracted correctly."""
        mock_response = _make_mock_200_response({
            "success": True,
            "full_transcript": "पानी की समस्या है।",
            "detected_language_code": "hi-IN",
            "model": "prisma-v3.0",
            "processing_time": 0.42,
        })
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            result = self.stt.transcribe_audio(self.audio, "test.wav", "hi")

        assert result["is_placeholder"] is False
        assert result["text"] == "पानी की समस्या है।"
        assert result["detected_language"] == "hi-IN"
        assert "Live API" in result["provider"]

    def test_live_success_fallback_transcript_field(self):
        """[MOCK] Verifies fallback to 'transcript' when 'full_transcript' absent."""
        mock_response = _make_mock_200_response({
            "transcript": "Water supply issue",
            "detected_language_code": "en-IN",
        })
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "en")

        assert result["text"] == "Water supply issue"

    def test_live_correct_auth_header_sent(self):
        """[MOCK] Verifies X-API-Key-ID header is used (not x-api-key)."""
        mock_response = _make_mock_200_response({"full_transcript": "test", "detected_language_code": "en-IN"})
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            self.stt.transcribe_audio(self.audio, "test.wav", "en")

        call_kwargs = mock_post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "X-API-Key-ID" in sent_headers, "Must use X-API-Key-ID header (not x-api-key)"
        assert "x-api-key" not in sent_headers, "Must NOT use deprecated x-api-key header"
        assert "Authorization" not in sent_headers, "Must NOT send Authorization/Bearer header"
        assert "x-token" not in sent_headers, "Must NOT send x-token header"

    def test_live_correct_audio_field_name(self):
        """[MOCK] Verifies multipart field is 'audio_file' (not 'audio')."""
        mock_response = _make_mock_200_response({"full_transcript": "test", "detected_language_code": "en-IN"})
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            self.stt.transcribe_audio(self.audio, "test.wav", "en")

        call_kwargs = mock_post.call_args
        sent_files = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files", {})
        assert "audio_file" in sent_files, "Multipart field must be 'audio_file' (not 'audio')"
        assert "audio" not in sent_files or "audio_file" in sent_files

    def test_live_correct_model_sent(self):
        """[MOCK] Verifies model=prisma-v3.0 is in form data."""
        mock_response = _make_mock_200_response({"full_transcript": "test", "detected_language_code": "en-IN"})
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            self.stt.transcribe_audio(self.audio, "test.wav", "en")

        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data", {})
        assert sent_data.get("model") == "prisma-v3.0"

    def test_live_correct_language_code_sent(self):
        """[MOCK] Verifies language_code=indic-auto is sent regardless of UI selection."""
        mock_response = _make_mock_200_response({"full_transcript": "test", "detected_language_code": "hi-IN"})
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            self.stt.transcribe_audio(self.audio, "test.wav", "hinglish")

        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data", {})
        assert sent_data.get("language_code") == "indic-auto", \
            "Must send language_code=indic-auto (not hi-en or hi-IN)"

    def test_live_correct_endpoint_used(self):
        """[MOCK] Verifies the correct Vachana.ai endpoint is called."""
        mock_response = _make_mock_200_response({"full_transcript": "test", "detected_language_code": "en-IN"})
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            self.stt.transcribe_audio(self.audio, "test.wav", "en")

        called_url = mock_post.call_args.args[0] if mock_post.call_args.args else \
                     mock_post.call_args.kwargs.get("url", mock_post.call_args[0][0])
        assert "api.vachana.ai" in called_url, \
            f"Must call api.vachana.ai endpoint, got: {called_url}"
        assert "/stt/v3" in called_url

    def test_live_http_error_returns_structured_result(self):
        """[MOCK] Verifies that an HTTP error returns a structured dict (not exception)."""
        mock_response = _make_mock_error_response(401, "Unauthorized")
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "en")

        assert result["text"] == ""
        assert result["is_placeholder"] is False
        assert "401" in result["diagnostic_message"] or "401" in result["provider"]

    def test_live_connection_error_handled(self):
        """[MOCK] Verifies ConnectionError is caught and returns a structured dict."""
        import requests as _requests
        with patch("backend.services.gnani_stt.requests.post",
                   side_effect=_requests.exceptions.ConnectionError("refused")):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "en")

        assert result["text"] == ""
        assert "Connection Error" in result["provider"]

    def test_live_timeout_handled(self):
        """[MOCK] Verifies Timeout is caught and returns a structured dict."""
        import requests as _requests
        with patch("backend.services.gnani_stt.requests.post",
                   side_effect=_requests.exceptions.Timeout()):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "en")

        assert result["text"] == ""
        assert "Timeout" in result["provider"]

    def test_get_status_info_configured(self):
        info = self.stt.get_status_info()
        assert info["configured"] is True
        assert info["status"] == "LIVE_ACTIVE"


# ---------------------------------------------------------------------------
# [PLACEHOLDER] TTS – no API key configured
# ---------------------------------------------------------------------------

class TestGnaniTTSPlaceholderMode:
    """[PLACEHOLDER] TTS behaviour when GNANI_TIMBRE_API_KEY is not set."""

    def setup_method(self):
        self.tts = GnaniTimbreTTSService()
        self.tts.api_key = ""

    def test_is_configured_false_when_no_key(self):
        assert self.tts.is_configured() is False

    def test_placeholder_returns_required_keys(self):
        result = self.tts.synthesize_speech("Test complaint text.", "hi")
        for key in ("audio_base64", "audio_format", "provider", "is_placeholder", "diagnostic_message"):
            assert key in result, f"Missing key: {key}"

    def test_placeholder_is_placeholder_true(self):
        result = self.tts.synthesize_speech("Test.", "en")
        assert result["is_placeholder"] is True

    def test_placeholder_audio_format(self):
        result = self.tts.synthesize_speech("नमस्ते।", "hi")
        assert result["audio_format"] == "audio/wav"

    def test_placeholder_base64_valid_and_nonempty(self):
        result = self.tts.synthesize_speech("नमस्ते।", "hi")
        b64 = result["audio_base64"]
        assert len(b64) > 50
        # Decodable
        raw = base64.b64decode(b64)
        assert raw[:4] == b"RIFF", "Placeholder audio must be a valid WAV"

    def test_placeholder_provider_label(self):
        result = self.tts.synthesize_speech("Test.", "en")
        assert "Placeholder" in result["provider"]

    def test_placeholder_diagnostic_mentions_env_var(self):
        result = self.tts.synthesize_speech("Test.", "hi")
        assert "GNANI_TIMBRE_API_KEY" in result["diagnostic_message"] \
               or "GNANI_API_KEY" in result["diagnostic_message"]

    def test_placeholder_all_language_codes(self):
        for lang in ("hi", "en", "hinglish"):
            result = self.tts.synthesize_speech("Test grievance.", lang)
            assert result["is_placeholder"] is True
            assert len(result["audio_base64"]) > 50

    def test_get_status_info_unconfigured(self):
        info = self.tts.get_status_info()
        assert info["configured"] is False
        assert info["status"] == "INTEGRATION_PLACEHOLDER"
        assert "X-API-Key-ID" in info["auth_header"]


# ---------------------------------------------------------------------------
# [MOCK] TTS – live API path with mocked requests.post
# ---------------------------------------------------------------------------

class TestGnaniTTSLiveMocked:
    """[MOCK] TTS live API path – mocked responses, no real network calls."""

    def setup_method(self):
        self.tts = GnaniTimbreTTSService()
        self.tts.api_key = "mock_test_key_do_not_use_for_real_calls"

    def test_live_success_returns_audio_bytes(self):
        """[MOCK] Verifies raw binary audio response is base64-encoded correctly."""
        fake_wav = generate_placeholder_wav(duration_sec=0.3)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response):
            result = self.tts.synthesize_speech("Test grievance.", "hi")

        assert result["is_placeholder"] is False
        assert result["audio_format"] == "audio/wav"
        assert "Live API" in result["provider"]
        decoded = base64.b64decode(result["audio_base64"])
        assert decoded == fake_wav

    def test_live_correct_auth_header_sent(self):
        """[MOCK] Verifies X-API-Key-ID header (not x-api-key) is used."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response) as mock_post:
            self.tts.synthesize_speech("Test.", "en")

        call_kwargs = mock_post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "X-API-Key-ID" in sent_headers, "Must use X-API-Key-ID header"
        assert "x-api-key" not in sent_headers, "Must NOT use deprecated x-api-key header"

    def test_live_correct_endpoint_used(self):
        """[MOCK] Verifies the correct Vachana.ai TTS endpoint is called."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response) as mock_post:
            self.tts.synthesize_speech("Test.", "en")

        called_url = mock_post.call_args.args[0] if mock_post.call_args.args else \
                     mock_post.call_args.kwargs.get("url", mock_post.call_args[0][0])
        assert "api.vachana.ai" in called_url, \
            f"Must call api.vachana.ai endpoint, got: {called_url}"
        assert "/api/v1/tts/inference" in called_url

    def test_live_correct_model_in_payload(self):
        """[MOCK] Verifies model=timbre-v3.0 is in JSON payload."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response) as mock_post:
            self.tts.synthesize_speech("Test.", "hi")

        call_kwargs = mock_post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
        assert sent_json.get("model") == "timbre-v3.0"

    def test_live_no_gender_field_in_payload(self):
        """[MOCK] Verifies deprecated 'gender' field is NOT in JSON payload."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response) as mock_post:
            self.tts.synthesize_speech("Test.", "hi", "female")

        call_kwargs = mock_post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
        assert "gender" not in sent_json, "'gender' is not an official Timbre v3 field"
        assert "voice" in sent_json, "'voice' (name-based) must be used instead"

    def test_live_voice_selection_female(self):
        """[MOCK] Verifies female voice maps to a named voice (not 'female' string)."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response) as mock_post:
            self.tts.synthesize_speech("Test.", "hi", "female")

        sent_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert sent_json["voice"] not in ("female", "male", ""), \
            "voice must be a named voice from the Timbre catalogue"

    def test_live_voice_selection_male(self):
        """[MOCK] Verifies male voice maps to a different named voice."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_resp_f = _make_mock_200_binary_response(fake_wav)
        mock_resp_m = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post") as mock_post:
            mock_post.side_effect = [mock_resp_f, mock_resp_m]
            self.tts.synthesize_speech("Test.", "hi", "female")
            self.tts.synthesize_speech("Test.", "hi", "male")

        calls = mock_post.call_args_list
        voice_f = (calls[0].kwargs.get("json") or calls[0][1].get("json", {}))["voice"]
        voice_m = (calls[1].kwargs.get("json") or calls[1][1].get("json", {}))["voice"]
        assert voice_f != voice_m, "Female and male voices should map to different names"

    def test_live_audio_config_in_payload(self):
        """[MOCK] Verifies audio_config is included in the JSON payload."""
        fake_wav = generate_placeholder_wav(duration_sec=0.1)
        mock_response = _make_mock_200_binary_response(fake_wav)
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response) as mock_post:
            self.tts.synthesize_speech("Test.", "en")

        sent_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert "audio_config" in sent_json
        assert sent_json["audio_config"].get("container") == "wav"

    def test_live_http_error_returns_placeholder_chime(self):
        """[MOCK] Verifies HTTP error falls back to placeholder chime (not exception)."""
        mock_response = _make_mock_error_response(403, "Forbidden")
        with patch("backend.services.gnani_tts.requests.post", return_value=mock_response):
            result = self.tts.synthesize_speech("Test.", "hi")

        assert result["is_placeholder"] is True
        assert "403" in result["diagnostic_message"]
        raw = base64.b64decode(result["audio_base64"])
        assert raw[:4] == b"RIFF"

    def test_live_connection_error_returns_placeholder_chime(self):
        """[MOCK] Verifies ConnectionError falls back to placeholder chime."""
        import requests as _requests
        with patch("backend.services.gnani_tts.requests.post",
                   side_effect=_requests.exceptions.ConnectionError("refused")):
            result = self.tts.synthesize_speech("Test.", "en")

        assert result["is_placeholder"] is True
        assert "Connection Error" in result["provider"]

    def test_live_timeout_returns_placeholder_chime(self):
        """[MOCK] Verifies Timeout falls back to placeholder chime."""
        import requests as _requests
        with patch("backend.services.gnani_tts.requests.post",
                   side_effect=_requests.exceptions.Timeout()):
            result = self.tts.synthesize_speech("Test.", "en")

        assert result["is_placeholder"] is True
        assert "Timeout" in result["provider"]

    def test_get_status_info_configured(self):
        info = self.tts.get_status_info()
        assert info["configured"] is True
        assert info["status"] == "LIVE_ACTIVE"


# ---------------------------------------------------------------------------
# [UNIT] Voice selection helper
# ---------------------------------------------------------------------------

class TestVoiceSelection:
    """[UNIT] Tests the _select_voice helper method."""

    def setup_method(self):
        self.tts = GnaniTimbreTTSService()
        self.tts.api_key = ""

    def test_female_returns_named_voice(self):
        voice = self.tts._select_voice("female")
        assert isinstance(voice, str) and len(voice) > 0
        assert voice not in ("female", "male")

    def test_male_returns_named_voice(self):
        voice = self.tts._select_voice("male")
        assert isinstance(voice, str) and len(voice) > 0
        assert voice not in ("female", "male")

    def test_unknown_gender_defaults_to_female_voice(self):
        voice_unknown = self.tts._select_voice("other")
        voice_female = self.tts._select_voice("female")
        assert voice_unknown == voice_female

    def test_empty_string_defaults_to_female_voice(self):
        voice = self.tts._select_voice("")
        assert voice == self.tts._select_voice("female")


# ---------------------------------------------------------------------------
# [UNIT] Audio format helpers
# ---------------------------------------------------------------------------

class TestAudioFormatHelpers:
    """[UNIT] Tests for MIME detection and WAV conversion helpers."""

    def test_detect_mime_wav(self):
        from backend.services.gnani_stt import _detect_mime
        assert _detect_mime("complaint.wav") == "audio/wav"

    def test_detect_mime_mp3(self):
        from backend.services.gnani_stt import _detect_mime
        assert _detect_mime("complaint.mp3") == "audio/mpeg"

    def test_detect_mime_ogg(self):
        from backend.services.gnani_stt import _detect_mime
        assert _detect_mime("recording.ogg") == "audio/ogg"

    def test_detect_mime_m4a(self):
        from backend.services.gnani_stt import _detect_mime
        assert _detect_mime("voice.m4a") == "audio/mp4"

    def test_detect_mime_webm(self):
        from backend.services.gnani_stt import _detect_mime
        assert _detect_mime("clip.webm") == "audio/webm"

    def test_detect_mime_unknown_defaults_to_wav(self):
        from backend.services.gnani_stt import _detect_mime
        assert _detect_mime("file.xyz") == "audio/wav"

    def test_convert_wav_passthrough(self):
        """[UNIT] WAV files must pass through unchanged."""
        from backend.services.gnani_stt import _try_convert_to_wav
        wav = _make_dummy_wav()
        out_bytes, out_name = _try_convert_to_wav(wav, "test.wav")
        assert out_bytes == wav, "WAV bytes must be unchanged"
        assert out_name == "test.wav"

    def test_convert_non_wav_returns_bytes(self):
        """[UNIT] Non-WAV conversion must return bytes and a filename."""
        from backend.services.gnani_stt import _try_convert_to_wav
        fake_ogg = b"OggS" + b"\x00" * 100   # fake OGG header stub
        out_bytes, out_name = _try_convert_to_wav(fake_ogg, "test.ogg")
        assert isinstance(out_bytes, bytes) and len(out_bytes) > 0
        assert isinstance(out_name, str) and len(out_name) > 0


# ---------------------------------------------------------------------------
# [MOCK] STT – schema compatibility (confidence=None must not crash)
# ---------------------------------------------------------------------------

class TestGnaniSTTSchemaSafety:
    """[MOCK] Verifies that Optional[float] confidence does not cause crashes."""

    def setup_method(self):
        self.stt = GnaniPrismaSTTService()
        self.stt.api_key = "mock_test_key_do_not_use"
        self.audio = _make_dummy_wav()

    def test_live_response_without_confidence_field(self):
        """[MOCK] Gnani may omit confidence; result dict must still be valid."""
        # Gnani response with no confidence field at all
        mock_response = _make_mock_200_response({
            "full_transcript": "पानी की समस्या है।",
            "detected_language_code": "hi-IN",
            "model": "prisma-v3.0",
            "processing_time": 0.33,
            # NOTE: no "confidence" key
        })
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "hi")

        assert result["confidence"] is None, "confidence must be None when Gnani omits it"
        assert result["text"] == "पानी की समस्या है।"
        assert result["is_placeholder"] is False

    def test_live_response_with_confidence_field(self):
        """[MOCK] When Gnani includes confidence it should be a float."""
        mock_response = _make_mock_200_response({
            "full_transcript": "Water pipe leaking.",
            "detected_language_code": "en-IN",
            "confidence": 0.97,
        })
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "en")

        assert result["confidence"] == 0.97
        assert isinstance(result["confidence"], float)

    def test_live_response_non_json_body_handled(self):
        """[MOCK] 200 response with non-JSON body must return structured error not exception."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.text = "<html>Gateway error</html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        with patch("backend.services.gnani_stt.requests.post", return_value=mock_resp):
            result = self.stt.transcribe_audio(self.audio, "test.wav", "en")

        assert result["text"] == ""
        assert "non-JSON" in result["diagnostic_message"] or "Unexpected" in result["diagnostic_message"]
        assert result["is_placeholder"] is False

    def test_error_result_has_all_required_keys(self):
        """[UNIT] _error_result must return all keys the TranscriptionResponse schema needs."""
        result = GnaniPrismaSTTService._error_result("hi", "some error", "Some Provider")
        for key in ("text", "confidence", "detected_language", "language",
                    "provider", "is_placeholder", "diagnostic_message"):
            assert key in result, f"Missing key: {key}"
        assert result["text"] == ""
        assert result["is_placeholder"] is False

    def test_live_ogg_file_sent_with_correct_mime(self):
        """[MOCK] OGG file must be sent with audio/ogg MIME, not audio/wav."""
        mock_response = _make_mock_200_response({
            "full_transcript": "test",
            "detected_language_code": "en-IN",
        })
        fake_ogg = b"OggS" + b"\x00" * 200  # fake OGG-like bytes

        with patch("backend.services.gnani_stt.requests.post", return_value=mock_response) as mock_post:
            # Patch conversion to bypass pydub (not installed in test env)
            with patch("backend.services.gnani_stt._try_convert_to_wav",
                       return_value=(fake_ogg, "recording.ogg")):
                self.stt.transcribe_audio(fake_ogg, "recording.ogg", "en")

        call_kwargs = mock_post.call_args
        sent_files = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files", {})
        # file tuple is (filename, bytes, mime)
        file_tuple = sent_files.get("audio_file", (None, None, None))
        assert file_tuple[2] == "audio/ogg", \
            f"OGG must be sent as audio/ogg, got: {file_tuple[2]}"


# ===========================================================================
# 4. Gnani Evon (Artha Sovereign Indic LLM) Tests
# ===========================================================================

class TestGnaniEvonService:
    def setup_method(self):
        from backend.services.gnani_evon import GnaniEvonService
        self.evon = GnaniEvonService()

    def test_evon_status_info_placeholder(self):
        """[PLACEHOLDER] When no key is set, returns placeholder status."""
        self.evon.api_key = ""
        status = self.evon.get_status_info()
        assert status["configured"] is False
        assert "Placeholder" in status["status"]
        assert "gnani-evon" in status["model"]

    def test_evon_placeholder_analysis_electricity(self):
        """[PLACEHOLDER] Categorizes and suggests lineman dispatch for power grievance."""
        self.evon.api_key = ""
        res = self.evon.analyze_grievance("Colony me bijli transformer sparks de raha hai aur power cut hai.")
        assert "Electricity" in res["root_cause_domain"]
        assert "lineman" in res["actionable_recommendation"].lower()
        assert res["is_placeholder"] is True

    def test_evon_placeholder_analysis_water(self):
        """[PLACEHOLDER] Categorizes and suggests tanker dispatch for water outage."""
        self.evon.api_key = ""
        res = self.evon.analyze_grievance("हमारे इलाके में पिछले 4 दिन से पानी बिल्कुल नहीं आ रहा है।")
        assert "Water" in res["root_cause_domain"]
        assert "water tanker" in res["actionable_recommendation"].lower()
        assert res["is_placeholder"] is True

    def test_evon_live_mock_200(self):
        """[MOCK] Live API call parsing with choices/message format."""
        self.evon.api_key = "test_evon_key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Evon sovereign analysis: Transformer failure detected. Recommend replacement."
                    }
                }
            ]
        }
        with patch("backend.services.gnani_evon.requests.post", return_value=mock_resp):
            res = self.evon.analyze_grievance("Transformer blast in sector 4")

        assert "Evon sovereign analysis" in res["reasoning"]
        assert res["is_placeholder"] is False
        assert "Live API" in res["provider"]

    def test_evon_empty_text(self):
        """[UNIT] Empty text handling returns graceful fallback."""
        res = self.evon.analyze_grievance("")
        assert "No complaint text" in res["reasoning"]

