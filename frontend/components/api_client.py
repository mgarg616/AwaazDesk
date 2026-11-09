"""
AwaazDesk - Frontend API Client
Manages all HTTP communication with the FastAPI backend.
"""
import os
import requests
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("awaazdesk.frontend.api")

# Base URL from environment or default to local FastAPI port
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

class AwaazDeskClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.auth_token: Optional[str] = None

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticates with backend using username and password; returns token and user info."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                self.auth_token = data.get("access_token")
                return {"success": True, "data": data}
            elif resp.status_code == 401:
                return {"success": False, "error": "Invalid username or password."}
            elif resp.status_code == 403:
                return {"success": False, "error": "Account is inactive or disabled."}
            else:
                return {"success": False, "error": f"Server error: {resp.text[:200]}"}
        except Exception as e:
            logger.error("Login request exception: %s", e)
            return {"success": False, "error": f"Connection error: {str(e)}"}

    def logout(self, token: Optional[str] = None) -> bool:
        """Notifies backend to revoke JWT server-side, then clears local client session token."""
        use_token = token or self.auth_token
        if use_token:
            try:
                headers = {"Authorization": f"Bearer {use_token}"}
                requests.post(f"{self.base_url}/api/auth/logout", headers=headers, timeout=5)
            except Exception as e:
                logger.warning("Logout backend notification exception: %s", e)
        self.auth_token = None
        return True

    def get_me(self, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches profile of the currently authenticated user."""
        use_token = token or self.auth_token
        if not use_token:
            return None
        try:
            headers = {"Authorization": f"Bearer {use_token}"}
            resp = requests.get(f"{self.base_url}/api/auth/me", headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error("Get me exception: %s", e)
            return None

    def get_health(self) -> Dict[str, Any]:
        """Fetches backend health status, ML model status, and Gnani API states."""
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "Error", "detail": resp.text}
        except Exception as e:
            return {"status": "Disconnected", "detail": str(e)}

    def preview_intelligence(self, text: str, language: str = "hinglish") -> Optional[Dict[str, Any]]:
        """Gets real-time ML category, urgency, sentiment, and summary without submitting."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/complaints/preview",
                json={"text": text, "language": language},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Preview HTTP failed (%s), falling back to local intelligence...", e)

        # Resilient local intelligence fallback
        try:
            from backend.ml.ml_classifier import ml_classifier
            from backend.services.urgency_engine import evaluate_urgency
            from backend.services.sentiment_engine import analyze_sentiment
            from backend.services.summarizer import generate_summary
            from backend.config import ML_CONFIDENCE_THRESHOLD

            ml_res = ml_classifier.predict_category(text)
            urgency_res = evaluate_urgency(text)
            sentiment_res = analyze_sentiment(text)
            summary_res = generate_summary(text)

            return {
                "category_ml": ml_res["category_ml"],
                "category_confidence": ml_res["category_confidence"],
                "category_top_probabilities": ml_res["category_top_probabilities"],
                "urgency_level_rule": urgency_res["urgency_level_rule"],
                "urgency_score": urgency_res["urgency_score"],
                "urgency_triggers": urgency_res["urgency_triggers"],
                "sentiment_polarity": sentiment_res["sentiment_polarity"],
                "sentiment_score": sentiment_res["sentiment_score"],
                "summary": summary_res["summary"],
                "needs_manual_review": bool(ml_res["category_confidence"] < ML_CONFIDENCE_THRESHOLD),
                "model_attribution": {
                    "category": "[ML Model: Scikit-learn TF-IDF + LogisticRegression]",
                    "urgency": "[Rule-Based Heuristic Engine]",
                    "sentiment": "[Lexical Sentiment Engine]",
                    "summary": "[Extractive Summarizer]"
                }
            }
        except Exception as local_err:
            logger.error("Local preview fallback error: %s", local_err)
            return None

    def transcribe_audio(self, audio_bytes: bytes, filename: str, language: str = "hinglish") -> Optional[Dict[str, Any]]:
        """
        Calls Gnani Prisma STT endpoint to transcribe uploaded audio.
        Sends the correct MIME type based on the file extension so the
        backend and Gnani receive accurate Content-Type metadata.
        Timeout is 100s (Gnani v3 short-clip calls may take up to 90s on
        busy servers; backend itself uses a 90s request timeout + overhead).
        """
        # Derive correct MIME type from file extension (never hardcode audio/wav)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "flac": "audio/flac",
        }
        mime_type = mime_map.get(ext, "audio/wav")

        try:
            files = {"file": (filename, audio_bytes, mime_type)}
            data = {"language": language}
            resp = requests.post(
                f"{self.base_url}/api/voice/transcribe",
                files=files,
                data=data,
                timeout=100   # Must exceed backend's 90s Gnani timeout + processing margin
            )
            if resp.status_code == 200:
                return resp.json()
            logger.error(
                "Transcribe failed: HTTP %d – %s",
                resp.status_code, resp.text[:300]
            )
            # Return structured error so frontend can show a meaningful message
            return {
                "text": "",
                "confidence": None,
                "language": language,
                "provider": f"Backend Error (HTTP {resp.status_code})",
                "is_placeholder": False,
                "diagnostic_message": f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}",
                "_client_error": True,
            }
        except requests.exceptions.Timeout:
            logger.error("Transcribe request timed out for file %s", filename)
            return {
                "text": "",
                "confidence": None,
                "language": language,
                "provider": "Client Timeout",
                "is_placeholder": False,
                "diagnostic_message": "Transcription request timed out (100s). Try a shorter audio clip.",
                "_client_error": True,
            }
        except Exception as e:
            logger.error("Transcribe exception: %s", e)
            return None

    def synthesize_speech(self, text: str, language: str = "hi", voice_gender: str = "female") -> Optional[Dict[str, Any]]:
        """Calls Gnani Timbre TTS endpoint to synthesize text into audio."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/voice/synthesize",
                json={"text": text, "language": language, "voice_gender": voice_gender},
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
            logger.error("Synthesize failed: %s", resp.text)
            return None
        except Exception as e:
            logger.error("Synthesize exception: %s", e)
            return None

    def submit_complaint(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submits a new grievance to FastAPI (with resilient local fallback if server offline)."""
        try:
            resp = requests.post(f"{self.base_url}/api/complaints", json=payload, timeout=15)
            if resp.status_code == 201:
                return resp.json()
            logger.error("Submit failed: %s", resp.text)
        except Exception as e:
            logger.warning("Submit HTTP connection failed (%s), attempting database insertion...", e)
            try:
                import random, json
                from datetime import datetime
                from backend.ml.ml_classifier import ml_classifier
                from backend.services.urgency_engine import evaluate_urgency
                from backend.services.sentiment_engine import analyze_sentiment
                from backend.services.summarizer import generate_summary
                from backend.database import create_complaint, get_complaint_by_ticket
                from backend.config import ML_CONFIDENCE_THRESHOLD

                raw_text = payload.get("raw_text", "").strip()
                ml_res = ml_classifier.predict_category(raw_text)
                urgency_res = evaluate_urgency(raw_text)
                sentiment_res = analyze_sentiment(raw_text)
                summary_res = generate_summary(raw_text)
                is_low_conf = bool(ml_res["category_confidence"] < ML_CONFIDENCE_THRESHOLD)
                ticket_id = f"AWA-2026-{random.randint(1000, 9999)}"

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

                data = {
                    "ticket_id": ticket_id,
                    "citizen_name": payload.get("citizen_name", "Anonymous"),
                    "citizen_phone": payload.get("citizen_phone"),
                    "citizen_email": payload.get("citizen_email"),
                    "language": payload.get("language", "en"),
                    "submission_channel": payload.get("submission_channel", "text"),
                    "original_audio_path": payload.get("original_audio_path"),
                    "raw_text": raw_text,
                    "transcription_confidence": payload.get("transcription_confidence", 1.0),
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
                    "needs_manual_review": 1 if is_low_conf else 0
                }
                create_complaint(data)
                created = get_complaint_by_ticket(ticket_id)
                if created:
                    return dict(created)
            except Exception as direct_err:
                logger.error("Direct complaint creation exception: %s", direct_err)
        return None

    def list_complaints(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        urgency: Optional[str] = None,
        language: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists complaints matching filter criteria (with database fallback)."""
        try:
            params = {"limit": limit}
            if category and category != "All":
                params["category"] = category
            if status and status != "All":
                params["status"] = status
            if urgency and urgency != "All":
                params["urgency"] = urgency
            if language and language != "All":
                params["language"] = language
            if search_query:
                params["search_query"] = search_query

            headers = {}
            use_token = token or self.auth_token
            if use_token:
                headers["Authorization"] = f"Bearer {use_token}"
            resp = requests.get(f"{self.base_url}/api/complaints", params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("FastAPI /api/complaints returned HTTP %d, falling back to database", resp.status_code)
        except Exception as e:
            logger.warning("List complaints HTTP failed (%s), querying database directly...", e)

        # Resilient local database fallback
        try:
            from backend.database import list_complaints as db_list_complaints
            local_results = db_list_complaints(
                category=category if category != "All" else None,
                status=status if status != "All" else None,
                urgency=urgency if urgency != "All" else None,
                language=language if language != "All" else None,
                search_query=search_query,
                limit=limit
            )
            if local_results:
                return [dict(r) for r in local_results]
        except Exception as db_err:
            logger.error("Database fallback error: %s", db_err)
        return []

    def get_complaint(self, ticket_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches single complaint with timeline audit events (with database fallback)."""
        try:
            headers = {}
            use_token = token or self.auth_token
            if use_token:
                headers["Authorization"] = f"Bearer {use_token}"
            resp = requests.get(f"{self.base_url}/api/complaints/{ticket_id}", headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Get complaint HTTP failed (%s), querying database directly...", e)

        # Resilient local database fallback
        try:
            from backend.database import get_complaint_by_ticket, get_ticket_timeline
            comp = get_complaint_by_ticket(ticket_id)
            if comp:
                timeline = get_ticket_timeline(ticket_id)
                return {"complaint": dict(comp), "timeline": [dict(t) for t in timeline]}
        except Exception as db_err:
            logger.error("Database fallback error in get_complaint: %s", db_err)
        return None

    def update_complaint(self, ticket_id: str, updates: Dict[str, Any], token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Updates complaint status, department, or admin notes. Requires admin token."""
        try:
            headers = {}
            use_token = token or self.auth_token
            if use_token:
                headers["Authorization"] = f"Bearer {use_token}"
            resp = requests.patch(f"{self.base_url}/api/complaints/{ticket_id}", json=updates, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            logger.error("Update complaint failed HTTP %s: %s", resp.status_code, resp.text)
            return None
        except Exception as e:
            logger.error("Update complaint exception: %s", e)
            return None

    def get_analytics(self) -> Optional[Dict[str, Any]]:
        """Retrieves aggregated metrics and distribution maps for Plotly charts."""
        try:
            resp = requests.get(f"{self.base_url}/api/analytics", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Get analytics HTTP failed (%s), querying database directly...", e)

        try:
            from backend.database import get_analytics_summary
            return get_analytics_summary()
        except Exception as db_err:
            logger.error("Database fallback error in get_analytics: %s", db_err)
            return None

    def get_ml_evaluation(self) -> Optional[Dict[str, Any]]:
        """Retrieves empirical ML evaluation report, classification report, and confusion matrix."""
        try:
            resp = requests.get(f"{self.base_url}/api/ml/evaluation", timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error("Get ML evaluation exception: %s", e)
            return None

    def retrain_ml_model(self, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Triggers ML model re-training. Requires admin token."""
        try:
            headers = {}
            use_token = token or self.auth_token
            if use_token:
                headers["Authorization"] = f"Bearer {use_token}"
            resp = requests.post(f"{self.base_url}/api/ml/retrain", headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            logger.error("Retrain failed HTTP %s: %s", resp.status_code, resp.text)
            return None
        except Exception as e:
            logger.error("Retrain exception: %s", e)
            return None

    # Alias for compatibility with frontend calls
    retrain_model = retrain_ml_model

    def analyze_with_evon(self, text: str, language: str = "hinglish") -> Optional[Dict[str, Any]]:
        """Invokes Gnani Evon v3.3 sovereign Indic LLM for deep grievance reasoning."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/evon/analyze",
                json={"text": text, "language": language},
                timeout=15
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error("Evon analyze exception: %s", e)
            try:
                from backend.services.gnani_evon import gnani_evon
                return gnani_evon.analyze_grievance(text, language=language)
            except Exception:
                return None

# Global client singleton
api_client = AwaazDeskClient()
