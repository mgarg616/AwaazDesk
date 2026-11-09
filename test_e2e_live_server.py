"""
End-to-End Live Integration Test for AwaazDesk FastAPI & Streamlit Components.
Starts FastAPI server on a test port (8005), exercises every single REST endpoint
via live HTTP requests, tests error conditions, verifies SQLite updates, and shuts down.
"""
import sys
import os
import time
import subprocess
import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_e2e_tests():
    port = 8005
    base_url = f"http://127.0.0.1:{port}"
    print(f"=== Starting FastAPI Server for Live E2E Testing on port {port} ===")

    cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "info"
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env
    )

    try:
        # 1. Wait for server readiness
        max_retries = 45  # allow up to 25 seconds for initial cold-start imports
        ready = False
        print("Waiting for server to become ready...")
        for i in range(max_retries):
            # Check if process died
            if server_proc.poll() is not None:
                print(f"Server process exited prematurely with code {server_proc.returncode}!")
                return False
            try:
                r = requests.get(f"{base_url}/", timeout=1)
                if r.status_code == 200:
                    ready = True
                    print(f"Server ready after {i+1} attempts ({(i+1)*0.5:.1f}s)!")
                    break
            except Exception:
                time.sleep(0.5)

        if not ready:
            print("ERROR: Server failed to start within timeout.")
            return False

        print("\n--- 1. Testing GET / ---")
        r = requests.get(f"{base_url}/")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "AwaazDesk" in r.json()["system"]
        print("PASSED: GET / returned operational status.")

        print("\n--- 2. Testing GET /api/health ---")
        r = requests.get(f"{base_url}/api/health")
        assert r.status_code == 200
        health = r.json()
        assert health["status"] == "Healthy"
        assert health["ml_classifier"]["ready"] is True
        assert "speech_services" in health
        print("PASSED: GET /api/health verified ML readiness & speech diagnostics.")

        print("\n--- 3. Testing POST /api/complaints/preview ---")
        preview_payload = {
            "text": "Road par deep potholes hain aur open sewer ka dhakkan toota hua hai, accident ka khatra hai.",
            "language": "hinglish"
        }
        r = requests.post(f"{base_url}/api/complaints/preview", json=preview_payload)
        assert r.status_code == 200
        prev_data = r.json()
        assert "category_ml" in prev_data
        assert "urgency_level_rule" in prev_data
        assert "sentiment_polarity" in prev_data
        assert "summary" in prev_data
        assert "ML Model" in prev_data["model_attribution"]["category"]
        assert "Rule-Based" in prev_data["model_attribution"]["urgency"]
        print(f"PASSED: POST /api/complaints/preview: Category='{prev_data['category_ml']}', Urgency='{prev_data['urgency_level_rule']}'")

        print("\n--- 4. Testing POST /api/complaints (Create Grievance) ---")
        new_complaint = {
            "citizen_name": "Sunita Patil",
            "citizen_phone": "+91 98220 54321",
            "citizen_email": "sunita.p@example.com",
            "language": "hi",
            "submission_channel": "text",
            "raw_text": "हमारे इलाके में पिछले चार दिनों से पानी बिल्कुल नहीं आ रहा है और नलों में गंदा पानी आ रहा है। तुरंत पानी का टैंकर भेजें।"
        }
        r = requests.post(f"{base_url}/api/complaints", json=new_complaint)
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
        created = r.json()
        ticket_id = created["ticket_id"]
        assert ticket_id.startswith("AWA-2026-")
        assert created["citizen_name"] == "Sunita Patil"
        assert created["status"] == "Open"
        assert created["category_ml"] == "Water Supply & Drainage"
        print(f"PASSED: Created Complaint with Ticket ID: {ticket_id}")

        print("\n--- 5. Testing GET /api/complaints (List & Filter History) ---")
        r = requests.get(f"{base_url}/api/complaints", params={"limit": 10})
        assert r.status_code == 200
        history = r.json()
        assert len(history) >= 1
        ticket_ids = [item["ticket_id"] for item in history]
        assert ticket_id in ticket_ids
        print(f"PASSED: Retrieved {len(history)} complaints from history.")

        # Test Filter by Category
        r_filtered = requests.get(f"{base_url}/api/complaints", params={"category": "Water Supply & Drainage"})
        assert r_filtered.status_code == 200
        filtered_list = r_filtered.json()
        for item in filtered_list:
            assert item["category_ml"] == "Water Supply & Drainage"
        print(f"PASSED: Filtered {len(filtered_list)} complaints by Category.")

        # Test Search Query
        r_search = requests.get(f"{base_url}/api/complaints", params={"search_query": "Sunita"})
        assert r_search.status_code == 200
        search_res = r_search.json()
        assert any(item["ticket_id"] == ticket_id for item in search_res)
        print("PASSED: Search query successfully retrieved matching grievance.")

        print(f"\n--- 6. Testing GET /api/complaints/{ticket_id} (Fetch Details & Timeline) ---")
        r = requests.get(f"{base_url}/api/complaints/{ticket_id}")
        assert r.status_code == 200
        detail = r.json()
        assert detail["complaint"]["ticket_id"] == ticket_id
        assert len(detail["timeline"]) >= 1
        assert detail["timeline"][0]["action"] == "Grievance Lodged"
        print(f"PASSED: Retrieved complaint details and audit timeline with {len(detail['timeline'])} events.")

        print(f"\n--- 7. Testing PATCH /api/complaints/{ticket_id} (Update Status & Notes) ---")
        patch_payload = {
            "status": "In Progress",
            "assigned_department": "Municipal Water Board - Zone 3",
            "admin_notes": "Emergency water tanker dispatched to citizen location.",
            "updated_by": "Executive Engineer - Sharma"
        }
        r = requests.patch(f"{base_url}/api/complaints/{ticket_id}", json=patch_payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        updated_resp = r.json()
        assert updated_resp["complaint"]["status"] == "In Progress"
        assert updated_resp["complaint"]["assigned_department"] == "Municipal Water Board - Zone 3"
        # Check timeline audit event added
        assert len(updated_resp["timeline"]) >= 2
        latest_event = updated_resp["timeline"][-1]
        assert latest_event["action"] == "Ticket Updated"
        assert latest_event["performed_by"] == "Executive Engineer - Sharma"
        print("PASSED: Successfully transitioned ticket status to 'In Progress' and logged audit timeline.")

        print("\n--- 8. Testing POST /api/voice/transcribe (Gnani Prisma STT) ---")
        # Dummy audio payload
        dummy_wav = b"RIFF....WAVEfmt " + b"\x00" * 200
        files = {"file": ("citizen_audio.wav", dummy_wav, "audio/wav")}
        data = {"language": "hinglish"}
        r = requests.post(f"{base_url}/api/voice/transcribe", files=files, data=data)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        stt_resp = r.json()
        assert "text" in stt_resp
        assert len(stt_resp["text"]) > 5
        assert "provider" in stt_resp
        assert "is_placeholder" in stt_resp
        print(f"PASSED: Gnani Prisma STT returned transcript: '{stt_resp['text'][:40]}...' (Provider: {stt_resp['provider']})")

        print("\n--- 9. Testing POST /api/voice/synthesize (Gnani Timbre TTS) ---")
        tts_payload = {
            "text": "Aapka ticket number AWA-2026-8812 update ho chuka hai.",
            "language": "hi",
            "voice_gender": "female"
        }
        r = requests.post(f"{base_url}/api/voice/synthesize", json=tts_payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        tts_resp = r.json()
        assert "audio_base64" in tts_resp
        assert len(tts_resp["audio_base64"]) > 50
        assert tts_resp["audio_format"] == "audio/wav"
        print(f"PASSED: Gnani Timbre TTS synthesized valid WAV audio payload ({len(tts_resp['audio_base64'])} bytes b64).")

        print("\n--- 10. Testing GET /api/analytics (Plotly Aggregations) ---")
        r = requests.get(f"{base_url}/api/analytics")
        assert r.status_code == 200
        analytics = r.json()
        assert "total_complaints" in analytics
        assert analytics["total_complaints"] >= 1
        assert "category_distribution" in analytics
        assert "urgency_distribution" in analytics
        assert "status_distribution" in analytics
        assert "language_distribution" in analytics
        print(f"PASSED: Analytics computed over {analytics['total_complaints']} total complaints.")

        print("\n--- 11. Testing POST /api/ml/retrain (Retraining Engine) ---")
        r = requests.post(f"{base_url}/api/ml/retrain")
        assert r.status_code == 200
        retrain_data = r.json()
        assert retrain_data["success"] is True
        assert retrain_data["samples_trained"] > 0
        print(f"PASSED: Model retrained on {retrain_data['samples_trained']} samples.")

        print("\n--- 12. Testing Error Handling & Edge Cases ---")
        # Nonexistent ticket
        r_err = requests.get(f"{base_url}/api/complaints/AWA-0000-NONEXISTENT")
        assert r_err.status_code == 404
        # Invalid status
        r_bad_status = requests.patch(f"{base_url}/api/complaints/{ticket_id}", json={"status": "INVALID_STATUS"})
        assert r_bad_status.status_code == 400
        print("PASSED: Verified 404 Not Found and 400 Bad Request validations.")

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL 12 LIVE END-TO-END INTEGRATION TESTS PASSED CLEANLY!")
        print("=" * 60)
        return True

    finally:
        print("\nTerminating test server...")
        server_proc.terminate()
        server_proc.wait()
        print("Test server shutdown completed.")

if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
