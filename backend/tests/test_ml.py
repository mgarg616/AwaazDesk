"""
Unit tests for ML Category Classifier, Urgency Engine, Sentiment Engine, and Summarizer.
"""
import pytest
from backend.services.ml_classifier import ml_classifier
from backend.services.urgency_engine import evaluate_urgency
from backend.services.sentiment_engine import analyze_sentiment
from backend.services.summarizer import generate_summary

def test_ml_classifier_hindi():
    """Tests ML category classification on a Hindi grievance."""
    text = "हमारे इलाके में पिछले चार दिनों से पानी बिल्कुल नहीं आ रहा है और नलों में गंदा पानी आ रहा है।"
    result = ml_classifier.predict_category(text)
    assert "category_ml" in result
    assert result["category_ml"] == "Water Supply & Drainage"
    assert result["category_confidence"] > 0.0
    assert "ML Model" in result["attribution"]

def test_ml_classifier_hinglish():
    """Tests ML category classification on a Hinglish power supply complaint."""
    text = "Colony me bijli transformer sparks de raha hai, dangerous electricity power cut issue hai."
    result = ml_classifier.predict_category(text)
    assert "category_ml" in result
    assert result["category_ml"] == "Electricity & Power Supply"
    assert "ML Model" in result["attribution"]

def test_ml_classifier_english_banking():
    """Tests ML category classification on an English banking complaint."""
    text = "ATM transaction failed and cash was not dispensed, but 5000 rupees debited from my bank account."
    result = ml_classifier.predict_category(text)
    assert result["category_ml"] == "Banking & Financial Services"

def test_urgency_detection_critical():
    """Tests critical life safety hazard trigger detection."""
    text = "Loose electric wire is hanging on street with live spark, severe danger of electric shock to school children!"
    result = evaluate_urgency(text)
    assert result["urgency_level_rule"] == "Critical"
    assert result["urgency_score"] >= 8.5
    assert len(result["urgency_triggers"]) > 0
    assert "Rule-Based Heuristic Engine" in result["attribution"]

def test_urgency_detection_medium():
    """Tests medium severity issue detection."""
    text = "Street light is turned off and road has minor pothole, please inspect when possible."
    result = evaluate_urgency(text)
    assert result["urgency_level_rule"] in ["Medium", "Low"]
    assert result["urgency_score"] < 8.0

def test_urgency_detection_water_duration_scaling():
    """
    Verifies that prolonged water outage is properly escalated:
    - 10 days without water is treated as an acute crisis (Critical, >= 8.5)
    - 3 days without water is treated as High urgency (6.5 <= score < 8.5)
    - Supports both Latin Hinglish and Devanagari Hindi.
    """
    # 10 days Hinglish
    res_10d_en = evaluate_urgency("10 din se pani nahi aa raha")
    assert res_10d_en["urgency_level_rule"] == "Critical"
    assert res_10d_en["urgency_score"] >= 8.5

    # 3 days Hinglish
    res_3d_en = evaluate_urgency("3 din se pani nahi aa raha")
    assert res_3d_en["urgency_level_rule"] == "High"
    assert 6.5 <= res_3d_en["urgency_score"] < 8.5
    assert res_10d_en["urgency_score"] > res_3d_en["urgency_score"]

    # 10 days Devanagari
    res_10d_hi = evaluate_urgency("१० दिन से पानी नहीं आ रहा")
    assert res_10d_hi["urgency_level_rule"] == "Critical"
    assert res_10d_hi["urgency_score"] >= 8.5

    # 3 days Devanagari
    res_3d_hi = evaluate_urgency("तीन दिन से पानी नहीं आ रहा")
    assert res_3d_hi["urgency_level_rule"] == "High"
    assert 6.5 <= res_3d_hi["urgency_score"] < 8.5
    assert res_10d_hi["urgency_score"] > res_3d_hi["urgency_score"]

def test_sentiment_analysis():
    """Tests lexical sentiment frustration score."""
    text = "Pathetic and terrible service by bank! Manager refused to speak and behaved terribly with citizen."
    result = analyze_sentiment(text)
    assert result["sentiment_score"] < 0
    assert result["sentiment_polarity"] in ["Negative", "Strongly Negative"]
    assert "Lexical Sentiment Engine" in result["attribution"]

def test_summarizer():
    """Tests extractive summarization."""
    text = "Potholes on ring road are causing heavy traffic jams. Commuters are facing severe delays every morning."
    result = generate_summary(text)
    assert len(result["summary"]) > 10
    assert "Extractive Rule-Based Summarizer" in result["attribution"]

def test_multilingual_preprocessing():
    """Tests multilingual text cleaning while preserving Devanagari."""
    from backend.training.evaluate_pipeline import preprocess_multilingual_text
    raw = "  Sadak Par KHAADE Hain!  पानी की सप्लाई बन्द है। Call +91 9876543210 www.test.com  "
    cleaned = preprocess_multilingual_text(raw)
    assert "sadak par khaade hain" in cleaned
    assert "पानी की सप्लाई बन्द है" in cleaned
    assert "www.test.com" not in cleaned
    assert "+91" not in cleaned

def test_evaluation_report_structure():
    """Tests that the evaluation report contains empirical metrics and honest limitation labels."""
    from backend.training.evaluate_pipeline import REPORT_PATH
    import json
    assert REPORT_PATH.exists()
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "empirical_metrics" in data
    assert "classification_report" in data
    assert "confusion_matrix" in data
    assert "sample_data_labeling_and_limitations" in data

    # Verify confusion matrix shape matches class count
    matrix = data["confusion_matrix"]["matrix"]
    labels = data["confusion_matrix"]["labels"]
    assert len(matrix) == len(labels) == 8
    for row in matrix:
        assert len(row) == 8


# -----------------------------------------------------------------------------
# Phase 1 ML Reliability & Human-in-the-Loop Tests
# -----------------------------------------------------------------------------
def test_unverified_complaints_excluded_from_retraining():
    """
    Verifies that unverified model predictions (category_ml) are strictly
    excluded from the retraining corpus, preventing self-training feedback loops.
    """
    from backend.training.train_classifier import load_training_data
    from backend.main import submit_complaint
    from backend.models.schemas import ComplaintCreateRequest

    unique_phrase = "UniqueUnverifiedTestPhraseXYZ12345 road breakdown"
    payload = ComplaintCreateRequest(
        citizen_name="Test Citizen",
        language="en",
        submission_channel="text",
        raw_text=unique_phrase
    )
    created = submit_complaint(payload)
    assert created.verified_category is None

    # Load training data
    texts, labels = load_training_data()
    # Unique unverified complaint must NOT be present
    assert unique_phrase not in texts


def test_verified_complaints_included_in_retraining():
    """
    Verifies that complaints reviewed and verified by an authorized officer
    ARE ingested into the retraining dataset.
    """
    from backend.training.train_classifier import load_training_data
    from backend.main import submit_complaint, update_complaint_status
    from backend.models.schemas import ComplaintCreateRequest, ComplaintUpdateRequest
    from backend.security.auth import create_access_token, get_current_user

    unique_phrase = "VerifiedRoadCrackIncidentABC98765 highway bridge collapse"
    payload = ComplaintCreateRequest(
        citizen_name="Verification Test Citizen",
        language="en",
        submission_channel="text",
        raw_text=unique_phrase
    )
    created = submit_complaint(payload)

    # Officer verifies the ticket with human-reviewed category
    officer_tok = create_access_token(data={"sub": "officer_test", "role": "officer"})
    officer_user = get_current_user(token=officer_tok)

    update_req = ComplaintUpdateRequest(
        verified_category="Civic Infrastructure & Sanitation",
        admin_notes="Officer on-site inspection confirmed bridge structural crack."
    )
    updated = update_complaint_status(created.ticket_id, update_req, current_officer=officer_user)
    assert updated["complaint"].verified_category == "Civic Infrastructure & Sanitation"
    assert updated["complaint"].verified_by is not None
    assert updated["complaint"].verified_at is not None

    # Load training data - verified complaint must now be present!
    texts, labels = load_training_data()
    assert unique_phrase in texts
    idx = texts.index(unique_phrase)
    assert labels[idx] == "Civic Infrastructure & Sanitation"


def test_citizens_cannot_verify_labels():
    """
    Verifies that citizens attempting to verify or alter grievance categories
    are blocked with HTTP 403 Forbidden.
    """
    from fastapi import HTTPException
    from backend.main import update_complaint_status
    from backend.models.schemas import ComplaintUpdateRequest
    from backend.security.auth import create_access_token, get_current_user, require_officer_or_admin

    citizen_tok = create_access_token(data={"sub": "citizen_demo", "role": "citizen"})
    citizen_user = get_current_user(token=citizen_tok)

    update_req = ComplaintUpdateRequest(
        verified_category="Healthcare & Civil Supplies"
    )

    with pytest.raises(HTTPException) as exc:
        # Dependency check will raise HTTP 403
        require_officer_or_admin(current_user=citizen_user)
    assert exc.value.status_code == 403
    assert "officer or administrative privileges required" in exc.value.detail.lower()


def test_officers_can_verify_and_correct_labels():
    """
    Verifies that municipal officers can correct and verify categories,
    which records audit metadata and clears manual review flag.
    """
    from backend.main import submit_complaint, update_complaint_status
    from backend.models.schemas import ComplaintCreateRequest, ComplaintUpdateRequest
    from backend.security.auth import create_access_token, get_current_user

    payload = ComplaintCreateRequest(
        citizen_name="Triage Citizen",
        language="hi",
        submission_channel="text",
        raw_text="नाली जाम होने से सारा पानी सड़क पर बह रहा है।"
    )
    created = submit_complaint(payload)

    officer_tok = create_access_token(data={"sub": "officer_test", "role": "officer"})
    officer_user = get_current_user(token=officer_tok)

    # Officer verifies and corrects category to Water Supply & Drainage
    update_req = ComplaintUpdateRequest(
        verified_category="Water Supply & Drainage",
        needs_manual_review=False
    )
    res = update_complaint_status(created.ticket_id, update_req, current_officer=officer_user)
    comp = res["complaint"]
    assert comp.verified_category == "Water Supply & Drainage"
    assert comp.needs_manual_review is False
    assert comp.verified_by in ["officer_test", "Test Officer", "Officer Test", "Officer"]
    assert comp.verified_at is not None


def test_unauthorized_users_cannot_change_labels():
    """
    Verifies that unauthenticated callers are rejected with HTTP 401 when attempting
    to verify categories or update tickets.
    """
    from fastapi import HTTPException
    from backend.security.auth import get_current_user

    with pytest.raises(HTTPException) as exc:
        get_current_user(token=None)
    assert exc.value.status_code == 401


def test_confidence_based_manual_triage_flagging():
    """
    Verifies that complaints with confidence below experimental threshold (0.35)
    are flagged with needs_manual_review=True, while original prediction and confidence
    are preserved for auditability.
    """
    from backend.main import submit_complaint
    from backend.models.schemas import ComplaintCreateRequest
    from backend.config import ML_CONFIDENCE_THRESHOLD

    # Ambiguous multi-domain sentence that tends to produce distributed probabilities
    payload = ComplaintCreateRequest(
        citizen_name="Ambiguous Citizen",
        language="en",
        submission_channel="text",
        raw_text="The contractor and the hospital staff had an argument about bills."
    )
    created = submit_complaint(payload)
    assert created.ticket_id.startswith("AWA-2026-")
    assert created.category_ml is not None
    assert created.category_confidence > 0.0

    # If confidence is below threshold, must be flagged for manual review
    if created.category_confidence < ML_CONFIDENCE_THRESHOLD:
        assert created.needs_manual_review is True
    else:
        assert created.needs_manual_review is False

    # Original model prediction is preserved for audit
    assert created.category_top_probabilities is not None


def test_existing_database_records_and_integrations_preserved():
    """
    Verifies backward compatibility:
    - Existing database complaint records remain intact.
    - Historical complaints have verified_category as None (not auto-verified).
    - Database integrity check passes.
    """
    from backend.database import get_db_connection
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("PRAGMA integrity_check")
    assert cur.fetchone()[0] == "ok"

    cur.execute("SELECT COUNT(*) FROM complaints")
    count = cur.fetchone()[0]
    assert count >= 37  # Baseline seeded records preserved

    # Historical seeded complaints must NOT be auto-verified
    cur.execute("SELECT COUNT(*) FROM complaints WHERE ticket_id = 'AWA-2026-9104' AND verified_category IS NULL")
    assert cur.fetchone()[0] == 1

    conn.close()
