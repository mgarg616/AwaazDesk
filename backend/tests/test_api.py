"""
Integration tests for FastAPI backend service endpoints.
Directly invokes API endpoint functions for optimal performance and portability.
"""
import pytest
from backend.database import init_db
from backend.main import (
    health_check,
    preview_complaint_intelligence,
    submit_complaint,
    get_complaints,
    get_complaint_details,
    update_complaint_status,
    get_analytics
)

# Ensure DB initialized
init_db(seed_sample_data=True)
from backend.models.schemas import (
    AnalysisPreviewRequest,
    ComplaintCreateRequest,
    ComplaintUpdateRequest
)

def test_health_endpoint():
    """Tests health_check()."""
    data = health_check()
    assert data["status"] == "Healthy"
    assert "ml_classifier" in data
    assert "speech_services" in data
    assert data["ml_classifier"]["ready"] is True

def test_complaints_preview():
    """Tests preview_complaint_intelligence()."""
    req = AnalysisPreviewRequest(
        text="Main road par open transformer sparks nikal raha hai, dangerous electricity issue hai please check.",
        language="hinglish"
    )
    data = preview_complaint_intelligence(req)
    assert data.category_ml == "Electricity & Power Supply"
    assert data.urgency_level_rule in ["Critical", "High"]
    assert "ML Model" in data.model_attribution["category"]
    assert "Rule-Based" in data.model_attribution["urgency"]

def test_create_and_fetch_complaint():
    """Tests submit_complaint(), get_complaint_details(), and update_complaint_status()."""
    req = ComplaintCreateRequest(
        citizen_name="Rohan Gupta",
        citizen_phone="+91 99999 11111",
        citizen_email="rohan@example.com",
        language="en",
        submission_channel="text",
        raw_text="Hospital in our block has no anti-venom medicine and doctor is constantly absent during emergencies."
    )
    created = submit_complaint(req)
    ticket_id = created.ticket_id
    assert ticket_id.startswith("AWA-2026-")
    assert created.citizen_name == "Rohan Gupta"
    assert created.category_ml == "Healthcare & Civil Supplies"

    # Fetch details
    details = get_complaint_details(ticket_id)
    assert details["complaint"].ticket_id == ticket_id
    assert len(details["timeline"]) >= 1

    # Update ticket status
    update_req = ComplaintUpdateRequest(
        status="In Progress",
        assigned_department="District Health Department",
        admin_notes="Emergency medical team dispatched.",
        updated_by="Senior Officer"
    )
    updated = update_complaint_status(ticket_id, update_req)
    assert updated["complaint"].status == "In Progress"
    assert updated["complaint"].admin_notes == "Emergency medical team dispatched."

def test_analytics_endpoint():
    """Tests get_analytics()."""
    data = get_analytics()
    assert "total_complaints" in data
    assert "status_distribution" in data
    assert "category_distribution" in data
    assert "urgency_distribution" in data
    assert data["total_complaints"] > 0

def test_ml_evaluation_endpoint():
    """Tests GET /api/ml/evaluation."""
    from backend.main import get_ml_evaluation_report
    report = get_ml_evaluation_report()
    assert "empirical_metrics" in report
    assert "confusion_matrix" in report
    assert "classification_report" in report
    assert "sample_data_labeling_and_limitations" in report
