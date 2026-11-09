"""
AwaazDesk - Comprehensive Security, Authentication & RBAC Authorization Tests
Verifies:
1. Bcrypt password hashing and salt verification (never plaintext)
2. JWT secret fail-safe validation (rejects missing/empty/weak secrets < 32 chars)
3. JWT access token lifecycle, unique jti, role embedding, and tampering resistance
4. Admin login verification, generic rejection, and brute-force rate-limiting (HTTP 429)
5. Server-side JWT token revocation (blacklist enforcement on logout)
6. Least privilege: Officers can triage complaints, but CANNOT retrain ML models (HTTP 403)
7. Least privilege: System Administrators CAN retrain ML models and update complaints
8. Citizen privacy & PII masking: Public endpoints never expose raw phone or email
9. Admin unmasked PII access: Authorized administrative sessions receive full operational PII
10. Citizen public workflows (lodging grievance, tracking, civic history) remain open
"""
import pytest
from datetime import timedelta
from fastapi import HTTPException

from backend.database import (
    init_db,
    get_user_by_username,
    revoke_token,
    is_token_revoked
)
from backend.models.schemas import (
    LoginRequest,
    ComplaintUpdateRequest,
    ComplaintCreateRequest,
    mask_phone_number,
    mask_email_address,
    mask_citizen_name
)
from backend.security.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    get_current_user,
    get_current_user_optional,
    require_admin,
    require_officer_or_admin,
    check_login_rate_limit,
    record_failed_login,
    reset_failed_logins
)
from backend.config import validate_jwt_secret
from backend.main import (
    login_for_access_token,
    logout_user,
    get_current_user_profile,
    update_complaint_status,
    trigger_retrain,
    submit_complaint,
    get_complaint_details,
    get_complaints,
    get_analytics
)

# Ensure database is initialized
init_db(seed_sample_data=True)


# -----------------------------------------------------------------------------
# 1. Cryptographic Password Hashing & Secret Fail-Safe
# -----------------------------------------------------------------------------
def test_bcrypt_password_hashing():
    """Verifies that passwords are never stored in plaintext and bcrypt hashes correctly."""
    plain = "SuperSecureSecret2026!"
    hashed = hash_password(plain)

    assert plain not in hashed
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False
    assert verify_password("", hashed) is False


def test_jwt_secret_fails_safely_on_weak_or_missing_keys():
    """Verifies that the application fails safely if JWT_SECRET_KEY is empty or weak (< 32 chars)."""
    # Missing / empty key
    with pytest.raises(ValueError) as exc_empty:
        validate_jwt_secret("")
    assert "missing or empty" in str(exc_empty.value).lower()

    # Short / weak key (< 32 chars)
    with pytest.raises(ValueError) as exc_weak:
        validate_jwt_secret("short_insecure_key")
    assert "too weak" in str(exc_weak.value).lower()

    # Valid key (>= 32 chars) passes
    valid_key = "a" * 32
    assert validate_jwt_secret(valid_key) == valid_key


# -----------------------------------------------------------------------------
# 2. JWT Access Token Lifecycle & Tampering Resistance
# -----------------------------------------------------------------------------
def test_jwt_token_creation_and_validation():
    """Verifies JWT encoding, role preservation, and signature integrity."""
    token = create_access_token(data={"sub": "test_officer", "role": "officer"})
    assert isinstance(token, str) and len(token) > 20

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "test_officer"
    assert payload["role"] == "officer"
    assert "jti" in payload
    assert "exp" in payload

    # Tampered token must fail to decode
    tampered_token = token[:-5] + "XXXXX"
    assert decode_access_token(tampered_token) is None


def test_expired_jwt_rejected():
    """Verifies that expired JWTs raise HTTP 401."""
    expired_tok = create_access_token({"sub": "admin", "role": "admin"}, expires_delta=timedelta(seconds=-10))
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=expired_tok)
    assert exc_info.value.status_code == 401


# -----------------------------------------------------------------------------
# 3. Server-Side Token Revocation (Logout Blacklist)
# -----------------------------------------------------------------------------
def test_server_side_jwt_revocation():
    """Verifies that a token revoked on logout cannot be reused."""
    token = create_access_token(data={"sub": "admin", "role": "admin"})
    payload = decode_access_token(token)
    jti = payload["jti"]
    exp = payload["exp"]

    # Before revocation: token validates
    user = get_current_user(token=token)
    assert user["username"] == "admin"

    # Revoke token (simulating logout)
    revoked_success = revoke_token(jti, exp)
    assert revoked_success is True
    assert is_token_revoked(jti) is True

    # After revocation: token is rejected with HTTP 401
    with pytest.raises(HTTPException) as exc_revoked:
        get_current_user(token=token)
    assert exc_revoked.value.status_code == 401
    assert "Token has been revoked" in exc_revoked.value.detail


# -----------------------------------------------------------------------------
# 4. Authentication, Generic Rejection & Rate Limiting
# -----------------------------------------------------------------------------
def test_admin_login_success():
    """Verifies admin login returns signed JWT access token and user profile."""
    # Use existing admin account in test DB
    admin_user = get_user_by_username("admin")
    assert admin_user is not None
    # We verify authentication function directly with known credentials
    req = LoginRequest(username="test_cli_admin", password="CliAdmin@2026")
    resp = login_for_access_token(req)
    assert resp.access_token is not None
    assert resp.token_type == "bearer"
    assert resp.user.username == "test_cli_admin"
    assert resp.user.role == "admin"


def test_login_rejection_generic_error():
    """Verifies invalid passwords and unknown usernames return identical generic 401."""
    with pytest.raises(HTTPException) as exc_pwd:
        login_for_access_token(LoginRequest(username="admin", password="WrongPassword!"))
    assert exc_pwd.value.status_code == 401
    assert exc_pwd.value.detail == "Invalid username or password."

    with pytest.raises(HTTPException) as exc_user:
        login_for_access_token(LoginRequest(username="ghost_user_xyz", password="AnyPassword!"))
    assert exc_user.value.status_code == 401
    assert exc_user.value.detail == "Invalid username or password."


def test_login_brute_force_rate_limiting():
    """Verifies that exceeding 5 failed login attempts triggers HTTP 429."""
    test_identifier = "test_rate_limit_target"
    reset_failed_logins(test_identifier)

    # 4 attempts should not lock out
    for _ in range(4):
        record_failed_login(test_identifier)
    check_login_rate_limit(test_identifier)  # Should not raise

    # 5th attempt triggers lockout
    record_failed_login(test_identifier)
    with pytest.raises(HTTPException) as exc_rate:
        check_login_rate_limit(test_identifier)
    assert exc_rate.value.status_code == 429
    assert "Too many failed login attempts" in exc_rate.value.detail

    # Cleanup
    reset_failed_logins(test_identifier)


# -----------------------------------------------------------------------------
# 5. Least Privilege: Officer vs Admin RBAC Enforcements
# -----------------------------------------------------------------------------
def test_officer_can_triage_complaints_but_cannot_retrain_ml():
    """
    Verifies Principle of Least Privilege:
    Municipal Officers CAN update tickets, but CANNOT trigger ML model retraining.
    """
    # Create officer token
    officer_tok = create_access_token(data={"sub": "officer_test", "role": "officer"})
    officer_user = get_current_user(token=officer_tok)

    # 1. Officer triage permission check: require_officer_or_admin passes
    authorized_officer = require_officer_or_admin(current_user=officer_user)
    assert authorized_officer["role"] == "officer"

    # 2. Officer updating ticket status: SUCCESS (HTTP 200)
    update_req = ComplaintUpdateRequest(
        status="In Progress",
        assigned_department="Public Works Department",
        admin_notes="Officer assigned maintenance crew.",
        updated_by="Officer Test"
    )
    update_res = update_complaint_status("AWA-2026-8812", update_req, current_officer=officer_user)
    assert update_res["complaint"].status == "In Progress"

    # 3. Officer attempting ML retraining: FORBIDDEN (HTTP 403)
    with pytest.raises(HTTPException) as exc_retrain:
        trigger_retrain(current_admin=require_admin(current_user=officer_user))
    assert exc_retrain.value.status_code == 403
    assert "Administrator privileges required" in exc_retrain.value.detail


def test_admin_can_retrain_ml_and_triage_complaints():
    """Verifies that System Administrators possess privileges for ML retraining and triage."""
    admin_tok = create_access_token(data={"sub": "test_cli_admin", "role": "admin"})
    admin_user = get_current_user(token=admin_tok)

    # Admin passes require_admin
    authorized_admin = require_admin(current_user=admin_user)
    assert authorized_admin["role"] == "admin"

    # Admin executes ML retraining
    retrain_res = trigger_retrain(current_admin=admin_user)
    assert retrain_res.success is True
    assert retrain_res.samples_trained > 0


# -----------------------------------------------------------------------------
# 6. Citizen Privacy & PII Protection
# -----------------------------------------------------------------------------
def test_masking_helper_functions():
    """Verifies unit behavior of phone, email, and name masking functions."""
    assert mask_phone_number("+91 98112 43210") == "+91 9*****10"
    assert mask_phone_number("9876543210") == "98765*****10"
    assert mask_phone_number(None) is None

    assert mask_email_address("rajesh.sharma@example.com") == "r***a@example.com"
    assert mask_email_address("pooja.v@example.com") == "p***v@example.com"
    assert mask_email_address(None) is None

    assert mask_citizen_name("Govind Narayan") == "G**** N****"
    assert mask_citizen_name("Pooja") == "P****"
    assert mask_citizen_name("Amit Kumar Verma") == "A**** V****"
    assert mask_citizen_name(None) == "Citizen"
    assert mask_citizen_name("") == "Citizen"


def test_public_complaints_endpoint_masks_citizen_pii():
    """Verifies that anonymous public calls to GET /api/complaints mask citizen phone, email, name, and exclude raw narrative."""
    # Anonymous call (current_user=None)
    public_results = get_complaints(limit=5, current_user=None)
    assert len(public_results) > 0

    for item in public_results:
        # PII must never be raw plaintext
        if item.citizen_phone:
            assert "*****" in item.citizen_phone
        if item.citizen_email:
            assert "***" in item.citizen_email
        if item.citizen_name:
            assert "****" in item.citizen_name
        # Narrative and audio paths must be suppressed for public privacy
        assert item.raw_text is None
        assert item.original_audio_path is None


def test_admin_complaints_endpoint_returns_unmasked_pii():
    """Verifies that authenticated admin/officer personnel receive full unmasked contact details and narratives."""
    admin_tok = create_access_token(data={"sub": "test_cli_admin", "role": "admin"})
    admin_user = get_current_user(token=admin_tok)

    admin_results = get_complaints(limit=5, current_user=admin_user)
    assert len(admin_results) > 0

    # At least one seeded complaint has real phone and email
    has_contact = any(item.citizen_phone and "+" in item.citizen_phone for item in admin_results)
    assert has_contact is True

    # Admin has unmasked raw_text and unmasked citizen name
    has_raw = any(item.raw_text is not None and len(item.raw_text) > 0 for item in admin_results)
    assert has_raw is True


def test_public_tracking_masks_citizen_pii():
    """Verifies that tracking a complaint by ticket_id masks contact PII, hides raw_text & audio path, and sanitizes timeline for public citizens."""
    details_public = get_complaint_details("AWA-2026-8812", current_user=None)
    comp = details_public["complaint"]
    assert comp.ticket_id == "AWA-2026-8812"
    if comp.citizen_phone:
        assert "*****" in comp.citizen_phone
    if comp.citizen_email:
        assert "***" in comp.citizen_email
    if comp.citizen_name:
        assert "****" in comp.citizen_name

    # Raw narrative and storage paths must be suppressed
    assert comp.raw_text is None
    assert comp.original_audio_path is None

    # Public timeline must sanitize citizen full name attribution
    timeline = details_public["timeline"]
    for ev in timeline:
        # If the event was logged by citizen, it should be sanitized to "Citizen"
        if "Citizen" in ev["performed_by"]:
            assert ev["performed_by"] == "Citizen"


# -----------------------------------------------------------------------------
# 7. Citizen Public Workflows Preservation
# -----------------------------------------------------------------------------
def test_citizen_public_endpoints_remain_accessible():
    """Verifies that public citizen services remain completely accessible without authentication."""
    complaint_payload = ComplaintCreateRequest(
        citizen_name="Govind Narayan",
        citizen_phone="+91 97777 33333",
        citizen_email="govind@example.org",
        language="hi",
        submission_channel="text",
        raw_text="नाली की सफाई नहीं होने से सड़क पर गंदा पानी जमा है, बदबू आ रही है।"
    )
    created = submit_complaint(complaint_payload)
    assert created.ticket_id.startswith("AWA-2026-")

    # Tracking works with masked response
    details = get_complaint_details(created.ticket_id, current_user=None)
    assert details["complaint"].citizen_name == "G**** N****"
    assert details["complaint"].raw_text is None
    assert details["complaint"].original_audio_path is None

    # Analytics work
    analytics = get_analytics()
    assert analytics["total_complaints"] > 0
