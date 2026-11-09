"""
AwaazDesk - Authentication & Authorization Security Module
Provides Bcrypt password hashing, JWT token handling, server-side token revocation,
role-based access control (RBAC), and brute-force rate-limiting.
"""
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    validate_jwt_secret
)

logger = logging.getLogger("awaazdesk.security")

# OAuth2 scheme for extracting Bearer tokens from the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# In-memory brute-force tracking: {identifier: [timestamp, timestamp, ...]}
_failed_login_records: Dict[str, List[float]] = {}
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 300  # 5 minutes


def check_login_rate_limit(identifier: str) -> None:
    """Checks whether an identifier (username or IP) has exceeded the failed login threshold."""
    now = time.time()
    attempts = _failed_login_records.get(identifier, [])
    # Filter attempts within lockout window
    recent_attempts = [t for t in attempts if now - t < LOCKOUT_DURATION_SECONDS]
    _failed_login_records[identifier] = recent_attempts

    if len(recent_attempts) >= MAX_LOGIN_ATTEMPTS:
        logger.warning("Rate limit exceeded for login identifier '%s'. Lockout active.", identifier)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please wait 5 minutes before trying again."
        )


def record_failed_login(identifier: str) -> None:
    """Records a failed login attempt timestamp for an identifier."""
    now = time.time()
    attempts = _failed_login_records.get(identifier, [])
    attempts.append(now)
    _failed_login_records[identifier] = attempts


def reset_failed_logins(identifier: str) -> None:
    """Clears failed login records upon successful authentication."""
    _failed_login_records.pop(identifier, None)


def hash_password(plain_password: str) -> str:
    """Hashes a plaintext password using bcrypt with a generated salt."""
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception as e:
        logger.warning("Password verification failed with exception: %s", e)
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a signed JWT access token containing subject, role, unique jti, and expiration.
    Enforces minimum key entropy via validate_jwt_secret.
    """
    secret = validate_jwt_secret(JWT_SECRET_KEY)
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    # Assign a unique JWT ID (jti) for server-side revocation tracking if not provided
    if "jti" not in to_encode:
        to_encode["jti"] = uuid.uuid4().hex

    to_encode.update({"exp": int(expire.timestamp())})
    encoded_jwt = jwt.encode(to_encode, secret, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT access token, verifying secret and signature."""
    try:
        secret = validate_jwt_secret(JWT_SECRET_KEY)
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug("JWT decode failed: %s", e)
        return None
    except ValueError as e:
        logger.error("JWT validation configuration error: %s", e)
        return None


def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency: Returns user dictionary if a valid Bearer token is present,
    or None if the request is anonymous / unauthenticated.
    """
    if not token:
        return None
    payload = decode_access_token(token)
    if payload is None:
        return None

    # Check server-side revocation
    jti = payload.get("jti")
    if jti:
        from backend.database import is_token_revoked
        if is_token_revoked(jti):
            return None

    username = payload.get("sub")
    if not username:
        return None

    from backend.database import get_user_by_username
    user = get_user_by_username(username)
    if user and user.get("is_active", 1):
        return user
    return None


def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    FastAPI dependency: Validates incoming Bearer token, verifies active status,
    and checks server-side revocation blacklist. Raises HTTP 401 if invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    # Enforce server-side token revocation check
    jti = payload.get("jti")
    if jti:
        from backend.database import is_token_revoked
        if is_token_revoked(jti):
            logger.warning("Revoked token access attempt with jti: %s", jti)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    username: Optional[str] = payload.get("sub")
    if username is None:
        raise credentials_exception

    from backend.database import get_user_by_username
    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception

    if not user.get("is_active", 1):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # Attach raw payload (for jti and exp access during logout)
    user_dict = dict(user)
    user_dict["_token_payload"] = payload
    return user_dict


def require_officer_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency: Enforces that user has at least municipal officer privileges
    (permitted for operational complaint triage, department assignment, and status updates).
    """
    allowed_roles = {"officer", "admin", "superadmin"}
    user_role = current_user.get("role", "citizen").lower()

    if user_role not in allowed_roles:
        logger.warning(
            "Access denied: User '%s' with role '%s' attempted officer-only action.",
            current_user.get("username"),
            user_role
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Municipal officer or administrative privileges required.",
        )

    return current_user


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency: Enforces that user is strictly a system Administrator or Superadmin.
    Officers are NOT permitted to execute administrator-only operations (e.g. ML model retraining).
    """
    allowed_roles = {"admin", "superadmin"}
    user_role = current_user.get("role", "citizen").lower()

    if user_role not in allowed_roles:
        logger.warning(
            "Access denied: User '%s' with role '%s' attempted admin-only action.",
            current_user.get("username"),
            user_role
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Administrator privileges required to perform this action.",
        )

    return current_user
