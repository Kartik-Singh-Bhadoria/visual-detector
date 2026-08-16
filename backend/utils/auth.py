# backend/utils/auth.py
import hmac
import hashlib
import base64
import json
import time
from fastapi import Header, HTTPException

from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

def create_token(email: str) -> str:
    """
    Generate a signed token for user session.
    Payload contains email and expiration time.
    """
    payload = {
        "email": email,
        "expires": time.time() + 86400  # Token valid for 24 hours
    }
    # base64 encode payload
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    # Create HMAC signature using key from settings
    secret = settings.auth_secret_key.encode()
    signature = hmac.new(secret, payload_b64.encode(), hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{payload_b64}.{signature_b64}"

def verify_token(token: str) -> bool:
    """
    Verify the token signature and expiration.
    """
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return False
        payload_b64, signature_b64 = parts
        
        # We need to add padding back to base64 if it was stripped
        def add_padding(s: str) -> bytes:
            return (s + "=" * ((4 - len(s) % 4) % 4)).encode()
            
        # Recreate expected signature
        secret = settings.auth_secret_key.encode()
        expected_signature = hmac.new(secret, payload_b64.encode(), hashlib.sha256).digest()
        expected_signature_b64 = base64.urlsafe_b64encode(expected_signature).decode().rstrip("=")
        
        # Constant time comparison
        if not hmac.compare_digest(signature_b64, expected_signature_b64):
            return False
            
        # Decode and check expiration
        payload_json = base64.urlsafe_b64decode(add_padding(payload_b64)).decode()
        payload = json.loads(payload_json)
        
        if time.time() > payload.get("expires", 0):
            logger.warning("Token expired")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return False

async def get_current_user(authorization: str = Header(None)) -> bool:
    """
    FastAPI dependency to secure routes.
    """
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Missing or malformed Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Authorization token is missing or malformed."
        )
    token = authorization.split("Bearer ")[1].strip()
    if not verify_token(token):
        logger.warning("Invalid or expired Authorization token")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token."
        )
    return True
