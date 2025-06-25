# cinfer/utils/security.py
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import hashlib

from passlib.context import CryptContext
from jose import JWTError, jwt

from core.config import get_config_manager
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration (fetch from global config)
# These should be in your config.yaml, e.g., under an 'auth' or 'jwt' section
config = get_config_manager()
# Ensure your config/config.yaml has these values, e.g.:
# auth:
#   jwt_secret_key: "YOUR_VERY_SECRET_KEY_CHANGE_THIS"
#   jwt_algorithm: "HS256"
#   jwt_access_token_expire_days: 30 # Default if not specified per token

JWT_SECRET_KEY = config.get_config("auth.jwt_secret_key", "a_default_fallback_secret_key_please_change")
JWT_ALGORITHM = config.get_config("auth.jwt_algorithm", "HS256")
DEFAULT_ACCESS_TOKEN_EXPIRE_DAYS = config.get_config("auth.jwt_access_token_expire_days", 36500) # 25 years default

if JWT_SECRET_KEY == "a_default_fallback_secret_key_please_change":
    logger.warning("WARNING: Using default JWT_SECRET_KEY. Please set a strong secret key in your configuration.") # Use proper logging


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)


def get_token_hash(token: str) -> str:
    """Hashes a token.i need hash value not change"""
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT access token.
    Args:
        data (dict): Data to encode in the token (e.g., subject/user_id).
        expires_delta (Optional[timedelta]): Expiration time from now.
                                            Defaults to DEFAULT_ACCESS_TOKEN_EXPIRE_DAYS.
    Returns:
        str: The encoded JWT access token.
    """
    to_encode = data.copy()
    logger.info(f"DEFAULT_ACCESS_TOKEN_EXPIRE_DAYS: {DEFAULT_ACCESS_TOKEN_EXPIRE_DAYS}")
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=DEFAULT_ACCESS_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire})
    # Add 'iat' (issued at) claim
    to_encode.update({"iat": datetime.now(timezone.utc)})
    # Add 'nbf' (not before) claim, can be same as 'iat'
    to_encode.update({"nbf": datetime.now(timezone.utc)})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes a JWT access token.
    Args:
        token (str): The encoded JWT access token.
    Returns:
        Optional[Dict[str, Any]]: The decoded token payload if valid, else None.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        # print(f"JWT Error: {e}") # Use proper logging
        return None

def generate_api_key(length: int = 32) -> str:
    """Generates a cryptographically secure random string for API keys."""
    return secrets.token_urlsafe(length)


