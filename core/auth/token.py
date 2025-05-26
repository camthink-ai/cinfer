# core/auth/token.py
import uuid
import logging
import secrets # For generating opaque refresh tokens
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Dict, Any

from core.database.base import DatabaseService
# AccessToken Schemas for external tokens
from schemas.tokens import Token as AccessTokenSchema, TokenCreate as AccessTokenCreateSchema, TokenUpdate as AccessTokenUpdateSchema, TokenDetail
# Schemas for admin login
from schemas.tokens import AdminLoginResponse
from core.auth.permission import Scope # Import Scope
from utils import security
from core.config import get_config_manager
from utils.exceptions import APIError
from utils.errors import ErrorCode
from fastapi import status

logger = logging.getLogger(f"cinfer.{__name__}")

class TokenService:
    def __init__(self, db_service: DatabaseService):
        self.db: DatabaseService = db_service
        self.config = get_config_manager()
        # Expiry for Admin Access Tokens
        self.admin_at_expire_minutes = self.config.get_config("auth.admin_access_token_expire_minutes", 1440) # Default 24h as per request
        # Expiry for Admin Refresh Tokens 
        self.admin_rt_expire_minutes = self.config.get_config("auth.admin_refresh_token_expire_minutes", 1440) # Default 24h as per request

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def _generate_opaque_token(self, length: int = 40) -> str:
        return secrets.token_urlsafe(length)

    # --- Methods for Admin Authentication (X-Auth-Token using AT/RT) ---

    def generate_admin_auth_tokens(self, user_id: str, username: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """
        Generates a new Access Token (AT) and Refresh Token (RT) for an admin user.
        Stores the hashed RT in the 'auth_tokens' database.
        Returns: (access_token, refresh_token, expires_in)
        """

        # 1. Generate Access Token (AT) - JWT
        at_expires_delta = timedelta(minutes=self.admin_at_expire_minutes)
        at_expires_at = datetime.now(timezone.utc) + at_expires_delta
        
        at_payload = {
            "sub": user_id,
            "username": username,
            "scopes": [Scope.ADMIN_FULL_ACCESS.value],
            "type": "admin_access", # Distinguish from other token types
            "exp": at_expires_at,
            "iat": datetime.now(timezone.utc),
            "jti": self._generate_id() # Unique ID for this AT
        }
        access_token_str = security.create_access_token(data=at_payload, expires_delta=at_expires_delta) # Ensures 'exp' is correctly set

        # 2. Generate Refresh Token (RT) - Opaque string
        refresh_token_str = self._generate_opaque_token()
        rt_expires_delta = timedelta(minutes=self.admin_rt_expire_minutes)
        rt_expires_at_db = datetime.now(timezone.utc) + rt_expires_delta

        # 3. Store HASH of RT in 'auth_tokens' table
        refresh_token_hash = security.get_password_hash(refresh_token_str)
        
        # Invalidate any old refresh tokens for this user (optional, for single-session style)
        # self.db.update("auth_tokens", {"user_id": user_id, "is_active": True}, {"is_active": False})

        auth_token_data_for_db = {
            "user_id": user_id,
            "token_value_hash": refresh_token_hash, # Stores hash of RT
            "created_at": datetime.now(timezone.utc),
            "expires_at": rt_expires_at_db,
            "is_active": True
        }
        
        inserted_rt_id = self.db.insert("auth_tokens", auth_token_data_for_db)
        if not inserted_rt_id:
            logger.error(f"Failed to save admin refresh token to database for user {user_id}.")
            return None, None, None
        
        logger.info(f"Admin AT and RT generated for user {user_id}. RT DB ID: {inserted_rt_id}")
        return access_token_str, refresh_token_str, rt_expires_delta.total_seconds()

    def refresh_admin_auth_tokens(self, provided_refresh_token: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """
        Refreshes admin tokens using a valid Refresh Token.
        Implements Refresh Token Rotation.
        Returns: (new_access_token, new_refresh_token, new_access_token_expires_in)
        """
        active_user_sessions = self.db.find("auth_tokens", {"is_active": True})
        found_session_record = None
        
        for session_record in active_user_sessions:
            if security.verify_password(provided_refresh_token, session_record["token_value_hash"]):
                found_session_record = session_record
                break
        
        if not found_session_record:
            logger.warning("Refresh failed: Provided refresh token not found or invalid.")
            raise APIError(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                error=ErrorCode.AUTH_INVALID_TOKEN, 
                override_message="Invalid refresh token."
                )

        # Check if the found RT has expired
        rt_expires_at = datetime.fromisoformat(found_session_record["expires_at"].replace("Z", "+00:00")) if isinstance(found_session_record["expires_at"], str) else found_session_record["expires_at"]
        if rt_expires_at < datetime.now(timezone.utc):
            logger.warning(f"Refresh failed: Refresh token (DB ID: {found_session_record['id']}) has expired.")
            self.db.update("auth_tokens", {"id": found_session_record["id"]}, {"is_active": False}) # Deactivate expired RT
            raise APIError(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                error=ErrorCode.AUTH_TOKEN_EXPIRED, 
                override_message="Refresh token expired."
                )

        # --- Refresh Token Rotation ---
        # Invalidate the used refresh token
        self.db.update("auth_tokens", {"id": found_session_record["id"]}, {"is_active": False})
        logger.info(f"Old refresh token (DB ID: {found_session_record['id']}) invalidated.")

        # Generate new AT and RT
        user_id = found_session_record["user_id"]
        user_record = self.db.find_one("users", {"id": user_id}) # Need username for new AT
        if not user_record:
             logger.error(f"User with ID {user_id} not found during token refresh for RT DB ID {found_session_record['id']}.")
             raise APIError(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                 error=ErrorCode.COMMON_INTERNAL_ERROR, 
                 override_message="User associated with refresh token not found."
                 )
        
        username = user_record["username"]
        new_access_token, new_refresh_token, new_rt_expires_in = self.generate_admin_auth_tokens(user_id, username)
        
        if not new_access_token: # Should not happen if generate_admin_auth_tokens is robust
            logger.error(f"Failed to generate new set of tokens during refresh for user {user_id}.")
            # Attempt to rollback RT invalidation or handle error state
            raise APIError(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                error=ErrorCode.COMMON_INTERNAL_ERROR, 
                override_message="Token refresh failed during new token generation."
                )
            
        logger.info(f"Admin tokens refreshed successfully for user {user_id}.")
        return new_access_token, new_refresh_token, new_rt_expires_in

    def validate_admin_access_token(self, access_token_str: str) -> Optional[Dict[str, Any]]:
        """
        Validates an Admin Access Token (AT JWT).
        Performs cryptographic validation (signature, expiry) and checks scopes.
        This AT is intended to be stateless after issuance for API calls.
        """
        payload = security.decode_access_token(access_token_str)
        if not payload:
            logger.warning("Admin Access Token (AT) JWT decoding failed.")
            return None

        user_id = payload.get("sub")
        token_type = payload.get("type")

        if not user_id or token_type != "admin_access":
            logger.warning("Admin AT JWT missing 'sub' claim or has incorrect 'type'.")
            return None
            
        # Ensure token has admin scope
        if Scope.ADMIN_FULL_ACCESS.value not in payload.get("scopes", []):
            logger.warning(f"Admin AT for user {user_id} lacks admin scope.")
            return None
        
        # Ensure the token is not expired
        if payload.get("exp") < datetime.now(timezone.utc).timestamp():
            logger.warning(f"Admin AT for user {user_id} has expired.")
            return None
            
        # No DB check for the AT itself, as it's short-lived and stateless.
        return payload

    def revoke_admin_refresh_token(self, refresh_token_to_revoke: str, user_id: Optional[str]=None) -> bool:
        """
        Revokes a specific Admin Refresh Token by finding its record via hash and marking it inactive.
        If user_id is provided, it narrows down the search.
        """
        filters = {"is_active": True}
        if user_id:
            filters["user_id"] = user_id
        
        active_sessions = self.db.find("auth_tokens", filters)
        revoked_count = 0
        for session in active_sessions:
            if security.verify_password(refresh_token_to_revoke, session["token_value_hash"]):
                self.db.update("auth_tokens", {"id": session["id"]}, {"is_active": False})
                logger.info(f"Admin Refresh Token (DB ID: {session['id']}) for user {session['user_id']} revoked.")
                revoked_count += 1
                # Break if you expect only one RT to match. If multiple devices can have unique RTs, continue.
        return revoked_count > 0


    def revoke_all_admin_refresh_tokens_for_user(self, user_id: str) -> int:
        """Revokes all active admin refresh tokens for a given user_id."""
        logger.info(f"Revoking all admin refresh tokens for user ID: {user_id}")
        return self.db.update("auth_tokens", {"user_id": user_id, "is_active": True}, {"is_active": False})
