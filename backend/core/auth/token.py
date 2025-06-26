# core/auth/token.py
import uuid
import logging
import json
import secrets # For generating opaque refresh tokens
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Dict, Any

from core.database.base import DatabaseService
# AccessToken Schemas for external tokens
from schemas.tokens import AccessTokenSchema, AccessTokenUpdateSchema, AccessTokenStatus, AccessTokenDetail, AccessTokenSortByEnum, AccessTokenSortOrderEnum
from schemas.tokens import AdminLoginResponse
from core.auth.permission import Scope # Import Scope
from utils import security
from core.config import get_config_manager
from utils.exceptions import CoreServiceException
from utils.errors import ErrorCode
from fastapi import status
from schemas.auth import ValidateTokenResult
logger = logging.getLogger(f"cinfer.{__name__}")

class TokenService:
    def __init__(self, db_service: DatabaseService):
        self.db: DatabaseService = db_service
        self.config = get_config_manager()
        # Expiry for Admin Access Tokens
        self.admin_at_expire_minutes = self.config.get_config("auth.admin_access_token_expire_minutes", 1440) # Default 24h as per request
        # Expiry for Admin Refresh Tokens 
        self.admin_rt_expire_minutes = self.config.get_config("auth.admin_refresh_token_expire_minutes", 1440) # Default 24h as per request

    def _serialize_list_to_json_str(self, data_list: Optional[List[str]]) -> str:
        """Serialize a list to a JSON string. If None, return an empty JSON list string."""
        if data_list is None:
            return "[]"
        return json.dumps(data_list)

    def _deserialize_json_str_to_list(self, json_str: Optional[str]) -> List[str]:
        """Deserialize a JSON string to a list. If empty or None, return an empty list."""
        if not json_str:
            return []
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON string: {json_str}. Returning empty list.")
            return []
        
    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def _generate_opaque_token(self, length: int = 40) -> str:
        return secrets.token_urlsafe(length)
    
    def _generate_opaque_open_api_token(self, prefix: str = "camthink-", length: int = 40) -> str:
        return f"{prefix}{secrets.token_urlsafe(length)}"
    
    def _generate_opaque_open_api_token_with_view(self, prefix: str = "camthink-", token_value: str = None) -> Tuple[str, str]:
        return f"{prefix}******{token_value[-4:]}"
    
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
        refresh_token_hash = security.get_token_hash(refresh_token_str)
        
        # Invalidate any old refresh tokens for this user (optional, for single-session style)


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
            raise CoreServiceException(error=ErrorCode.COMMON_INTERNAL_ERROR)
        
        logger.info(f"Admin AT and RT generated for user {user_id}. RT DB ID: {inserted_rt_id}")
        return access_token_str, refresh_token_str, rt_expires_delta.total_seconds()

    def refresh_admin_auth_tokens(self, provided_refresh_token: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """
        Refreshes admin tokens using a valid Refresh Token.
        Implements Refresh Token Rotation.
        Returns: (new_access_token, new_refresh_token, new_access_token_expires_in)
        """
        hashed_refresh_token = security.get_token_hash(provided_refresh_token)
        found_session_record = self.db.find_one("auth_tokens", {"token_value_hash": hashed_refresh_token, "is_active": True})
    
        if not found_session_record:
            logger.warning("Refresh failed: Provided refresh token not found or invalid.")
            raise CoreServiceException(error=ErrorCode.AUTH_INVALID_TOKEN)

        # Check if the found RT has expired
        rt_expires_at = datetime.fromisoformat(found_session_record["expires_at"].replace("Z", "+00:00")) if isinstance(found_session_record["expires_at"], str) else found_session_record["expires_at"]
        if rt_expires_at < datetime.now(timezone.utc):
            logger.warning(f"Refresh failed: Refresh token (DB ID: {found_session_record['id']}) has expired.")
            self.db.update("auth_tokens", {"id": found_session_record["id"]}, {"is_active": False}) # Deactivate expired RT
            raise CoreServiceException(error=ErrorCode.AUTH_TOKEN_EXPIRED)

        # --- Refresh Token Rotation ---
        # Invalidate the used refresh token
        self.db.update("auth_tokens", {"id": found_session_record["id"]}, {"is_active": False})
        logger.info(f"Old refresh token (DB ID: {found_session_record['id']}) invalidated.")

        # Generate new AT and RT
        user_id = found_session_record["user_id"]
        user_record = self.db.find_one("users", {"id": user_id}) # Need username for new AT
        if not user_record:
             logger.error(f"User with ID {user_id} not found during token refresh for RT DB ID {found_session_record['id']}.")
             raise CoreServiceException(error=ErrorCode.COMMON_INTERNAL_ERROR)
        
        username = user_record["username"]
        new_access_token, new_refresh_token, new_rt_expires_in = self.generate_admin_auth_tokens(user_id, username)
        
        if not new_access_token: # Should not happen if generate_admin_auth_tokens is robust
            logger.error(f"Failed to generate new set of tokens during refresh for user {user_id}.")
            # Attempt to rollback RT invalidation or handle error state
            raise CoreServiceException(error=ErrorCode.COMMON_INTERNAL_ERROR)
            
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
        hashed_refresh_token = security.get_token_hash(refresh_token_to_revoke)
        for session in active_sessions:
            if hashed_refresh_token == session["token_value_hash"]:
                self.db.update("auth_tokens", {"id": session["id"]}, {"is_active": False})
                logger.info(f"Admin Refresh Token (DB ID: {session['id']}) for user {session['user_id']} revoked.")
                revoked_count += 1
                # Break if you expect only one RT to match. If multiple devices can have unique RTs, continue.
        return revoked_count > 0


    def revoke_all_admin_refresh_tokens_for_user(self, user_id: str) -> int:
        """Revokes all active admin refresh tokens for a given user_id."""
        logger.info(f"Revoking all admin refresh tokens for user ID: {user_id}")
        return self.db.update("auth_tokens", {"user_id": user_id, "is_active": True}, {"is_active": False})
    

    # --- Methods for External API Access Tokens (X-Access-Token for OpenAPI) ---
    # These methods (create_access_token, validate_access_token, etc.) remain largely
    # the same as in the previous step, operating on the 'access_tokens' table.
    # Ensure they use the correct table name 'access_tokens'.

    def generate_external_api_token(
        self,
        name: str,
        user_id: str, 
        allowed_models: Optional[List[str]] = None,
        ip_whitelist: Optional[List[str]] = None,
        rate_limit: Optional[int] = 100, 
        monthly_limit: Optional[int] = 0, 
        remark: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[AccessTokenSchema]]:
        """
        Creates an External API token.
        Returns the original token string (for user) and the Pydantic Schema of the token record stored in the database.
        """
        #Generate a unique database record ID (usually UUID) and the actual token value for the user
        actual_token_value = self._generate_opaque_open_api_token()   
        hashed_token_value = security.get_token_hash(actual_token_value)
        viewed_token_value = self._generate_opaque_open_api_token_with_view(token_value=actual_token_value)
        db_record_id = self._generate_id() # access_tokens.id (PK)

        token_data_for_db = {
            "id": db_record_id,
            "user_id": user_id,
            "name": name,
            "token_value_hash": hashed_token_value,
            "token_value_view": viewed_token_value,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "status": AccessTokenStatus.ACTIVE.value,
            "allowed_models": self._serialize_list_to_json_str(allowed_models),
            "ip_whitelist": self._serialize_list_to_json_str(ip_whitelist),
            "rate_limit": rate_limit or 100, 
            "monthly_limit": monthly_limit,
            "used_count": 0,
            "remark": remark,
        }



        inserted_id = self.db.insert("access_tokens", token_data_for_db)
        if not inserted_id : 
            logger.error(f"Failed to save API access token metadata for name {name} to database.")
            return None, None
        
        created_token_db_data = self.db.find_one("access_tokens", {"id": db_record_id})
        if not created_token_db_data:
            logger.error(f"Failed to retrieve newly created API access token {db_record_id} from DB.")
            return None, None
        
        created_token_db_data["allowed_models"] = self._deserialize_json_str_to_list(created_token_db_data.get("allowed_models"))
        created_token_db_data["ip_whitelist"] = self._deserialize_json_str_to_list(created_token_db_data.get("ip_whitelist"))

        return actual_token_value, AccessTokenSchema(**created_token_db_data)

    def validate_external_api_token(self, token_string: str, client_ip: Optional[str] = None) -> ValidateTokenResult:
        """
        Ensures OpenAPI 的 X-Access-Token is valid.
        Does not check for expiration.
        Checks IP whitelist and monthly call limit.
        Provides explanation for rate limit enforcement.
        """
        if not token_string:
            return ValidateTokenResult(is_valid=False, error_code=ErrorCode.TOKEN_NOT_FOUND.to_dict())

        hashed_token_value = security.get_token_hash(token_string)

        logger.info(f"Validating external API token: {token_string} with hash: {hashed_token_value}")
        
        # Find token record by hash value
        token_db_data = self.db.find_one("access_tokens", {"token_value_hash": hashed_token_value})
        
        if not token_db_data:
            logger.warning(f"API Access Token not found for the provided token string (hash match failed).")
            return ValidateTokenResult(is_valid=False, error_code=ErrorCode.TOKEN_NOT_FOUND.to_dict())

        # Manually convert JSON strings in DB to lists to match AccessTokenSchema
        token_db_data["allowed_models"] = self._deserialize_json_str_to_list(token_db_data.get("allowed_models"))
        token_db_data["ip_whitelist"] = self._deserialize_json_str_to_list(token_db_data.get("ip_whitelist"))
        logger.info(f"Token DB data type: {type(token_db_data['allowed_models'])}")
        logger.info(f"Token DB data: {token_db_data}")
        
        try:
            token_schema = AccessTokenSchema(**token_db_data)
        except Exception as e: # Pydantic ValidationError
            logger.error(f"Failed to parse token data from DB into AccessTokenSchema for token ID {token_db_data.get('id')}: {e}")
            return ValidateTokenResult(is_valid=False, error_code=ErrorCode.TOKEN_NOT_FOUND.to_dict())

        # 1. Check if token is active
        if not token_schema.status == AccessTokenStatus.ACTIVE.value:
            logger.warning(f"API Access Token ID {token_schema.id} (name: {token_schema.name}) is not active.")
            return ValidateTokenResult(is_valid=False, error_code=ErrorCode.TOKEN_NOT_FOUND.to_dict())

        # 2. Check IP whitelist (if configured and client_ip provided)
        if client_ip and token_schema.ip_whitelist: # ip_whitelist is List[str]
            if client_ip not in token_schema.ip_whitelist:
                logger.warning(f"Client IP {client_ip} not in whitelist for token ID {token_schema.id} (name: {token_schema.name}). Whitelist: {token_schema.ip_whitelist}")
                return ValidateTokenResult(is_valid=False, error_code=ErrorCode.TOKEN_IP_FORBIDDEN.to_dict())

        # 3. Check monthly call limit 
        if token_schema.monthly_limit is not None and token_schema.monthly_limit >= 0:
            if token_schema.used_count >= token_schema.monthly_limit:
                logger.warning(f"Monthly limit exceeded for token ID {token_schema.id} (name: {token_schema.name}). Used: {token_schema.used_count}, Limit: {token_schema.monthly_limit}")
                return ValidateTokenResult(is_valid=False, error_code=ErrorCode.TOKEN_QUOTA_EXCEEDED.to_dict())
       
        logger.debug(f"Rate limit for token ID {token_schema.id} is {token_schema.rate_limit} (enforcement mechanism not fully implemented here).")

        return ValidateTokenResult(is_valid=True, token_data=token_schema)

    def revoke_access_token(self, access_token_id: str) -> bool:
        """Revoke an Access Token (by its database record ID)"""
        logger.info(f"Revoking access token by record ID: {access_token_id}")
        rows_updated = self.db.update("access_tokens", {"id": access_token_id}, {"status": AccessTokenStatus.REVOKED.value, "updated_at": datetime.now(timezone.utc)})
        return rows_updated > 0
    
    def disable_access_token(self, access_token_id: str) -> bool:
        """Disable an Access Token (by its database record ID)"""
        logger.info(f"Disabling access token by record ID: {access_token_id}")
        rows_updated = self.db.update("access_tokens", {"id": access_token_id}, {"status": AccessTokenStatus.DISABLED.value, "updated_at": datetime.now(timezone.utc)})
        return rows_updated > 0
    
    def enable_access_token(self, access_token_id: str) -> bool:
        """Enable an Access Token (by its database record ID)"""
        logger.info(f"Enabling access token by record ID: {access_token_id}")
        rows_updated = self.db.update("access_tokens", {"id": access_token_id}, {"status": AccessTokenStatus.ACTIVE.value, "updated_at": datetime.now(timezone.utc)})
        return rows_updated > 0

    def get_access_token_by_id(self, access_token_id: str) -> Optional[AccessTokenDetail]:
        """Get Access Token details by database record ID"""
        data = self.db.find_one("access_tokens", {"id": access_token_id})
        if data:
            remaining_requests = None
            if data.get("monthly_limit"):
                remaining_requests = data.get("monthly_limit") - data.get("used_count")
            view_data = {
                "id": data.get("id"),
                "name": data.get("name"),
                "token": data.get("token_value_view"),
                "remaining_requests": remaining_requests,
                "rate_limit": data.get("rate_limit"),
                "monthly_limit": data.get("monthly_limit"),
                "created_at": int(datetime.fromisoformat(data["created_at"]).timestamp()*1000),
                "updated_at": int(datetime.fromisoformat(data["updated_at"]).timestamp()*1000),
                "status": data.get("status"),
                "allowed_models": self._deserialize_json_str_to_list(data.get("allowed_models")),
                "ip_whitelist": self._deserialize_json_str_to_list(data.get("ip_whitelist")),
                "remark": data.get("remark"),
            }
            return AccessTokenDetail(**view_data)
        return None

    def get_access_token_by_name(self, name: str) -> Optional[AccessTokenDetail]:
        """Get Access Token details by name"""
        data = self.db.find_one("access_tokens", {"name": name})
        if data:
            remaining_requests = None
            logger.info(f"Monthly limit: {data.get('monthly_limit')}")
            if data.get("monthly_limit"):
                remaining_requests = data.get("monthly_limit") - data.get("used_count")
            view_data = {
                "id": data.get("id"),
                "name": data.get("name"),
                "token": data.get("token_value_view"),
                "remaining_requests": remaining_requests,
                "rate_limit": data.get("rate_limit"),
                "monthly_limit": data.get("monthly_limit"),
                "created_at": int(datetime.fromisoformat(data["created_at"]).timestamp()*1000),
                "updated_at": int(datetime.fromisoformat(data["updated_at"]).timestamp()*1000),
                "status": data.get("status"),
                "allowed_models": self._deserialize_json_str_to_list(data.get("allowed_models")),
                "ip_whitelist": self._deserialize_json_str_to_list(data.get("ip_whitelist")),
                "remark": data.get("remark"),
            }
            return AccessTokenDetail(**view_data)
        return None

    def list_access_tokens(
        self, 
        status: Optional[AccessTokenStatus] = None, 
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
        sort_by: Optional[AccessTokenSortByEnum] = None,
        sort_order: Optional[AccessTokenSortOrderEnum] = None,
        search_term: Optional[str] = None,
        search_fields: Optional[List[str]] = None
    ) -> List[AccessTokenDetail]:
        """List Access Tokens"""
        filters = {}
        if status is not None: filters["status"] = status
        else: filters["status__in"] = [AccessTokenStatus.ACTIVE.value, AccessTokenStatus.DISABLED.value]
        if user_id: filters["user_id"] = user_id    
        order_by = "created_at DESC"

        logger.info(f"Listing access tokens with filters: {filters}")
        logger.info(f"Search term: {search_term}")
        logger.info(f"Search fields: {search_fields}")
        
        if sort_by:
            sort_key = sort_by.value
            sort_order_key = sort_order.value if sort_order else "DESC"
            order_by = f"{sort_key} {sort_order_key}"
        
        
        token_data_list = self.db.find("access_tokens", filters=filters, limit=page_size, offset=(page - 1) * page_size, order_by=order_by, search_term=search_term, search_fields=search_fields)
        
        result_list = []
        for data in token_data_list:
            remaining_requests = None
            if data.get("monthly_limit"):
                remaining_requests = data.get("monthly_limit") - data.get("used_count")
            logger.info(f"Token data: {data}")
            view_data = {
                "id": data.get("id"),
                "name": data.get("name"),
                "token": data.get("token_value_view"),
                "remaining_requests": remaining_requests,
                "rate_limit": data.get("rate_limit"),
                "monthly_limit": data.get("monthly_limit"),
                "created_at": int(datetime.fromisoformat(data["created_at"]).timestamp()*1000),
                "updated_at": int(datetime.fromisoformat(data["updated_at"]).timestamp()*1000),
                "status": data.get("status"),
                "allowed_models": self._deserialize_json_str_to_list(data.get("allowed_models")),
                "ip_whitelist": self._deserialize_json_str_to_list(data.get("ip_whitelist")),
                "remark": data.get("remark"),
            }
            result_list.append(AccessTokenDetail(**view_data))
        return result_list

    def update_access_token(self, access_token_id: str, update_payload: AccessTokenUpdateSchema) -> Optional[AccessTokenDetail]:
        """Update Access Token information (by its database record ID)"""
        update_data_dict = update_payload.model_dump()
        if not update_data_dict:
            return self.get_access_token_by_id(access_token_id)
        
        #check if name is being modified
        if "name" in update_data_dict:
            #check if the new name is already in use
            existing_token = self.db.find_one("access_tokens", {"name": update_data_dict["name"], "id__ne": access_token_id})
            if existing_token:
                raise CoreServiceException(error=ErrorCode.TOKEN_NAME_ALREADY_IN_USE)

        # Special handling for list fields, ensuring they are stored as JSON strings
        if "allowed_models" in update_data_dict and update_data_dict["allowed_models"] is not None:
            update_data_dict["allowed_models"] = self._serialize_list_to_json_str(update_data_dict["allowed_models"])
        if "ip_whitelist" in update_data_dict and update_data_dict["ip_whitelist"] is not None:
            update_data_dict["ip_whitelist"] = self._serialize_list_to_json_str(update_data_dict["ip_whitelist"])

        update_data_dict["updated_at"] = datetime.now(timezone.utc)

        # Special handling for monthly_limit, ensuring used_count is reset to 0
        if "monthly_limit" in update_data_dict:
            update_data_dict["used_count"] = 0
        
        rows_updated = self.db.update("access_tokens", {"id": access_token_id}, update_data_dict)
        if rows_updated > 0:
            return self.get_access_token_by_id(access_token_id)
        logger.warning(f"Failed to update access token {access_token_id} or no changes made.")
        return None
    
    def count_access_tokens(self, status: Optional[AccessTokenStatus] = None, user_id: Optional[str] = None, search_term: Optional[str] = None, search_fields: Optional[List[str]] = None) -> int:
        """Count Access Tokens"""
        filters = {}
        if status is not None: filters["status"] = status
        else: filters["status__in"] = [AccessTokenStatus.ACTIVE.value, AccessTokenStatus.DISABLED.value]
        if user_id: filters["user_id"] = user_id
        return self.db.count("access_tokens", filters=filters, search_term=search_term, search_fields=search_fields)

    def increment_access_token_usage(self, token_id: str, count: int = 1) -> bool:
        """Increment the usage count for an Access Token (by its database record ID)"""
        # Direct database atomic update operation is more efficient and secure, avoiding race conditions
        # For example: UPDATE access_tokens SET used_count = used_count + ?, updated_at = ? WHERE id = ?
        # If your db.update does not support this expression, you need to read first and then write, as follows:
        
        token = self.db.find_one("access_tokens", {"id": token_id})
        if not token:
            logger.warning(f"Cannot increment usage for non-existent access token ID: {token_id}")
            return False
  
        new_used_count = (token["used_count"] if token["used_count"] is not None else 0) + count

        access_token_id = token["id"]
        
        # Check monthly limit 
        if token["monthly_limit"] is not None and token["monthly_limit"] > 0 and new_used_count > token["monthly_limit"]:
            logger.warning(f"Usage increment for token ID {access_token_id} would exceed monthly limit. Current: {token.used_count}, Increment: {count}, Limit: {token.monthly_limit}. Usage not incremented to exceed limit.")
            #self.db.update("access_tokens", {"id": access_token_id}, {"status": AccessTokenStatus.DISABLED.value, "updated_at": datetime.now(timezone.utc)})
            return False # Indicates limit exceeded

        updated_rows = self.db.update(
            "access_tokens", 
            {"id": access_token_id}, 
            {"used_count": new_used_count, "updated_at": datetime.now(timezone.utc)}
        )
        if updated_rows > 0:
            logger.info(f"Incremented usage for token ID {access_token_id} by {count}. New count: {new_used_count}")
            return True
        logger.warning(f"Failed to increment usage for token ID: {access_token_id}")
        return False
    
    def reset_all_monthly_usage_counts(self) -> int:
        """Reset all monthly usage counts for all access tokens"""
        # Get all active tokens with monthly limits
        active_tokens = self.db.find("access_tokens", {"monthly_limit__gt": 0})
        if not active_tokens:
            logger.info("No tokens with monthly limits found. No usage counts reset.")
            return 0
        
        # Reset usage counts for each token
        for token in active_tokens: 
            self.db.update(
                "access_tokens",
                {"id": token["id"]},
                {"used_count": 0, "updated_at": datetime.now(timezone.utc)}
            )
        return len(active_tokens)
