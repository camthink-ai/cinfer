# core/auth/service.py
import logging
from typing import Optional, List, Dict, Any

from fastapi import Request as FastAPIRequest

from .token import TokenService
from .rate_limiter import RateLimiter
from schemas.auth import AuthResult
from schemas.tokens import  AccessTokenSchema # For type hint of validated external token
from core.auth.permission import Scope, check_scopes as util_check_scopes
from utils.exceptions import ErrorCode
# security import not directly needed here for decoding, as TokenService handles it.

logger = logging.getLogger(f"cinfer.{__name__}")

class AuthService:
    def __init__(self,
                 token_service: TokenService,
                 rate_limiter: RateLimiter):
        self.token_service: TokenService = token_service
        self.rate_limiter: RateLimiter = rate_limiter

    def _extract_token_from_request(self, request: FastAPIRequest, header_name: str) -> Optional[str]:
        return request.headers.get(header_name.lower())

    def _get_client_ip(self, request: FastAPIRequest) -> str:
        return request.client.host if request.client else "127.0.0.1"

    async def authenticate_request(self,
                                   request: FastAPIRequest,
                                   token_type: str, 
                                   header_name: str
                                  ) -> AuthResult:
        client_ip = self._get_client_ip(request)

        token_str = self._extract_token_from_request(request, header_name)
        if not token_str:
            return AuthResult(error_code=ErrorCode.TOKEN_NOT_FOUND.to_dict())

        user_id: Optional[str] = None
        #can be user_id for admin AT, or access_token.id for external AT
        token_identifier: Optional[str] = None 
        token_scopes: List[str] = []
        token_specific_rate_limit: Optional[int] = None


        if token_type == "auth": # Internal Admin X-Auth-Token (this is the AT)
            admin_at_payload = self.token_service.validate_admin_access_token(token_str)
            if not admin_at_payload:
                logger.warning(f"Invalid or expired Admin Access Token (X-Auth-Token) from IP: {client_ip}")
                return AuthResult(error_code=ErrorCode.AUTH_INVALID_TOKEN.to_dict())
            
            user_id = admin_at_payload.get("sub")
            token_scopes = admin_at_payload.get("scopes", [])
            token_identifier = user_id


        elif token_type == "access": # OpenAPI X-Access-Token
            token_db_data = self.token_service.validate_external_api_token(token_str, client_ip)
            if not token_db_data:
                logger.warning(f"Invalid or expired OpenAPI Access Token (X-Access-Token) from IP: {client_ip}")
                return AuthResult(error_code=ErrorCode.AUTH_INVALID_TOKEN.to_dict())
            
            user_id = token_db_data.user_id
            token_identifier = token_db_data.id
            token_specific_rate_limit = token_db_data.rate_limit

            #rate limit
            action_key = f"{token_type}_api_request"
            if not self.rate_limiter.check_limit(
                token_id=token_identifier,
                action=action_key,
                token_requests_limit=token_specific_rate_limit,
                token_period_seconds=60
            ):
                # ... (rate limit exceeded error handling)
                logger.warning(f"Rate limit exceeded for {token_type} token ID: {token_identifier}, IP: {client_ip}")
                return AuthResult(error_code=ErrorCode.TOKEN_RATE_LIMIT_EXCEEDED.to_dict())

            #update usage
            self.rate_limiter.increment(token_id=token_identifier, action=action_key)
            self.token_service.increment_access_token_usage(token_identifier)

        else:
            # ... (unknown token_type error handling)
            logger.error(f"Unknown token_type '{token_type}' in authenticate_request.")
            return AuthResult(error_code=ErrorCode.COMMON_INTERNAL_ERROR.to_dict())

        
        return AuthResult(
            is_authenticated=True,
            user_id=user_id,
            # For 'auth' type, token_id might be the AT's JTI if needed, or user_id.
            # For 'access' type, it's the access_token_record_id.
            token_id=token_identifier, 
            token_scopes=token_scopes
        )

