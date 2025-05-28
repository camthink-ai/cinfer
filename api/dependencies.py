# api/dependencies.py
from typing import List, Optional, Any 
from fastapi import Request, Depends, status, Header

from core.database.base import DatabaseService
from core.config import ConfigManager
from core.engine.service import EngineService
from core.model.manager import ModelManager
from core.auth.service import AuthService
from core.auth.token import TokenService
from core.request.queue_manager import QueueManager
from core.request.processor import RequestProcessor
from schemas.auth import AuthResult
from core.auth.permission import Scope, check_scopes as util_check_scopes
from utils.errors import ErrorCode
from utils.exceptions import APIError

# --- Service Getters (remain mostly the same) ---
def get_db_service(request: Request) -> DatabaseService:
    if not hasattr(request.app.state, 'db'): 
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Database service not available.")
    return request.app.state.db

def get_config(request: Request) -> ConfigManager:
    if not hasattr(request.app.state, 'config'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Configuration service not available.")
    return request.app.state.config

def get_engine_svc(request: Request) -> EngineService:
    if not hasattr(request.app.state, 'engine_service'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Engine service not available.")
    return request.app.state.engine_service

def get_model_mgr(request: Request) -> ModelManager:
    if not hasattr(request.app.state, 'model_manager'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Model manager not available.")
    return request.app.state.model_manager

def get_auth_svc_dependency(request: Request) -> AuthService:
    if not hasattr(request.app.state, 'auth_service'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Auth service not available.")
    return request.app.state.auth_service

def get_token_svc_dependency(request: Request) -> TokenService:
    if not hasattr(request.app.state, 'token_service'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Token service not available.")
    return request.app.state.token_service

def get_request_proc(request: Request) -> RequestProcessor:
    if not hasattr(request.app.state, 'request_processor'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Request processor not available.")
    return request.app.state.request_processor

def get_queue_mgr(request: Request) -> QueueManager:
    if not hasattr(request.app.state, 'queue_manager'):
        raise APIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error=ErrorCode.COMMON_SERVICE_UNAVAILABLE, override_message="Queue manager not available.")
    return request.app.state.queue_manager


# --- New Core Authentication Dependencies ---

async def get_internal_auth_result(
    request: Request,
    # x_auth_token: Optional[str] = Header(None, description="Internal Admin Authentication Token"), # Header dependency extracts it
    auth_service: AuthService = Depends(get_auth_svc_dependency)
) -> AuthResult:
    """
    Authenticates internal admin requests using X-Auth-Token.
    """

    auth_result = await auth_service.authenticate_request(request, token_type="auth", header_name="X-Auth-Token")
    if not auth_result.is_authenticated:
        raise APIError(
            status_code=auth_result.status_code or status.HTTP_401_UNAUTHORIZED,
            error=ErrorCode.COMMON_UNAUTHORIZED,
            override_message=auth_result.error_message or "Not authenticated for internal API",
        )
    # Ensure it's an admin session by checking scope (AuthService should fill scopes from validated token)
    if not util_check_scopes([Scope.ADMIN_FULL_ACCESS], auth_result.token_scopes or []):
        raise APIError(
            status_code=status.HTTP_403_FORBIDDEN,
            error=ErrorCode.COMMON_INSUFFICIENT_PERMISSIONS,
            override_message="Administrative privileges required.",
        )
    return auth_result

async def get_openapi_auth_result(
    request: Request,
    # x_access_token: Optional[str] = Header(None, description="OpenAPI Access Token"), # Header dependency extracts it
    auth_service: AuthService = Depends(get_auth_svc_dependency)
) -> AuthResult:
    """
    Authenticates OpenAPI requests using X-Access-Token.
    """
    auth_result = await auth_service.authenticate_request(request, token_type="access", header_name="X-Access-Token")
    if not auth_result.is_authenticated:
        raise APIError(
            status_code=auth_result.status_code or status.HTTP_401_UNAUTHORIZED,
            error=ErrorCode.COMMON_UNAUTHORIZED,
            override_message=auth_result.error_message or "Not authenticated for OpenAPI",
        )
    return auth_result

# --- Scope and User Info Dependencies (adapt to use specific auth results) ---

def require_internal_scopes(required_scopes: List[Scope]):
    """Dependency factory for checking scopes on internal admin routes."""
    async def scope_checker(
        auth_result: AuthResult = Depends(get_internal_auth_result)
    ):
        # Internal routes usually default to ADMIN_FULL_ACCESS, but can be more granular if needed.
        # get_internal_auth_result already checks for ADMIN_FULL_ACCESS.
        # This can be used for *additional* specific internal scopes if any.
        if not util_check_scopes(required_scopes, auth_result.token_scopes or []):
            scopes_str = ", ".join(s.value for s in required_scopes)
            raise APIError(
                status_code=status.HTTP_403_FORBIDDEN,
                error=ErrorCode.COMMON_INSUFFICIENT_PERMISSIONS,
                override_message=f"Not enough permissions for internal API. Requires scope(s): {scopes_str}",
            )
        return auth_result
    return scope_checker

def require_openapi_scopes(required_scopes: List[Scope]):
    """Dependency factory for checking scopes on OpenAPI routes."""
    async def scope_checker(
        auth_result: AuthResult = Depends(get_openapi_auth_result)
    ):
        if not util_check_scopes(required_scopes, auth_result.token_scopes or []):
            scopes_str = ", ".join(s.value for s in required_scopes)
            raise APIError(
                status_code=status.HTTP_403_FORBIDDEN,
                error=ErrorCode.COMMON_INSUFFICIENT_PERMISSIONS,
                override_message=f"Not enough permissions for OpenAPI. Requires scope(s): {scopes_str}",
            )
        return auth_result
    return scope_checker

async def get_current_admin_user_id(auth_result: AuthResult = Depends(get_internal_auth_result)) -> str:
    """Gets user_id from a validated internal admin token."""
    if not auth_result.user_id: 
        raise APIError(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, error=ErrorCode.COMMON_INTERNAL_ERROR, override_message="Admin User ID not found in token.")
    return auth_result.user_id

async def get_current_access_token_user_id(auth_result: AuthResult = Depends(get_openapi_auth_result)) -> Optional[str]:
    """Gets user_id (who created/owns the token) from a validated access token, if present."""
    return auth_result.user_id 

# require_admin_access can now simply depend on get_internal_auth_result
# as that dependency already ensures ADMIN_FULL_ACCESS scope.
async def require_admin_user(auth_result: AuthResult = Depends(get_internal_auth_result)):
    """
    Ensures the request is from an authenticated admin user via X-Auth-Token.
    Returns the AuthResult object for further use if needed.
    """
    return auth_result