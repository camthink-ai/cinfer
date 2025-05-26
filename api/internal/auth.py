# cinfer/api/internal/auth.py
import logging
import uuid
from datetime import datetime, timezone # Ensure timezone is imported
from typing import List, Annotated, Union

from fastapi import APIRouter, Depends, HTTPException, status, Body, Header
from fastapi.security import OAuth2PasswordRequestForm 

from core.database.base import DatabaseService
from core.auth.token import TokenService
from core.auth.permission import Scope
from utils import security
from schemas.tokens import AdminLoginResponse
from schemas.users import UserInDB
from schemas.common import SuccessResponse
from utils.exceptions import APIError
from utils.errors import ErrorCode
from pydantic import BaseModel
from schemas.auth import AuthResult
from api.dependencies import get_db_service, get_token_svc_dependency, get_internal_auth_result

logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter()

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class AdminRefreshTokenRequest(BaseModel):
    refresh_token: str

class AdminRegisterRequest(BaseModel):
    username: str
    password: str

@router.post("/login", response_model=AdminLoginResponse, summary="Admin Login")
async def login_for_admin_tokens(
    login_request: AdminLoginRequest = Body(...),
    db: DatabaseService = Depends(get_db_service),
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Admin login attempt for username: {login_request.username}")
    user_data = db.find_one("users", {"username": login_request.username, "status": "active"}) # Ensure user is active
    if not user_data:
        raise APIError(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            error=ErrorCode.AUTH_INVALID_CREDENTIALS
            )
    
    user = UserInDB(**user_data)
    if not security.verify_password(login_request.password, user.password_hash):
        raise APIError(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            error=ErrorCode.AUTH_INVALID_CREDENTIALS
            )
    if not user.is_admin:
        raise APIError(
            status_code=status.HTTP_403_FORBIDDEN, 
            error=ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS, 
            override_message="User is not an administrator."
            )

    access_token, refresh_token, rt_expires_in = token_service.generate_admin_auth_tokens(
        user_id=user.id,
        username=user.username
    )

    if not access_token or not refresh_token or not rt_expires_in:
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            error=ErrorCode.COMMON_INTERNAL_ERROR, 
            override_message="Could not generate admin tokens."
            )
    
    logger.info(f"Admin user '{user.username}' logged in. AT/RT generated.")
    return AdminLoginResponse(
        refresh_token=refresh_token,
        access_token=access_token,
        expires_in=int(rt_expires_in)
        #user details
    )

@router.post("/refresh-token", response_model=AdminLoginResponse, summary="Refresh Admin Access Token")
async def refresh_admin_tokens_endpoint(
    refresh_request: AdminRefreshTokenRequest = Body(...),
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info("Attempting to refresh admin tokens.")
    try:
        new_access_token, new_refresh_token, new_rt_expires_in = token_service.refresh_admin_auth_tokens(
            provided_refresh_token=refresh_request.refresh_token
        )
        if not new_access_token or not new_refresh_token or not new_rt_expires_in: # Should be caught by exceptions in service
             raise APIError(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                 error=ErrorCode.COMMON_INTERNAL_ERROR, 
                 override_message="Token refresh failed during new token generation (controller)."
                 )

        logger.info("Admin tokens refreshed successfully via endpoint.")
        return AdminLoginResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=int(new_rt_expires_in)
        )
    except APIError as e: # Catch specific APIErrors from service (like invalid/expired RT)
        raise e # Re-raise them to be handled by FastAPI exception handlers
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during token refresh: {e}", exc_info=True)
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            error=ErrorCode.COMMON_INTERNAL_ERROR, 
            override_message="An unexpected error occurred during token refresh."
            )


@router.post("/logout", response_model=SuccessResponse, summary="Admin Logout")
async def logout_admin(
    # To logout, client needs to send its current Refresh Token to be invalidated
    # Or, if AT is sent, we might be able to find RT associated with it (more complex)
    # Simpler: client sends RT for invalidation.
    x_auth_token: Annotated[Union[str, None], Header(description="x_auth_token for admin")] = None,
    refresh_request: AdminRefreshTokenRequest = Body(...), # Expect RT to invalidate specific session
    token_service: TokenService = Depends(get_token_svc_dependency),
    admin_auth: AuthResult = Depends(get_internal_auth_result) 
):
    logger.info(f"Admin logout attempt.")
    revoked = token_service.revoke_admin_refresh_token(
                                            refresh_token_to_revoke=refresh_request.refresh_token,
                                            user_id=admin_auth.user_id
                                            )
    # revoked = token_service.revoke_all_admin_refresh_tokens_for_user(admin_auth.user_id) # Alternative: revoke all for user
    
    if revoked:
        return SuccessResponse(success=True, message="Logout successful. Refresh token has been invalidated.")
    else:
        # This could happen if RT was already invalid/expired or not found
        logger.warning(f"Logout: Refresh token not found or already invalid during logout attempt.")
        return SuccessResponse(success=False, message="Logout processed (token may have been already invalid).")



# It creates a user who can then use the /login endpoint.
@router.post("/register", response_model=SuccessResponse, summary="Admin Registration (Initial Setup)")
async def register_admin(
    register_request: AdminRegisterRequest = Body(...),
    db: DatabaseService = Depends(get_db_service)
):
    logger.info(f"Admin registration attempt for username: {register_request.username}")
    existing_user = db.find_one("users", {"username": register_request.username})
    if existing_user:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST, 
            error=ErrorCode.AUTH_USER_EXISTS
            )
    
    user_data_to_insert = {
        "id": str(uuid.uuid4()), 
        "username": register_request.username,
        "password_hash": security.get_password_hash(register_request.password),
        "is_admin": True, 
        "created_at": datetime.now(timezone.utc), 
        "status": "active"
    }
    user_id_pk = db.insert("users", user_data_to_insert)
    if not user_id_pk:
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            error=ErrorCode.COMMON_INTERNAL_ERROR, 
            override_message="Admin user registration failed."
            )
    logger.info(f"Admin user '{register_request.username}' (ID: {user_id_pk}) registered.")
    return SuccessResponse(success=True, message="Admin user registered successfully. You can now login.")