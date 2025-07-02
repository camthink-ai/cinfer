# cinfer/api/internal/tokens.py
import logging
from typing import List, Optional, Annotated, Union
from datetime import timedelta
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status, Body, Header, Query
from utils.exceptions import APIError
from utils.errors import ErrorCode


from core.auth.token import TokenService
from schemas.tokens import AccessTokenSchema, AccessTokenUpdateSchema, AccessTokenCreateRequest, AccessTokenDetail, AccessTokenStatusQueryEnum, AccessTokenSortByEnum, AccessTokenSortOrderEnum
from schemas.common import Message, IdResponse, UnifiedAPIResponse, PaginationInfo
# Use the new admin-specific authentication dependency
from api.dependencies import require_admin_user, get_token_svc_dependency, get_current_admin_user_id
from schemas.auth import AuthResult # To use AuthResult from require_admin_user

logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter(dependencies=[Depends(require_admin_user)], include_in_schema=False) # Protected by admin auth



@router.post(
    "", 
    response_model=UnifiedAPIResponse[AccessTokenDetail], 
    status_code=status.HTTP_201_CREATED,
    response_model_exclude_none=True, 
    summary="Create a New API Access Token (Admin)"
)
async def create_new_api_access_token(
    token_create_payload: AccessTokenCreateRequest,
    token_service: TokenService = Depends(get_token_svc_dependency),
    user_id: str = Depends(get_current_admin_user_id)
):
    logger.info(f"Admin request to create new access token: {token_create_payload.name}")

    # Check if the token name is already taken
    if token_service.get_access_token_by_name(token_create_payload.name):
        raise APIError(
            error=ErrorCode.TOKEN_NAME_ALREADY_IN_USE
        )
    
    #TODO:check model is valid

    jwt_string, created_access_token_db = token_service.generate_external_api_token(
        name=token_create_payload.name,
        user_id=user_id,
        allowed_models=token_create_payload.allowed_models,
        ip_whitelist=token_create_payload.ip_whitelist,
        rate_limit=token_create_payload.rate_limit,
        monthly_limit=token_create_payload.monthly_limit,
        remark=token_create_payload.remark
    )

    if not jwt_string or not created_access_token_db:
        logger.error(f"Failed to create access token '{token_create_payload.name}' via TokenService.")
        raise APIError(
            error=ErrorCode.COMMON_INTERNAL_ERROR,
            override_message="Failed to create access token."
        )
    

    
    token_detail_response = AccessTokenDetail(
        id=created_access_token_db.id,
        name=created_access_token_db.name,
        token=jwt_string,
        ip_whitelist=created_access_token_db.ip_whitelist,
        allowed_models=created_access_token_db.allowed_models,
        rate_limit=created_access_token_db.rate_limit,
        monthly_limit=created_access_token_db.monthly_limit ,
        created_at=int(created_access_token_db.created_at.timestamp()*1000),
        updated_at=int(created_access_token_db.updated_at.timestamp()*1000),
        remaining_requests=(created_access_token_db.monthly_limit - created_access_token_db.used_count) if created_access_token_db.monthly_limit else None,
        remark=created_access_token_db.remark,
        status=created_access_token_db.status
    )
    logger.info(f"Access Token '{token_detail_response.name}' (ID: {created_access_token_db.id}) created successfully by admin.")
    return UnifiedAPIResponse(
        success=True, 
        message="Access token created successfully.",
        data=token_detail_response
    )


@router.get(
        "", 
        response_model=UnifiedAPIResponse[List[AccessTokenDetail]], 
        response_model_exclude_none=True, 
        summary="List API Access Tokens (Admin)"
)
async def list_api_access_tokens(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=10, le=100, description="Number of items per page"),
    sort_by: Optional[AccessTokenSortByEnum] = Query(None, description="Sort by field"),
    sort_order: Optional[AccessTokenSortOrderEnum] = Query(None, description="Sort order"),
    status: Optional[AccessTokenStatusQueryEnum] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by name(partial match) or id(partial match)"),
    user_id: str = Depends(get_current_admin_user_id),
    token_service: TokenService = Depends(get_token_svc_dependency)
):

    logger.info(f"Admin request to list all access tokens. Filters: status={status}, user_id={user_id}, search={search}")
    access_tokens = token_service.list_access_tokens(status=status, user_id=user_id, page=page, page_size=page_size, sort_by=sort_by, sort_order=sort_order, search_term=search, search_fields=["name", "id"])
    total_items = token_service.count_access_tokens(status=status, user_id=user_id, search_term=search, search_fields=["name", "id"])
    logger.info(f"Total access tokens: {total_items}")
    total_pages = (total_items + page_size - 1) // page_size

    if not access_tokens:
        return UnifiedAPIResponse(
            success=True, 
            message="No access tokens found.",
            data=[]
        )
    
    
    pagination = PaginationInfo(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        page_size=page_size
    )
    return UnifiedAPIResponse(
        success=True, 
        message="Access tokens listed successfully.",
        data=access_tokens,
        pagination=pagination
    )


@router.get(
        "/{access_token_id}", 
        response_model=UnifiedAPIResponse[AccessTokenDetail], 
        response_model_exclude_none=True, 
        summary="Get Access Token Details (Admin)"
)
async def get_api_access_token_details(
    access_token_id: str,
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Admin request for details of access token ID: {access_token_id}")
    access_token = token_service.get_access_token_by_id(access_token_id)
    if not access_token:
        logger.warning(f"Access Token ID '{access_token_id}' not found (admin view).")
        raise APIError(
            error=ErrorCode.TOKEN_NOT_FOUND,
            override_message=f"Access Token with ID '{access_token_id}' not found."
        )
    return UnifiedAPIResponse(
        success=True,
        message="Access token details retrieved successfully.",
        data=access_token
    )


@router.put(
        "/{access_token_id}", 
        response_model=UnifiedAPIResponse[AccessTokenDetail], 
        response_model_exclude_none=True, 
        summary="Update Access Token Information (Admin)"
)
async def update_api_access_token(
    access_token_id: str,
    token_update_payload: AccessTokenUpdateSchema,
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Admin request to update access token ID: {access_token_id} with data: {token_update_payload.model_dump(exclude_unset=True)}")
    try:
        updated_token = token_service.update_access_token(access_token_id, token_update_payload)
    except APIError as e:
        logger.error(f"Failed to update access token ID '{access_token_id}': {e}")
        raise e
    
    if not updated_token:
        logger.warning(f"Failed to update access token ID '{access_token_id}'.")
        raise APIError(
            error=ErrorCode.TOKEN_NOT_FOUND,
            override_message=f"Access Token with ID '{access_token_id}' not found or update failed."
        )
    logger.info(f"Access Token ID '{access_token_id}' updated successfully by admin.")
    return UnifiedAPIResponse(
        success=True,
        message="Access token updated successfully.",
        data=updated_token
    )


@router.delete(
        "/{access_token_id}", 
        response_model=UnifiedAPIResponse, 
        response_model_exclude_none=True, 
        summary="Revoke/Delete an API Access Token (Admin)"
)
async def revoke_api_access_token(
    access_token_id: str,
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Admin request to revoke access token ID: {access_token_id}")
    success = token_service.revoke_access_token(access_token_id)
    if not success:
        logger.error(f"Failed to revoke access token ID '{access_token_id}'.")
        raise APIError(
            error=ErrorCode.TOKEN_NOT_FOUND,
            override_message=f"Access Token with ID '{access_token_id}' not found or revocation failed."
        )
    return UnifiedAPIResponse(
        success=True,
        message=f"Access Token with ID '{access_token_id}' revoked successfully."
    )

@router.post(
    "/{access_token_id}/disable",
    response_model=UnifiedAPIResponse,
    response_model_exclude_none=True, 
    summary="Disable an API Access Token (Admin)"
)
async def disable_api_access_token(
    access_token_id: str,
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Admin request to disable access token ID: {access_token_id}")
    success = token_service.disable_access_token(access_token_id)
    if not success:
        logger.error(f"Failed to disable access token ID '{access_token_id}'.")
        raise APIError(
            error=ErrorCode.TOKEN_NOT_FOUND,
            override_message=f"Access Token with ID '{access_token_id}' not found or disable failed."
        )
    return UnifiedAPIResponse(
        success=True,
        message=f"Access Token with ID '{access_token_id}' disabled successfully."
    )


@router.post(
    "/{access_token_id}/enable",
    response_model=UnifiedAPIResponse,
    response_model_exclude_none=True, 
    summary="Enable an API Access Token (Admin)"
)
async def enable_api_access_token(
    access_token_id: str,
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Admin request to enable access token ID: {access_token_id}")
    success = token_service.enable_access_token(access_token_id)
    if not success:
        logger.error(f"Failed to enable access token ID '{access_token_id}'.")
        raise APIError(
            error=ErrorCode.TOKEN_NOT_FOUND,
            override_message=f"Access Token with ID '{access_token_id}' not found or enable failed."
        )
    return UnifiedAPIResponse(
        success=True,
        message=f"Access Token with ID '{access_token_id}' enabled successfully."
    )


