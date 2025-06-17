# cinfer/api/openapi/models.py
import logging
from typing import List, Optional, Dict, Any
from fastapi import Query
from fastapi import APIRouter, Depends, HTTPException, status
from core.auth.token import TokenService
from core.model.manager import ModelManager
from schemas.models import ModelPublicView, ModelOpenAPIView, ModelOpenApiDetails, ModelStatusEnum
from api.dependencies import get_openapi_auth_result, require_openapi_scopes, get_token_svc_dependency, require_access_token # Updated dependencies
from core.auth.permission import Scope
from schemas.auth import AuthResult
from api.dependencies import get_model_mgr, get_db_service, get_request_proc, get_engine_svc
from schemas.common import UnifiedAPIResponse, PaginationInfo
from schemas.engine import InferenceResponse, InferenceBatchResponse
from pydantic import BaseModel, ValidationError
from schemas.request import InferenceRequestData
from utils.exceptions import APIError
from utils.errors import ErrorCode
from core.database import DatabaseService
from fastapi import Body, Path
from core.request.processor import RequestProcessor
from core.engine.service import EngineService
logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter(dependencies=[Depends(require_access_token)])

class PredictRequestBody(BaseModel):
    inputs: Any
    parameters: Optional[Dict[str, Any]] = None
    priority: Optional[int] = 0


@router.get(
    "",
    response_model=UnifiedAPIResponse[List[ModelOpenAPIView]],
    response_model_exclude_none=True,
    summary="List Available Models (Published)",
)
async def list_available_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=10, le=100, description="Number of items per page"),
    model_manager: ModelManager = Depends(get_model_mgr),
    db_service: DatabaseService = Depends(get_db_service)
):
    logger.info(f"OpenAPI request to list models. Page: {page}, Page size: {page_size}")
    
    filters = {"status": ModelStatusEnum.PUBLISHED}

    try:
        db_models = await model_manager.list_models(filters=filters, page=page, page_size=page_size)
        total_items =  db_service.count("models", filters=filters)
        total_pages = (total_items + page_size - 1) // page_size
        if not db_models:
            logger.warning("No published models found.")
            return UnifiedAPIResponse(
                success=True,
                message="No published models found.",
                data=[]
            )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise APIError(
            error=ErrorCode.COMMON_INTERNAL_SERVER_ERROR,
            override_message=str(e)
        )
    
    pagination = PaginationInfo(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        page_size=page_size
    )
    
    public_models: List[ModelOpenAPIView] = []
    for model_db in db_models:
        public_models.append(ModelOpenAPIView.model_validate(model_db))

    return UnifiedAPIResponse(
        success=True,
        message="Models listed successfully",
        data=public_models,
        pagination=pagination
    )

@router.get(
    "/{model_id}",
    response_model=UnifiedAPIResponse[ModelOpenApiDetails],
    response_model_exclude_none=True,
    summary="Get Public Details of a Specific Model"
)
async def get_public_model_details(
    model_id: str,
    auth_result: AuthResult = Depends(get_openapi_auth_result),
    model_manager: ModelManager = Depends(get_model_mgr),
    token_service: TokenService = Depends(get_token_svc_dependency) # For fetching full token details
):
    logger.info(f"OpenAPI request for details of model ID: {model_id}")
    model_db = await model_manager.get_model(model_id)
    
    if not model_db or model_db.status != ModelStatusEnum.PUBLISHED:
        logger.warning(f"Public view: Model ID '{model_id}' not found or not published.")
        raise APIError(
            error=ErrorCode.COMMON_NOT_FOUND,
            override_message=f"Model with ID '{model_id}' not found or is not published."
        )

    # Check if this X-Access-Token is allowed to use this specific model_id
    if auth_result.token_id:
        access_token_details = token_service.get_access_token_by_id(auth_result.token_id)
        if access_token_details and access_token_details.allowed_models:
            if "ALL" in access_token_details.allowed_models:
                pass
            elif model_id in access_token_details.allowed_models:
                pass
            else:
                raise APIError(
                    error=ErrorCode.COMMON_FORBIDDEN,
                    override_message=f"Access Token {auth_result.token_id} not authorized for model {model_id}.")
    model_details: ModelOpenApiDetails = ModelOpenApiDetails.model_validate(model_db)
    return UnifiedAPIResponse(
        success=True,
        message="Model details retrieved successfully",
        data=model_details
    )


@router.post(
    "/{model_id}/infer",
    response_model=UnifiedAPIResponse[InferenceResponse],
    response_model_exclude_none=True,
    summary="Perform Synchronous Inference on a Model"
)
async def perform_synchronous_inference(
    model_id: str = Path(..., description="ID of the model to use for inference"),
    request_body: PredictRequestBody = Body(...),
    auth_result: AuthResult = Depends(get_openapi_auth_result), # Use OpenAPI auth
    request_processor: RequestProcessor = Depends(get_request_proc),
    token_service: TokenService = Depends(get_token_svc_dependency),
    engine_service: EngineService = Depends(get_engine_svc),
    model_manager: ModelManager = Depends(get_model_mgr)
):
    logger.info(f"Sync inference request for model ID: {model_id} by X-Access-Token ID: {auth_result.token_id}")

    # Authorization: Check if token is allowed to use this specific model_id
    if auth_result.token_id: # auth_result.token_id is the DB ID of the access_token
        access_token_details = token_service.get_access_token_by_id(auth_result.token_id)
        if access_token_details and access_token_details.allowed_models:
           if "ALL" in access_token_details.allowed_models:
               pass
           elif model_id in access_token_details.allowed_models:
               pass
           else:
               raise APIError(
                   error=ErrorCode.COMMON_FORBIDDEN,
                   override_message=f"Access Token {auth_result.token_id} not authorized for model {model_id}.")
           
    #validate inputs
    model_info = await model_manager.get_model(model_id)
    input_schema = engine_service.get_input_validator(model_id, model_info)
    if not input_schema:
        raise APIError(
            error=ErrorCode.COMMON_BAD_REQUEST,
            override_message=f"Model {model_id} has no input schema."
        )
    try:
        model_input_schema = input_schema.model_validate(request_body.inputs)
        # logger.info(f"Model input schema: {model_input_schema}")
    except ValidationError as e:
        raise APIError(
            error=ErrorCode.COMMON_BAD_REQUEST,
            override_message=f"Invalid input schema for model {model_id}: {e}"
        )

    inputs_list = await request_processor.convert_inputs_to_inference_list(model_input_schema)
    inference_payload_dict = {
        "model_id": model_id,
        "inputs": inputs_list, 
        "parameters": request_body.parameters,
        "priority": request_body.priority 
    }

    result = await request_processor.process_request(inference_payload_dict)
    
    if not result.success:
        logger.error(f"Inference failed for model {model_id}, request by X-Access-Token {auth_result.token_id}: {result.error_message}")
        raise APIError(
            error=ErrorCode.COMMON_BAD_REQUEST,
            override_message=result.error_message or "Inference processing failed."
        )
    
    logger.info(f"Sync inference successful for model {model_id}, X-Access-Token {auth_result.token_id}. Processing time: {result.processing_time_ms} ms")
    
    return UnifiedAPIResponse(
        success=True,
        message="Inference successful",
        data=InferenceResponse(
            outputs=result.outputs[0],
            processing_time_ms=result.processing_time_ms
        )
    )


@router.post(
    "/{model_id}/batch-infer",
    response_model=UnifiedAPIResponse[InferenceBatchResponse],
    response_model_exclude_none=True,
    summary="Perform Synchronous Batch Inference on a Model"
)
async def perform_synchronous_batch_inference(
    model_id: str = Path(..., description="ID of the model to use for inference"),
    request_body: PredictRequestBody = Body(...),
    auth_result: AuthResult = Depends(get_openapi_auth_result), # Use OpenAPI auth
    request_processor: RequestProcessor = Depends(get_request_proc),
    token_service: TokenService = Depends(get_token_svc_dependency)
):
    logger.info(f"Sync inference request for model ID: {model_id} by X-Access-Token ID: {auth_result.token_id}")

    # Authorization: Check if token is allowed to use this specific model_id
    if auth_result.token_id: # auth_result.token_id is the DB ID of the access_token
        access_token_details = token_service.get_access_token_by_id(auth_result.token_id)
        if access_token_details and access_token_details.allowed_models:
           if "ALL" in access_token_details.allowed_models:
               pass
           elif model_id in access_token_details.allowed_models:
               pass
           else:
               raise APIError(
                   error=ErrorCode.COMMON_FORBIDDEN,
                   override_message=f"Access Token {auth_result.token_id} not authorized for model {model_id}.")
    
    inference_payload_dict = {
        "model_id": model_id,
        "inputs": request_body.inputs.model_dump(),
        "parameters": request_body.parameters,
        "priority": request_body.priority
    }

    result = await request_processor.process_request(inference_payload_dict)
    
    if not result.success:
        logger.error(f"Inference failed for model {model_id}, request by X-Access-Token {auth_result.token_id}: {result.error_message}")
        raise APIError(
            error=ErrorCode.COMMON_BAD_REQUEST,
            override_message=result.error_message or "Inference processing failed."
        )
    
    logger.info(f"Sync inference successful for model {model_id}, X-Access-Token {auth_result.token_id}. Processing time: {result.processing_time_ms} ms")

    return UnifiedAPIResponse(
        success=True,
        message="Batch inference successful",
        data=InferenceBatchResponse(
            outputs=result.outputs,
            processing_time_ms=result.processing_time_ms
        )
    )
