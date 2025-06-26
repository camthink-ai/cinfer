# cinfer/api/internal/models.py
import logging
import shutil 
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Body, Form, Query

from core.model.manager import ModelManager
from schemas.models import (
    Model as ModelSchema,
    ModelCreate, # This is ModelMetadataBase
    ModelMetadataBase, # Use this for form data if file is separate
    ModelUpdate,
    DeploymentResult,
    ModelPublicView,
    ModelViewDetails,
    ModelStatusEnum,
    ModelFileInfo,
    ModelSortByEnum,
    ModelSortOrderEnum
)
from schemas.common import UnifiedAPIResponse, PaginationInfo # Assuming PaginatedResponse
from api.dependencies import get_model_mgr, get_current_admin_user_id
from api.dependencies import require_admin_user
from utils.errors import ErrorCode, ErrorDetail
from utils.exceptions import APIError
from core.database import DatabaseService
from api.dependencies import get_db_service

router = APIRouter(dependencies=[Depends(require_admin_user)])

logger = logging.getLogger(f"cinfer.{__name__}")


# --- Temporary File Handling ---
TEMP_UPLOAD_DIR = Path("data/temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get(
    "",
    response_model=UnifiedAPIResponse[List[ModelPublicView]],
    response_model_exclude_none=True,
    summary="List Available Models (Published)",
  
)
async def list_available_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=10, description="Number of items per page"),
    status: Optional[ModelStatusEnum] = Query(None, description="Filter by status"),
    engine_type: Optional[str] = Query(None, description="Filter by engine type"),
    sort_by: Optional[ModelSortByEnum] = Query(None, description="Sort by field"),
    sort_order: Optional[ModelSortOrderEnum] = Query(None, description="Sort order"),
    search: Optional[str] = Query(None, description="Search by name (partial match) or id (partial match)"),
    user_id: str = Depends(get_current_admin_user_id),
    model_manager: ModelManager = Depends(get_model_mgr),
    db_service: DatabaseService = Depends(get_db_service)
):
    logger.info(f"Admin request to list all published models. Filters: status={status},  engine_type={engine_type}, user_id={user_id}")
    filters = {}
    search_fields = None
    if engine_type:
        engine_type = engine_type.split(",")
        filters["engine_type__in"] = engine_type
    if status:
        filters["status"] = status  

    try:
        db_models = await model_manager.list_models(filters=filters, page=page, page_size=page_size, sort_by=sort_by, sort_order=sort_order, search_term=search, search_fields=["name", "id"])
        total_items = db_service.count("models", filters=filters, search_term=search, search_fields=["name", "id"])
        total_pages = (total_items + page_size - 1) // page_size
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise APIError(
            error=ErrorCode.COMMON_INTERNAL_ERROR,
            override_message=str(e)
        )
    if not db_models:
        return UnifiedAPIResponse(
            success=True,
            message="No models found.",
            data=[]
        )
    
    pagination = PaginationInfo(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        page_size=page_size
    )
    
    public_models: List[ModelPublicView] = []
    for model_db in db_models:
        model_db.created_at = int(model_db.created_at.timestamp()*1000)
        model_db.updated_at = int(model_db.updated_at.timestamp()*1000)
        public_models.append(ModelPublicView.model_validate(model_db))

    return UnifiedAPIResponse(
        success=True,
        message="Models listed successfully.",
        data=public_models,
        pagination=pagination
    )

@router.get(
    "/{model_id}",
    response_model=UnifiedAPIResponse[ModelViewDetails],
    response_model_exclude_none=True, 
    summary="Get Public Details of a Specific Model"
)
async def get_public_model_details(
    model_id: str,
    model_manager: ModelManager = Depends(get_model_mgr),
):
    logger.info(f"Admin request for details of model ID: {model_id}")
    model_db = await model_manager.get_model(model_id)
    
    if not model_db:
        logger.warning(f"Public view: Model ID '{model_id}' not found.")
        raise APIError(
            error=ErrorCode.MODEL_NOT_FOUND
        )
    
    model_details = ModelViewDetails.model_validate(model_db)
    absolute_file_path = model_manager.store.get_model_file_path(model_db.file_path)
    params_yaml = model_manager.store.read_yaml_from_file(model_db.params_path)
    # logger.info(f"Params yaml: {params_yaml}")
    if not params_yaml:
        logger.error(f"Failed to read yaml file for model ID {model_id}.")
        raise APIError(
            error=ErrorCode.MODEL_YAML_NOT_FOUND
        )
      
    model_details.params_yaml = params_yaml
    model_details.model_file_info = ModelFileInfo(
            name=model_db.file_path.split("/")[-1],
            size_bytes=Path(absolute_file_path).stat().st_size,
        )

    return UnifiedAPIResponse(
        success=True,
        message="Model details retrieved successfully.",
        data=model_details
    )


# --- Register a New Model ---
#multipart/form-data
@router.post(
    "",
    response_model=UnifiedAPIResponse[ModelSchema],
    response_model_exclude_none=True, 
    status_code=status.HTTP_201_CREATED,
    summary="Register a New Model"
)
async def register_new_model(
    name: str = Form(...),
    model_file: UploadFile = File(...),
    params_yaml: str = Form(...),
    remark: str = Form(None),
    engine_type: str = Form(...),
    model_manager: ModelManager = Depends(get_model_mgr),
    user_id: str = Depends(get_current_admin_user_id),
    db_service: DatabaseService = Depends(get_db_service)
):
    #check if the model name already exists
    if db_service.count("models", {"name": name}) > 0:
        raise APIError(
            error=ErrorCode.MODEL_EXISTS
        )
    model_create = ModelCreate(
        name=name,
        model_file=model_file,
        params_yaml=params_yaml,
        remark=remark,
        engine_type=engine_type
    )
    temp_file_path: Optional[Path] = None
    try:
        # Save uploaded file to a temporary location
        temp_file_path = TEMP_UPLOAD_DIR / f"temp_{model_file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
        
        registered_model = await model_manager.register_model(
            temp_file_path=str(temp_file_path),
            original_filename=model_file.filename or "model_file", # Ensure filename is not None
            metadata=model_create,
            created_by_user_id=user_id
        )

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise APIError(
            error=ErrorCode.MODEL_REGISTRATION_FAILED,
            override_message=str(e)
        )
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        if model_file:
            await model_file.close()

    return UnifiedAPIResponse(
        success=True,
        message="Model registered successfully.",
        data=registered_model
    )


# --- Update a Model ---
@router.put(
    "/{model_id}",
    response_model=UnifiedAPIResponse[ModelSchema],
    response_model_exclude_none=True,
    summary="Update a Model"
)
async def update_model(
    model_id: str,
    model_file: Optional[UploadFile] = File(None),
    name: Optional[str] = Form(None),
    remark: Optional[str] = Form(None),
    engine_type: Optional[str] = Form(None),
    status: Optional[ModelStatusEnum] = Form(None),
    params_yaml: Optional[str] = Form(None),
    model_manager: ModelManager = Depends(get_model_mgr),
    db_service: DatabaseService = Depends(get_db_service)   
):
    if name and db_service.count("models", {"name": name, "id__ne": model_id}) > 0:
        raise APIError(
            error=ErrorCode.MODEL_EXISTS
        )
    model_update = ModelUpdate(
        name=name,
        remark=remark or "",
        engine_type=engine_type,
        status=status,
        params_yaml=params_yaml
    )
    try:
        if model_file:
            temp_file_path = TEMP_UPLOAD_DIR / f"temp_{model_file.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(model_file.file, buffer)
            model_update.file_path = str(temp_file_path)
            model_update.file_name = model_file.filename
        updated_model = await model_manager.update_model(
            model_id=model_id,
            updates=model_update
        )
    except Exception as e:
        logger.error(f"Failed to update model: {e}")
        raise APIError(
            error=ErrorCode.MODEL_UPDATE_FAILED,
            override_message=str(e)
        )
    return UnifiedAPIResponse(
        success=True,
        message="Model updated successfully.",
        data=updated_model
    )


# --- Delete a Model ---
@router.delete(
    "/{model_id}",
    response_model=UnifiedAPIResponse[bool],
    response_model_exclude_none=True,
    summary="Delete a Model"
)
async def delete_model(
    model_id: str,
    model_manager: ModelManager = Depends(get_model_mgr),
):
    deleted = await model_manager.delete_model(model_id)
    if not deleted:
        raise APIError(
            error=ErrorCode.MODEL_NOT_FOUND
        )
    return UnifiedAPIResponse(
        success=True,
        message="Model deleted successfully.",
        data=deleted
    )


# --- Publish a Model ---
@router.post(
    "/{model_id}/publish",
    response_model=UnifiedAPIResponse,
    response_model_exclude_none=True,
    summary="Publish a Model"
)
async def publish_model(
    model_id: str,
    model_manager: ModelManager = Depends(get_model_mgr),
):
    logger.info(f"Admin request to publish model ID: {model_id}")
    published_model = await model_manager.publish_model(model_id,test_data_config=None)
    if not published_model.success:
        raise APIError(
            error=ErrorDetail.to_error_detail(published_model.error_code),
            override_message=published_model.message
        )
    return UnifiedAPIResponse(
        success=True,
        message=published_model.message,
    )
    
# --- Unpublish a Model ---
@router.post(
    "/{model_id}/unpublish",
    response_model=UnifiedAPIResponse,
    response_model_exclude_none=True,
    summary="Unpublish a Model"
)
async def unpublish_model(
    model_id: str,
    model_manager: ModelManager = Depends(get_model_mgr),
):
    unpublished_model = await model_manager.unpublish_model(model_id)
    if not unpublished_model.success:
        raise APIError(
            error=ErrorCode.MODEL_UNPUBLISH_FAILED,
            override_message=unpublished_model.message
        )
    return UnifiedAPIResponse(
        success=True,
        message=unpublished_model.message,
    )



