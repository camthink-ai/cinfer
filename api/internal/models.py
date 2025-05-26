# cinfer/api/internal/models.py
import logging
import shutil 
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Body, Form

from core.model.manager import ModelManager
from schemas.models import (
    Model as ModelSchema,
    ModelCreate, # This is ModelMetadataBase
    ModelMetadataBase, # Use this for form data if file is separate
    ModelUpdate,
    DeploymentResult
)
from schemas.common import Message, PaginatedResponse, IdResponse # Assuming PaginatedResponse
from api.dependencies import get_model_mgr # Dependency getter
from api.dependencies import require_admin_user 
router = APIRouter(dependencies=[Depends(require_admin_user)])

logger = logging.getLogger(f"cinfer.{__name__}")


