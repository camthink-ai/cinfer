# cinfer/api/openapi/models.py
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from core.auth.token import TokenService
from core.model.manager import ModelManager
from schemas.models import ModelPublicView
from api.dependencies import get_openapi_auth_result, require_openapi_scopes, get_token_svc_dependency # Updated dependencies
from core.auth.permission import Scope
from schemas.auth import AuthResult
from api.dependencies import get_model_mgr

logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter()

