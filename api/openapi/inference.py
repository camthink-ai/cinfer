# cinfer/api/openapi/inference.py
import logging
from typing import Dict, Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status, Body, Path as FastApiPath
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from core.request.processor import RequestProcessor
from schemas.request import InferenceRequestData # Re-using InferenceRequestData directly
from core.engine.base import InferenceResult
# Updated dependencies for OpenAPI
from api.dependencies import get_openapi_auth_result, require_openapi_scopes, get_request_proc, get_token_svc_dependency
from core.auth.permission import Scope
from schemas.auth import AuthResult
from core.auth.token import TokenService # For fetching token details

logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter()

