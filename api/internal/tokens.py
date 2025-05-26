# cinfer/api/internal/tokens.py
import logging
from typing import List, Optional, Annotated
from datetime import timedelta
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status, Body


from core.auth.token import TokenService
from schemas.tokens import (
    Token as AccessTokenSchema, # Represents an Access Token
    TokenCreate as AccessTokenCreateSchema,
    TokenUpdate as AccessTokenUpdateSchema,
    TokenDetail # For response of created Access Token
)
from schemas.common import Message, IdResponse
# Use the new admin-specific authentication dependency
from api.dependencies import require_admin_user, get_token_svc_dependency
from schemas.auth import AuthResult # To use AuthResult from require_admin_user

logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter(dependencies=[Depends(require_admin_user)]) 


