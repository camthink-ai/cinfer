# cinfer/api/internal/auth.py
import logging
import uuid
from datetime import datetime, timezone 
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm

from core.database.base import DatabaseService
from core.auth.token import TokenService
from core.auth.permission import Scope
from utils import security
from schemas.tokens import AdminLoginResponse # Use the new response schema
from schemas.users import UserInDB
from schemas.common import Message # For register response
from utils.exceptions import APIError
from utils.errors import ErrorCode
from pydantic import BaseModel
from api.dependencies import get_db_service, get_token_svc_dependency

logger = logging.getLogger(f"cinfer.{__name__}")
router = APIRouter()

