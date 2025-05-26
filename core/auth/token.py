# core/auth/token.py
import uuid
import logging
import secrets # For generating opaque refresh tokens
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Dict, Any

from core.database.base import DatabaseService
# AccessToken Schemas for external tokens
from schemas.tokens import Token as AccessTokenSchema, TokenCreate as AccessTokenCreateSchema, TokenUpdate as AccessTokenUpdateSchema, TokenDetail
# Schemas for admin login
from schemas.tokens import AdminLoginResponse
from core.auth.permission import Scope # Import Scope
from utils import security
from core.config import get_config_manager
from utils.exceptions import APIError
from utils.errors import ErrorCode
from fastapi import status

logger = logging.getLogger(f"cinfer.{__name__}")

class TokenService:
    def __init__(self, db_service: DatabaseService):
        self.db: DatabaseService = db_service
        self.config = get_config_manager()
        self.admin_at_expire_minutes = self.config.get_config("auth.admin_access_token_expire_minutes", 1440) # Default 24h as per request
        self.admin_rt_expire_minutes = self.config.get_config("auth.admin_refresh_token_expire_minutes", 1440) # Default 24h as per request

   