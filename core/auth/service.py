# core/auth/service.py
import logging
from typing import Optional, List, Dict, Any

from fastapi import Request as FastAPIRequest

from .token import TokenService
from .ip_filter import IPFilter
from .rate_limiter import RateLimiter
from schemas.auth import AuthResult, QuotaResult
from schemas.tokens import Token as AccessTokenSchema # For type hint of validated external token
from core.auth.permission import Scope, check_scopes as util_check_scopes
# security import not directly needed here for decoding, as TokenService handles it.

logger = logging.getLogger(f"cinfer.{__name__}")

class AuthService:
    def __init__(self,
                 token_service: TokenService,
                 ip_filter: IPFilter,
                 rate_limiter: RateLimiter):
        self.token_service: TokenService = token_service
        self.ip_filter: IPFilter = ip_filter
        self.rate_limiter: RateLimiter = rate_limiter

   