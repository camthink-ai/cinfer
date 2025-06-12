from typing import Optional, Dict, Any
from fastapi import HTTPException

from typing import Optional, Dict, Any
from fastapi import HTTPException
from .errors import ErrorDetail, ErrorCode

class APIError(HTTPException):
    def __init__(
        self,
        error: ErrorDetail,
        details: Optional[Dict[str, Any]] = None,
        override_message: Optional[str] = None,
    ):
        self.error_code = error.code
        self.details = details or {}
        #override message if provided
        message = override_message or error.message
        super().__init__(status_code=error.status_code, detail=message)