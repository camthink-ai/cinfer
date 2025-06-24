from typing import Optional, Dict, Any
from fastapi import HTTPException

from typing import Optional, Dict, Any
from fastapi import HTTPException
from .errors import ErrorDetail, ErrorCode

class APIError(HTTPException):
    """
    API layer exceptions.
    """
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

class CoreServiceException(Exception):
    """
    Core layer services 
    It carries complete error information through an ErrorDetail object.
    """
    def __init__(
        self,
        error: ErrorDetail, 
        details: Optional[Any] = None,
        override_message: Optional[str] = None
    ):
        """
        Args:
            error (ErrorDetail): ErrorCode class defined ErrorDetail instance.
            override_message (Optional[str]): override the default message of ErrorDetail.
            details (Optional[Any]): additional structured error information (e.g., list of failed field validations).
        """
        self.status_code = error.status_code
        self.error_code = error.code
        self.details = details or {}
        #override message if provided
        self.message = override_message or error.message
        super().__init__(self.message)