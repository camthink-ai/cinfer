# cinfer/schemas/auth.py
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from utils.errors import ErrorCode
from schemas.tokens import AccessTokenSchema
class AuthResult(BaseModel):
    """
    Represents the result of an authentication attempt.
    """
    is_authenticated: bool = Field(False, description="True if authentication was successful.")
    user_id: Optional[str] = Field(None, description="ID of the authenticated user, if available.")
    token_id: Optional[str] = Field(None, description="ID of the token used for authentication.")
    token_scopes: Optional[List[str]] = Field(default_factory=list, description="Scopes associated with the token.")
    error_code: Optional[Dict[str, Any]] = Field(None, description="Error code if authentication failed.")

class ValidateTokenResult(BaseModel):
    """
    Represents the result of a token validation attempt.
    """
    is_valid: bool = Field(False, description="True if token is valid.")
    token_data: Optional[AccessTokenSchema] = Field(None, description="Token data if token is valid.")
    error_code: Optional[Dict[str, Any]] = Field(None, description="Error code if token validation failed.")
