# cinfer/schemas/auth.py
from pydantic import BaseModel, Field
from typing import Optional, List, Any

from .tokens import Token as TokenSchema # To include full token details if needed

class AuthResult(BaseModel):
    """
    Represents the result of an authentication attempt.
    """
    is_authenticated: bool = Field(False, description="True if authentication was successful.")
    user_id: Optional[str] = Field(None, description="ID of the authenticated user, if available.")
    token_id: Optional[str] = Field(None, description="ID of the token used for authentication.")
    token_scopes: Optional[List[str]] = Field(default_factory=list, description="Scopes associated with the token.")
    # token_details: Optional[TokenSchema] = Field(None, description="Full details of the validated token.") # Optionally include full token
    error_message: Optional[str] = Field(None, description="Error message if authentication failed.")
    status_code: Optional[int] = Field(None, description="HTTP status code associated with the auth result (e.g., 401, 403).")

class QuotaResult(BaseModel):
    """
    Represents the result of a quota check.
    """
    allowed: bool = Field(False, description="True if the action is within quota limits.")
    message: Optional[str] = Field(None, description="Details about the quota status.")
    remaining_quota: Optional[int] = Field(None, description="Remaining quota units, if applicable.")
    limit: Optional[int] = Field(None, description="The total quota limit, if applicable.")
    used: Optional[int] = Field(None, description="The current usage count, if applicable.")