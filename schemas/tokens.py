# cinfer/schemas/tokens.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from schemas.users import UserBase
class TokenBase(BaseModel):
    """
    Base schema for API token data.
    """
    name: str = Field(..., description="A descriptive name for the token") #
    is_active: Optional[bool] = Field(True, description="Whether the token is currently active") #
    allowed_models: Optional[List[str]] = Field(None, description="List of model IDs this token can access (JSON array in DB)") #
    ip_whitelist: Optional[List[str]] = Field(None, description="List of IP addresses allowed to use this token (JSON array in DB)") #
    rate_limit: Optional[int] = Field(100, description="Requests per minute allowed for this token") #
    monthly_limit: Optional[int] = Field(10000, description="Total requests per month allowed for this token") #
    remark: Optional[str] = Field(None, description="Additional remarks or notes for the token") #

class TokenCreate(TokenBase):
    """
    Schema for creating a new API token.
    Some fields might be set by the system or have defaults.
    """
    expires_at: Optional[datetime] = Field(None, description="Timestamp when the token expires. If None, might not expire or use system default.") #

class TokenUpdate(BaseModel):
    """
    Schema for updating an existing API token.
    All fields are optional for partial updates.
    """
    name: Optional[str] = None
    is_active: Optional[bool] = None
    allowed_models: Optional[List[str]] = None
    ip_whitelist: Optional[List[str]] = None
    rate_limit: Optional[int] = None
    monthly_limit: Optional[int] = None
    expires_at: Optional[datetime] = None # Allow extending/changing expiry
    remark: Optional[str] = None


class TokenInDBBase(TokenBase):
    """
    Base schema for token data as stored in the database.
    """
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(..., description="Unique token ID") #
    value: str = Field(..., description="The actual token string (usually JWT or a secure random string, should be write-only or system generated)") #
    created_at: datetime = Field(..., description="Timestamp of token creation") #
    expires_at: Optional[datetime] = Field(None, description="Timestamp when the token expires") #
    used_count: Optional[int] = Field(0, description="How many times this token has been used (approximate or for monthly counts)") #

class Token(TokenInDBBase):
    """
    Schema for representing an API token in API responses.
    The actual 'value' (token string) is often returned only upon creation.
    For listing/getting tokens, the 'value' might be omitted or partially masked for security.
    The design doc 4.4.4 implies 'token' (value) is shown.
    """
    # value field is inherited and will be shown as per 4.4.4
    pass


class AdminLoginResponse(BaseModel): 
    access_token: str = Field(description="The short-lived Access Token (for X-Auth-Token header)")
    refresh_token: str = Field(description="The longer-lived Refresh Token")
    expires_in: int = Field(description="Access token validity period in seconds") # 
    #user: Optional[UserBase] = Field(description="The user details")


# Schema for the Token structure shown in design doc 4.4.4
class TokenDetail(BaseModel):
    """
    Represents the detailed structure of a token as shown in documentation [4.4.4].
    """
    model_config = ConfigDict(from_attributes=True)
    token: str = Field(description="The token string value itself.") #
    name: str #
    created_at: datetime #
    expires_at: Optional[datetime] = None #
    is_active: bool #
    allowed_models: Optional[List[str]] = None #
    ip_whitelist: Optional[List[str]] = None #
    rate_limit: Optional[int] = None #
    monthly_limit: Optional[int] = None #
    used_count: Optional[int] = None #
