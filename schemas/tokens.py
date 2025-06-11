# cinfer/schemas/tokens.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from schemas.users import UserBase
from enum import Enum

class AccessTokenSchema(BaseModel): 
    id: str = Field(description="The unique identifier for the access token")
    user_id: str = Field(description="The user ID associated with the access token")
    name: str = Field(description="The name of the access token")
    token_value_hash: str = Field(description="The hashed value of the access token")
    token_value_view: str = Field(description="The viewable value of the access token")
    created_at: datetime = Field(description="The creation timestamp of the access token")
    updated_at: datetime = Field(description="The last update timestamp of the access token")
    status: str = Field(description="The status of the access token")
    allowed_models: List[str] = Field(description="The models allowed to be used with the access token")
    ip_whitelist: List[str] = Field(description="The IP addresses allowed to use the access token")
    rate_limit: Optional[int] = Field(description="The rate limit for the access token")
    monthly_limit: Optional[int] = Field(description="The monthly limit for the access token")
    used_count: Optional[int] = Field(description="The number of times the access token has been used")
    remark: Optional[str] = Field(description="The remark for the access token")
    model_config = ConfigDict(from_attributes=True)


class AccessTokenUpdateSchema(BaseModel):
    name: Optional[str] = None
    allowed_models: Optional[List[str]] = None
    ip_whitelist: Optional[List[str]] = None
    rate_limit: Optional[int] = None
    monthly_limit: Optional[int] = None
    remark: Optional[str] = None

class AccessTokenCreateRequest(BaseModel):
    name: str
    allowed_models: Optional[List[str]] = None
    ip_whitelist: Optional[List[str]] = None
    rate_limit: Optional[int] = None
    monthly_limit: Optional[int] = None
    remark: Optional[str] = None

class AccessTokenDetail(BaseModel):
    id: str
    name: str
    token: str
    ip_whitelist: List[str]
    allowed_models: List[str]
    rate_limit: int
    monthly_limit: int
    created_at: int
    updated_at: int
    remaining_requests: int
    remark: str
    status: str

class AccessTokenSortByEnum(str, Enum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"

class AccessTokenSortOrderEnum(str, Enum):
    ASC = "asc"
    DESC = "desc"

class AccessTokenStatusQueryEnum(str, Enum): 
    ACTIVE = "active"
    DISABLED = "disabled"

class AccessTokenStatus(str, Enum): 
    ACTIVE = "active"
    DISABLED = "disabled"
    REVOKED = "revoked"

class AdminLoginResponse(BaseModel): 
    access_token: str = Field(description="The short-lived Access Token (for X-Auth-Token header)")
    refresh_token: str = Field(description="The longer-lived Refresh Token")
    expires_in: int = Field(description="Access token validity period in seconds") # 
    #user: Optional[UserBase] = Field(description="The user details")


