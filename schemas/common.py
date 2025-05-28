# cinfer/schemas/common.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, TypeVar, Generic, List, Dict, Any



DataT = TypeVar('DataT')

class Message(BaseModel):
    """
    A simple message schema for generic API responses.
    """
    message: str

class PaginatedResponse(BaseModel, Generic[DataT]):
    """
    A generic paginated response schema.
    """
    total: int
    page: int
    size: int
    items: List[DataT]
    # Example: has_more: bool

class IdResponse(BaseModel):
    """
    A simple response schema for returning an ID.
    """
    id: str


class UnifiedAPIResponse(BaseModel, Generic[DataT]):
    """
    Unified API response structure.
    """
    model_config = ConfigDict(from_attributes=True)
    success: bool = Field(True, description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Related prompt information")
    data: Optional[DataT] = Field(None, description="Actual response data payload")
    error_code: Optional[str] = Field(None, description="Application-specific error code, appears when success=False")
    error_details: Optional[Any] = Field(None, description="Detailed error information, such as validation error list, appears when success=False")