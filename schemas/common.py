# cinfer/schemas/common.py
from pydantic import BaseModel
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

class ErrorResponse(BaseModel):
    """
    A generic error response schema.
    """
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = {}