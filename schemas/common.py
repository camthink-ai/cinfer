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

class PaginationInfo(BaseModel):
    total_items: int = Field(..., description="Total number of available items")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number (usually starts from 1)")
    page_size: int = Field(..., description="Number of items per page")

class UnifiedAPIResponse(BaseModel, Generic[DataT]):
    """
    Unified API response structure.
    """
    model_config = ConfigDict(from_attributes=True)
    success: bool = Field(True, description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Related prompt information")
    data: Optional[DataT] = Field(None, description="Actual response data payload")
    pagination: Optional[PaginationInfo] = Field(None, description="Pagination information (if applicable)") 
    error_code: Optional[str] = Field(None, description="Application-specific error code, appears when success=False")
    error_details: Optional[Any] = Field(None, description="Detailed error information, such as validation error list, appears when success=False")

#should be updated to array of dicts
class SystemMetrics(BaseModel):
    """
    A schema for system metrics.
    """
    timestamp: str = Field(..., description="The timestamp of the metrics")
    cpu_usage: float = Field(..., description="The CPU usage")
    mem_usage: float = Field(..., description="The memory usage")
    gpu_usage: float = Field(..., description="The GPU usage")


class SystemInfo(BaseModel):
    """
    A schema for system info.
    """
    system_name: str = Field(..., description="The name of the system")
    hardware_acceleration: List[Any] = Field(..., description="The hardware acceleration info")
    os_info: Dict[str, Any] = Field(..., description="The OS info")
    software_name: str = Field(..., description="The software name")
    software_version: str = Field(..., description="The software version info")
    models_stats: Dict[str, Any] = Field(..., description="The models metrics")
    access_tokens_stats: Dict[str, Any] = Field(..., description="The access token metrics")
    
