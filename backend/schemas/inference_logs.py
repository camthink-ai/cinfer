# cinfer/schemas/inference_logs.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime

class InferenceLogBase(BaseModel):
    """
    Base schema for inference log data.
    """
    model_id: str = Field(..., description="ID of the model used for inference") #
    token_id: Optional[str] = Field(None, description="ID of the token used for authorization (if any)") #
    request_id: Optional[str] = Field(None, description="Unique ID for the inference request") #
    client_ip: Optional[str] = Field(None, description="IP address of the client making the request") #
    # request_data and response_data are stored as TEXT (JSON) in DB
    request_data: Optional[Dict[str, Any]] = Field(None, description="The request payload sent for inference")
    response_data: Optional[Dict[str, Any]] = Field(None, description="The response payload received from inference")
    status: Optional[str] = Field(None, description="Status of the inference (e.g., 'success', 'error')") #
    error_message: Optional[str] = Field(None, description="Error message if inference failed") #
    latency_ms: Optional[float] = Field(None, description="Time taken for the inference in milliseconds") #


class InferenceLogCreate(InferenceLogBase):
    """
    Schema for creating a new inference log entry.
    Most fields are set by the system during or after an inference call.
    """
    pass


class InferenceLogInDB(InferenceLogBase):
    """
    Schema representing an inference log entry as stored in the database.
    """
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(..., description="Unique log entry ID") #
    created_at: datetime = Field(..., description="Timestamp when the log entry was created") #


class InferenceLog(InferenceLogInDB):
    """
    Schema for representing an inference log in API responses.
    """
    pass