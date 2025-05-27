# cinfer/schemas/request.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, Union
import uuid

from schemas.models import ValidationResult # Using existing ValidationResult for request validation
from core.engine.base import InferenceInput # For the actual data payload structure

class InferenceRequestParams(BaseModel):
    """
    Parameters for an inference request, complementing the main input data.
    (Placeholder for now, can be extended based on model needs)
    """
    # Example:
    # top_k: Optional[int] = Field(None, description="Return top K results")
    # confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence threshold for predictions")
    custom_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific custom parameters")

class InferenceRequestData(BaseModel):
    """
    Defines the structure for the actual input data for an inference request.
    This corresponds to the 'inputs' field in InferenceRequest.
    The design document (4.3.1) shows 'inputs: dict'. We'll use List[InferenceInput]
    to align with the engine's predict method which takes List[InferenceInput].
    This allows for potential batching at the request level or multiple named inputs.
    """
    # Option 1: Flexible dictionary (as per 4.3.1 diagram for InferenceRequest.inputs)
    # data: Dict[str, Any] = Field(..., description="Input data for the model, e.g., {'image': base64_string}")
    
    # Option 2: List of structured inputs (aligns with engine's predict method)
    input_list: List[InferenceInput] = Field(..., description="List of structured inputs for the model.")


class InferenceRequest(BaseModel):
    """
    Represents an incoming inference request.
    Corresponds to the InferenceRequest class in document section 4.3.1.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the inference request")
    model_id: str = Field(..., description="ID of the AI model to use for inference")
    # inputs: Dict[str, Any] = Field(..., description="Input data payload for the model") # As per 4.3.1 diagram
    inputs: InferenceRequestData = Field(..., description="Structured input data for the model") # Using structured data
    parameters: Optional[InferenceRequestParams] = Field(default_factory=InferenceRequestParams, description="Additional inference parameters")
    priority: int = Field(0, ge=0, description="Request priority (higher value means higher priority)") #
    timeout_ms: Optional[int] = Field(None, gt=0, description="Optional timeout for the request in milliseconds") #

    # For use in priority queue, need to define comparison based on priority
    def __lt__(self, other: 'InferenceRequest') -> bool:
        # Higher numerical priority means it's "less than" for min-heap (PriorityQueue)
        return self.priority > other.priority # Higher number = higher priority

    def validate_request(self) -> ValidationResult: #
        """Performs basic validation on the request."""
        # Basic checks, more can be added (e.g., against model's input schema if available)
        if not self.model_id:
            return ValidationResult(valid=False, errors=["model_id cannot be empty."])
        if not self.inputs.input_list: # Assuming input_list from InferenceRequestData
             return ValidationResult(valid=False, errors=["inputs.input_list cannot be empty."])
        # Further validation can be added here, e.g., checking input data types
        # against a known model input schema (fetched via ModelManager).
        return ValidationResult(valid=True, message="Request is valid.")


class QueueStatus(BaseModel):
    """
    Represents the status of a request queue for a specific model.
    As per document section 4.3.1 QueueManager.get_queue_status.
    """
    model_id: str
    queue_size: int
    active_workers: int
    max_workers: int
    pending_requests: int

class HealthStatus(BaseModel):
    """
    Represents the health status of the request processing system or a component.
    As per document section 4.3.1 RequestProcessor.health_check.
    """
    status: str = Field(..., description="Overall health status (e.g., 'OK', 'DEGRADED', 'ERROR')")
    message: Optional[str] = Field(None, description="Optional message providing more details")
    component_statuses: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Status of individual components")

class SystemStatus(BaseModel):
    """
    Represents the status of the system.
    """
    init: bool = Field(..., description="Overall status of the system")
