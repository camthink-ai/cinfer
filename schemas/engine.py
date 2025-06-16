from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# --- Data Models for Engine Information ---
class EngineInfo(BaseModel):
    """
    Holds information about a specific engine instance.
    Returned by IEngine.get_info().
    """
    engine_name: str = Field(..., description="Name of the inference engine (e.g., ONNX, TensorRT)")
    engine_version: Optional[str] = Field(None, description="Version of the inference engine")
    model_loaded: bool = Field(False, description="Indicates if a model is currently loaded")
    loaded_model_path: Optional[str] = Field(None, description="Path of the currently loaded model")
    available_devices: List[str] = Field(default_factory=list, description="List of available computation devices (e.g., ['CPU', 'GPU:0'])")
    current_device: Optional[str] = Field(None, description="The device currently used by the engine for the loaded model")
    engine_status: str = Field("uninitialized", description="Current status of the engine (e.g., uninitialized, ready, error)")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Any other engine-specific information")

class PropertyDefinition(BaseModel):
    description: str
    type: str
    required: bool
    items: Optional['PropertyDefinition'] = None
    properties: Optional[Dict[str, 'PropertyDefinition']] = None
    minLength: Optional[int] = None
    maxLength: Optional[int] = None

PropertyDefinition.model_rebuild()

class InputOutputDefinition(BaseModel):
    name: str
    description: str
    type: str
    format: Optional[str] = None
    required: bool
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Optional[Any] = None
    items: Optional['PropertyDefinition'] = None

class ModelIODefinitionFile(BaseModel):
    #description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    inputs: List[InputOutputDefinition]
    outputs: List[InputOutputDefinition]


class InferenceInput(BaseModel):
    """
    Represents a single input for inference.
    This is a generic placeholder; specific models might need more structured inputs.
    """
    data: Any # Could be numpy array, image path, raw bytes, etc.
    metadata: Optional[Dict[str, Any]] = None # e.g., input tensor name

class InferenceOutput(BaseModel):
    """
    Represents a single output from inference.
    """
    data: Any # Could be numpy array, list of detections, etc.
    metadata: Optional[Dict[str, Any]] = None # e.g., output tensor name

class InferenceResult(BaseModel):
    """
    Represents the result of an inference process, including test inference.
    Returned by BaseEngine.test_inference() and potentially predict methods.
    """
    success: bool
    outputs: Optional[List[InferenceOutput]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None

class ResourceRequirements(BaseModel):
    """
    Describes the resource requirements of an engine or model.
    Returned by BaseEngine.get_resource_requirements().
    """
    cpu_cores: Optional[float] = Field(None, description="Number of CPU cores required/used")
    memory_gb: Optional[float] = Field(None, description="Memory in GB required/used")
    gpu_count: Optional[int] = Field(None, description="Number of GPUs required/used")
    gpu_memory_gb: Optional[float] = Field(None, description="GPU memory in GB per GPU required/used")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom resource requirements")

class Infere(BaseModel):
    """
    Represents the output of an inference process.
    """
    data: Any # Could be numpy array, list of detections, etc.
    metadata: Optional[Dict[str, Any]] = None # e.g., output tensor name

class InferenceResponse(BaseModel):
    outputs: Optional[InferenceOutput] = None
    processing_time_ms: Optional[float] = None

class InferenceBatchResponse(BaseModel):
    outputs: Optional[List[InferenceOutput]] = None
    processing_time_ms: Optional[float] = None
