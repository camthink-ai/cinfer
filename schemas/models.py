# cinfer/schemas/models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

# Reflects ModelMetadata mentioned in 4.2.1 ModelManager.register_model
class ModelMetadataBase(BaseModel):
    """
    Base schema for AI model metadata.
    """
    name: str = Field(..., description="Name of the AI model") #
    description: Optional[str] = Field(None, description="Detailed description of the model") #
    engine_type: str = Field(..., description="Type of inference engine required (e.g., 'onnx', 'tensorrt', 'pytorch')") #
    # These are from the Model class in 4.2.1, not directly in models table but part of metadata
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Schema definition for model inputs") #
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Schema definition for model outputs") #
    # Config for model-specific config, now it is a placeholder
    config: Optional[Dict[str, Any]] = None


class ModelCreate(ModelMetadataBase):
    """
    Schema for registering/creating a new AI model.
    File path might be handled separately or as part of a multipart request.
    The 'file_path', 'config_path', 'params_path' from DB schema (5.3.2)
    are typically set by the system upon file upload.
    """
    # created_by: Optional[str] = Field(None, description="ID of the user who created the model") # System will set this
    pass


class ModelUpdate(BaseModel):
    """
    Schema for updating an existing AI model. All fields are optional.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    engine_type: Optional[str] = None # Changing engine type might be complex
    status: Optional[str] = None # e.g., 'draft', 'published', 'deprecated'
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    # Potentially allow updating file_path, config_path, params_path if model files are replaced
    file_path: Optional[str] = Field(None, description="Path to the model file (system-managed)")
    params_path: Optional[str] = Field(None, description="Path to additional model parameters file (system-managed)")



class ModelInDBBase(ModelMetadataBase):
    """
    Base schema for model data as stored in/retrieved from the database.
    """
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(..., description="Unique model ID") #
    file_path: str = Field(..., description="Path to the model file") #
    params_path: Optional[str] = Field(None, description="Path to additional model parameters file") #
    created_by: Optional[str] = Field(None, description="ID of the user who created the model") #
    created_at: datetime = Field(..., description="Timestamp of model creation") #
    updated_at: datetime = Field(..., description="Timestamp of last model update") #
    status: str = Field("draft", description="Current status of the model (e.g., 'draft', 'published', 'deprecated')") #


class Model(ModelInDBBase):
    """
    Schema for representing an AI model in API responses.
    """
    # input_schema, output_schema, config are inherited from ModelMetadataBase
    # and are part of the Model class in section 4.2.1
    pass

# Schema for deployment result mentioned in 4.2.1 ModelManager.load_model
class DeploymentResult(BaseModel):
    success: bool
    message: Optional[str] = None
    model_id: Optional[str] = None
    status: Optional[str] = None # e.g., 'LOADING', 'LOADED', 'ERROR'

# Schema for validation result mentioned in 4.2.1 Model.validate_input and ModelValidator
class ValidationResult(BaseModel):
    valid: bool
    errors: Optional[List[str]] = None
    message: Optional[str] = None

class ModelPublicView(ModelMetadataBase): # Inherits common metadata fields
    """
    Schema for publicly exposing AI model details.
    Omits sensitive information like internal file paths.
    """
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(..., description="Unique model ID")
    # name, description, engine_type, input_schema, output_schema, config are inherited
    status: str = Field(..., description="Current status of the model (e.g., 'published')")
    created_at: datetime = Field(..., description="Timestamp of model creation")
    updated_at: datetime = Field(..., description="Timestamp of last model update")
    # We explicitly OMIT file_path, params_path, created_by
    # as they might be internal details.