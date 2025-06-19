# cinfer/schemas/models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from fastapi import UploadFile, File
from enum import Enum # 导入 Enum


class ModelStatusEnum(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ERROR = "error"


# --- Base Model ---

class ModelMetadataBase(BaseModel):
    """Base schema for AI model metadata."""
    name: str = Field(..., description="Name of the AI model")
    remark: Optional[str] = Field(None, description="Detailed description of the model") # 修改：remark -> description
    engine_type: str = Field(..., description="Type of inference engine required (e.g., 'onnx', 'tensorrt')")
    input_schema: Optional[List[Dict[str, Any]]] = Field(None, description="Schema definition for model inputs")
    output_schema: Optional[List[Dict[str, Any]]] = Field(None, description="Schema definition for model outputs")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific config, e.g., for processing strategy") # 恢复并使用 config

class ModelInDBBase(ModelMetadataBase):
    """Base schema for model data as stored in/retrieved from the database."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique model ID")
    file_path: str = Field(..., description="Path to the model file")
    params_path: Optional[str] = Field(None, description="Path to additional model parameters file")
    created_by: Optional[str] = Field(None, description="ID of the user who created the model")
    created_at: Union[datetime, int] = Field(..., description="Timestamp of model creation")
    updated_at: Union[datetime, int] = Field(..., description="Timestamp of last model update")
    status: ModelStatusEnum = Field(ModelStatusEnum.DRAFT, description="Current status of the model") # 修改：使用枚举

# --- API input model ---

class ModelCreate(ModelMetadataBase): 
    """Schema for registering a new AI model."""
    model_file: UploadFile 
    params_yaml: Optional[str] = Field(None, description="YAML content for model parameters, e.g., for dynamic I/O schemas")

class ModelUpdate(BaseModel):
    """Schema for updating an existing AI model. All fields are optional."""
    name: Optional[str] = None
    remark: Optional[str] = None 
    engine_type: Optional[str] = None
    status: Optional[ModelStatusEnum] = None 
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    params_yaml: Optional[str] = None
    params_path: Optional[str] = None
    input_schema: Optional[List[Dict[str, Any]]] = None
    output_schema: Optional[List[Dict[str, Any]]] = None
    config: Optional[Dict[str, Any]] = Field(None, description="Model-specific config")


# --- API output model ---

class Model(ModelInDBBase):
    """Schema for representing a full AI model in internal API responses."""
    pass

class ModelFileInfo(BaseModel):
    name: str = Field(..., description="Name of the model file")
    size_bytes: int = Field(..., description="Size of the model file in bytes")

class ModelViewDetails(Model):
    """Schema for representing detailed model view in API responses."""
    model_file_info: Optional[ModelFileInfo] = Field(None, description="Information about the model file")
    params_yaml: Optional[str] = Field(None, description="YAML content for model parameters")


class ModelPublicView(BaseModel): 
    """Schema for publicly exposing AI model details."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Name of the AI model")
    remark: Optional[str] = Field(None, description="Detailed description of the model")
    engine_type: str = Field(..., description="Type of inference engine required")
    status: ModelStatusEnum = Field(..., description="Current status of the model (usually 'published')")
    created_at: int = Field(..., description="Timestamp of model creation")
    updated_at: int = Field(..., description="Timestamp of last model update")

class ModelSortByEnum(str, Enum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"

class ModelSortOrderEnum(str, Enum):
    ASC = "asc"
    DESC = "desc"

class ModelOpenAPIView(BaseModel):
    """Schema for publicly exposing AI model details for OpenAPI."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Name of the AI model")
    remark: Optional[str] = Field(None, description="Detailed description of the model")
    engine_type: str = Field(..., description="Type of inference engine required")

class ModelOpenApiDetails(BaseModel):
    """Schema for publicly exposing AI model details for OpenAPI."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique model ID")
    name: str = Field(..., description="Name of the AI model")
    remark: Optional[str] = Field(None, description="Detailed description of the model")
    engine_type: str = Field(..., description="Type of inference engine required")
    input_schema: Optional[List[Dict[str, Any]]] = Field(None, description="Schema definition for model inputs")
    output_schema: Optional[List[Dict[str, Any]]] = Field(None, description="Schema definition for model outputs")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific config, e.g., for processing strategy")

# --- operation result model ---

class DeploymentResult(BaseModel):
    success: bool
    message: Optional[str] = None
    model_id: Optional[str] = None
    status: Optional[ModelStatusEnum] = None 

class ValidationResult(BaseModel):
    valid: bool
    errors: Optional[List[str]] = None
    message: Optional[str] = None
    data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None