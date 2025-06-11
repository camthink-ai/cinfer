# cinfer/models/manager.py
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import ValidationError
from core.model.validator import ModelValidator
from .model_store import ModelStore
from core.engine.service import EngineService
from core.database.base import DatabaseService
from schemas.models import (
    Model as ModelSchema,
    ModelCreate,
    ModelUpdate,
    ModelMetadataBase, # For type hinting if directly passing metadata
    DeploymentResult,
    ValidationResult as SchemaValidationResult,
    ModelFileInfo,
    ModelStatusEnum
)
from core.config import get_config_manager # For default versioning etc.

logger = logging.getLogger(f"cinfer.{__name__}")


class ModelManager:
    """
    Central coordinator for AI model lifecycle management.
    Handles model registration, validation, storage, deployment, and CRUD operations.
    As per document section 4.2.1, 4.2.2.
    """
    def __init__(self,
                 db_service: DatabaseService,
                 engine_service: EngineService,
                 model_store: ModelStore,
                 model_validator: ModelValidator):
        self.db: DatabaseService = db_service
        self.engine_service: EngineService = engine_service
        self.store: ModelStore = model_store
        self.validator: ModelValidator = model_validator
        self.config_manager = get_config_manager() # Get global config manager
        self._model_cache: Dict[str, ModelSchema] = {}

    def _generate_id(self) -> str:
        """Generates a unique ID for models."""
        return str(uuid.uuid4())
    
    def _clear_cache_for_model(self, model_id: str) -> None:
        """Clear cached model data."""
        if model_id in self._model_cache:
            del self._model_cache[model_id]
            logger.info(f"Cleared cached model data for model ID: {model_id}")
    
    def _get_model_file_info(self, file_path: str) -> ModelFileInfo:
        """Get information about a model file."""
        return ModelFileInfo(
            name=file_path.split("/")[-1],
            size_bytes=Path(file_path).stat().st_size,
        )

    async def register_model(self,
                             temp_file_path: str,
                             original_filename: str,
                             metadata: ModelCreate,
                             created_by_user_id: Optional[str] = None
                            ) -> Optional[ModelSchema]:
        """
        Registers a new model: validates, stores file, saves metadata to DB.
        Corresponds to the upload process in document section 4.2.3.
        Args:
            temp_file_path (str): Path to the temporary uploaded model file.
            original_filename (str): Original name of the uploaded file.
            metadata (ModelCreate): Pydantic model with metadata.
            created_by_user_id (Optional[str]): ID of the user performing the registration.
        Returns:
            Optional[ModelSchema]: The registered model information, or None on failure.
        """
        # 1. Validate yaml file
        logger.info(f"Validating yaml file for model {metadata.name}")
        input_schema, output_schema = self.validator.validate_yaml_schema(metadata.params_yaml)
        if not input_schema or not output_schema:
            logger.error(f"Yaml file validation failed: {input_schema.errors} {output_schema.errors}")
            raise ValueError(f"Yaml file validation failed: {input_schema.errors} {output_schema.errors}")

        # 2. Validate Model File (static validation using engine)
        # The validator needs engine-specific config from the main config if any.
        # The ModelValidator is initialized with ConfigManager, so it can fetch it.
        file_validation = self.validator.validate_model_file(
            file_path=temp_file_path,
            engine_type=metadata.engine_type
        )
        if not file_validation.valid:
            logger.error(f"Model file validation failed: {file_validation.errors}")  
            raise ValueError(f"Model file validation failed: {file_validation.errors}")
        
        # 3. Validate Model-specific Config 
        if metadata.config:
            config_validation = self.validator.validate_model_config(metadata.config, metadata.engine_type)
            if not config_validation.valid:
                logger.error(f"Model-specific config validation failed: {config_validation.errors}")  
                raise ValueError(f"Model-specific config validation failed: {config_validation.errors}")
            
        

        # 4. Prepare model record for DB
        model_id = self._generate_id()

        # 5. Store Model File using ModelStore
        # ModelStore saves to a structured path like /<model_id>/<filename>
        relative_file_path = self.store.save_model_file(
            temp_file_path=temp_file_path,
            model_id=model_id,
            original_filename=original_filename
        )
        if not relative_file_path:
            logger.error(f"Failed to save model file for model ID {model_id}")  
            raise ValueError(f"Failed to save model file for model ID {model_id}.")
        
        # 6. Store yaml file
        yaml_file_path = self.store.save_yaml_file(
            params_yaml=metadata.params_yaml,
            model_id=model_id
        )
        if not yaml_file_path:
            logger.error(f"Failed to save yaml file for model ID {model_id}")  
            raise ValueError(f"Failed to save yaml file for model ID {model_id}.")
        
        # 7. Construct DB record
        logger.info(f"Constructing DB record for model {metadata.name}")
        now = datetime.utcnow()
        model_data_for_db = {
            "id": model_id,
            "name": metadata.name,
            "remark": metadata.remark,
            "engine_type": metadata.engine_type,
            "file_path": relative_file_path, # Relative path from ModelStore
            "params_path": yaml_file_path, 
            "created_by": created_by_user_id,
            "created_at": now,
            "updated_at": now,
            "status": ModelStatusEnum.DRAFT, # Initial status as per diagram/common practice
            "input_schema": input_schema,
            "output_schema": output_schema,
            # "model_config_json": metadata.config.model_dump_json() if metadata.config else None,
            # "input_schema_json": metadata.input_schema.model_dump_json() if metadata.input_schema else None,
            # "output_schema_json": metadata.output_schema.model_dump_json() if metadata.output_schema else None,
        }

        # 6. Save Model Metadata to Database
        inserted_id = self.db.insert("models", model_data_for_db)
        if not inserted_id : # Check if insert failed
            logger.error(f"Failed to save model metadata to database for model ID {model_id}.")  
            # If DB insert fails, try to delete the stored file
            self.store.delete_model_and_yaml_file(relative_file_path, yaml_file_path)
            raise ValueError(f"Failed to save model metadata to database for model ID {model_id}.")

        logger.info(f"Model '{metadata.name}' (ID: {model_id}) registered successfully.")  
        # Construct the full ModelSchema object to return, including the Pydantic config, input/output schemas
        # These come from the input `metadata` object.
        final_model_data = {**model_data_for_db,
                              "config": metadata.config
                             }
        return ModelSchema(**final_model_data)


    async def get_model(self, model_id: str) -> Optional[ModelSchema]:
        """Retrieves model details from the database."""
        if model_id in self._model_cache:
            logger.info(f"Cache hit for model ID: {model_id}")
            return self._model_cache[model_id]
        logger.info(f"Cache miss for model ID: {model_id}")
        # Get model data from DB
        model_data = self.db.find_one("models", {"id": model_id})
        if model_data:
            try:
                if model_data.get("input_schema"):
                    model_data["input_schema"] = json.loads(model_data["input_schema"])
                if model_data.get("output_schema"):
                    model_data["output_schema"] = json.loads(model_data["output_schema"])
                model_schema = ModelSchema(**model_data)
                params_yaml = self.store.read_yaml_from_file(model_schema.params_path)
                if not params_yaml:
                    logger.error(f"Failed to read yaml file for model ID {model_id}.")
                    return None
                model_schema.input_schema, model_schema.output_schema = self.validator.validate_yaml_schema(params_yaml)
                self._model_cache[model_id] = model_schema
                return model_schema
            except ValidationError as e:
                logger.error(f"Data from DB for model {model_id} failed ModelSchema validation: {e}")  
                return None
        return None

    async def list_models(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 10
    ) -> List[ModelSchema]:
        """Lists all models, optionally filtered."""
        model_data_list = self.db.find("models", filters=filters or {}, limit=page_size, offset=(page - 1) * page_size, order_by="created_at DESC")
     
        models = []
        for data in model_data_list:
            try:
                if data.get("input_schema"):
                    data["input_schema"] = json.loads(data["input_schema"])
                if data.get("output_schema"):
                    data["output_schema"] = json.loads(data["output_schema"])
                models.append(ModelSchema(**data))
            except ValidationError: # Skip invalid records
                logger.error(f"Skipping model record due to validation error: {data.get('id')}")  
                continue
        return models

    async def update_model(self, model_id: str, updates: ModelUpdate) -> Optional[ModelSchema]:
        """
        Updates model metadata in the database.
        File updates would require a more complex process (don't have it yet).
        """
        update_data = updates.model_dump(exclude_unset=True) # Get only provided fields
        if not update_data:
            logger.info("No update data provided.")  
            return self.get_model(model_id) # Return current model if no updates

        update_data["updated_at"] = datetime.utcnow()
        
        rows_updated = self.db.update("models", {"id": model_id}, update_data)
        if rows_updated > 0:
            logger.info(f"Model {model_id} updated successfully.")  
            self._clear_cache_for_model(model_id)
            return self.get_model(model_id)
        logger.error(f"Model {model_id} not found or update failed.")  
        return None
    


    async def delete_model(self, model_id: str) -> bool:
        """
        Deletes a model: its file from store and record from DB.
        Ensures model is unloaded first.
        """
        logger.info(f"Attempting to delete model ID: {model_id}")  
        model_info = await self.get_model(model_id)
        if not model_info:
            logger.error(f"Model {model_id} not found for deletion.")  
            return False # Or True if "not found" is considered a successful deletion state

        # 1. Unload model from engine service if loaded
        self.engine_service.unload_model(model_id) # unload_model is idempotent

        # 2. Delete model file and yaml file from store
        if model_info.file_path: # file_path is relative
            if not self.store.delete_model_and_yaml_file(model_info.file_path, model_info.params_path):
                logger.warning(f"Warning: Failed to delete model file {model_info.file_path} for model {model_id}.")  
                # Continue to delete DB record? Or fail here? 
                pass
        
        # 3. Delete model record from DB
        rows_deleted = self.db.delete("models", {"id": model_id})
        if rows_deleted > 0:
            logger.info(f"Model {model_id} deleted successfully from database.")  
            self._clear_cache_for_model(model_id)
            return True
        
        logger.error(f"Failed to delete model {model_id} from database (was it already deleted?).")  
        return False


    def publish_model(self, model_id: str, test_data_config: Optional[Dict[str, Any]] = None) -> DeploymentResult:
        """
        Publishes a model: loads into engine, tests inference, updates status.
        Corresponds to sequence in 4.2.3.
        Args:
            model_id (str): ID of the model to publish.
            test_data_config (Optional[Dict[str, Any]]): Configuration for test inference data.
        Returns:
            DeploymentResult containing success status and message.
        """
        pass

    def unpublish_model(self, model_id: str) -> DeploymentResult:
        """
        Unpublishes a model: unloads from engine, updates status to "draft" or "deprecated". 
        """
        pass