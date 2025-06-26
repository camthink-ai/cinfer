# cinfer/models/manager.py
import uuid
import json
import logging
import base64
import io
from PIL import Image
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
    ModelStatusEnum,
    ModelSortByEnum,
    ModelSortOrderEnum
)
from core.config import get_config_manager # For default versioning etc.
from schemas.engine import InferenceInput, InferenceResult, InputOutputDefinition, ModelIODefinitionFile
from utils.errors import ErrorCode


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
    
    def _generate_dummy_image_base64(self, width: int = 64, height: int = 64) -> str:
        """Generate a simple Base64 encoded image string."""
        img = Image.new('RGB', (width, height), color = 'red')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

    def _generate_for_input(self, definition: InputOutputDefinition) -> Any:
        """Generate sample test inputs for a single input definition."""
        # if default is not None and not required, use default
        if definition.default is not None and not definition.required:
            return definition.default

        if definition.type == "string":
            logger.info(f"Generating sample test input for string type: {definition.format}")
            definition.format = definition.format.split(" | ")
            logger.info(f"Definition format: {definition.format}")
            if "base64" in definition.format:
                return self._generate_dummy_image_base64()
            elif "uri" in definition.format:
                return "https://example.com/test_image.jpg"
            else:
                return "sample_text"
        
        elif definition.type == "float":
            min_val = definition.minimum if definition.minimum is not None else 0.0
            max_val = definition.maximum if definition.maximum is not None else 1.0
            return (min_val + max_val) / 2.0 # return a middle value
            
        elif definition.type == "integer":
            min_val = int(definition.minimum) if definition.minimum is not None else 0
            max_val = int(definition.maximum) if definition.maximum is not None else 10
            return (min_val + max_val) // 2

        elif definition.type == "boolean":
            return True
        
        # for array and object, need more complex recursive logic, here simplified
        elif definition.type == "array":
            return []
        elif definition.type == "object":
            return {}
            
        return None

    def _generate_sample_test_inputs(self, input_definitions: List[InputOutputDefinition]) -> List[InferenceInput]:
        """
        Generate a complete request body dictionary for the model's inputs.
        """
        request_body = []
        data_body = {}
        metadata_body = {}
        for i in range(len(input_definitions)):
            if i == 0:
                data_body = self._generate_for_input(input_definitions[i])
            else:
                metadata_body[input_definitions[i].name] = self._generate_for_input(input_definitions[i])
        request_body.append(InferenceInput(data=data_body, metadata=metadata_body))
        return request_body

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
        input_validation, output_validation, config_validation = self.validator.validate_yaml_schema(metadata.params_yaml)
        if not input_validation.valid or not output_validation.valid:
            error_messages = []
            if not input_validation.valid and input_validation.errors:
                error_messages.extend(input_validation.errors)
            if not output_validation.valid and output_validation.errors:
                error_messages.extend(output_validation.errors)
            if not config_validation.valid and config_validation.errors:
                error_messages.extend(config_validation.errors)
            error_msg = "YAML validation failed: " + "; ".join(error_messages)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Input schema: {input_validation.data}")
        logger.info(f"Output schema: {output_validation.data}")
        logger.info(f"Config schema: {config_validation.data}")

        # 2. Validate Model File (static validation using engine)
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
            "file_path": relative_file_path,
            "params_path": yaml_file_path, 
            "created_by": created_by_user_id,
            "created_at": now,
            "updated_at": now,
            "status": ModelStatusEnum.DRAFT,
            "input_schema": input_validation.data,
            "output_schema": output_validation.data,
            "config": config_validation.data
        }

        # 8. Save Model Metadata to Database
        inserted_id = self.db.insert("models", model_data_for_db)
        if not inserted_id:
            logger.error(f"Failed to save model metadata to database for model ID {model_id}.")  
            # If DB insert fails, try to delete the stored file
            self.store.delete_model_and_yaml_file(relative_file_path, yaml_file_path)
            raise ValueError(f"Failed to save model metadata to database for model ID {model_id}.")

        logger.info(f"Model '{metadata.name}' (ID: {model_id}) registered successfully.")  
        return ModelSchema(**model_data_for_db)


    async def get_model(self, model_id: str) -> Optional[ModelSchema]:
        """Retrieves model details from the database."""
        logger.info(f"Getting model with ID: {model_id}")
        if model_id in self._model_cache:
            logger.info(f"Cache hit for model ID: {model_id}")
            logger.debug(f"Cache: {self._model_cache}")
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
                if model_data.get("config"):
                    model_data["config"] = json.loads(model_data["config"])
                model_data["created_at"] = int(datetime.fromisoformat(model_data["created_at"]).timestamp()*1000)
                model_data["updated_at"] = int(datetime.fromisoformat(model_data["updated_at"]).timestamp()*1000)
                model_schema = ModelSchema(**model_data)
                self._model_cache[model_id] = model_schema
                logger.debug(f"Cache after get_model: {self._model_cache}")
                return model_schema
            except ValidationError as e:
                logger.error(f"Data from DB for model {model_id} failed ModelSchema validation: {e}")  
                return None
        return None

    async def list_models(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 10,
        sort_by: Optional[ModelSortByEnum] = None,
        sort_order: Optional[ModelSortOrderEnum] = None,
        search_term: Optional[str] = None,
        search_fields: Optional[List[str]] = None
    ) -> List[ModelSchema]:
        """Lists all models, optionally filtered."""
        order_by = "created_at DESC"
        if sort_by:
            sort_key = sort_by.value
            sort_order_key = sort_order.value if sort_order else "DESC"
            order_by = f"{sort_key} {sort_order_key}"

        model_data_list = self.db.find("models", filters=filters or {}, limit=page_size, offset=(page - 1) * page_size, order_by=order_by, search_term=search_term, search_fields=search_fields)
     
        models = []
        for data in model_data_list:
            try:
                if data.get("input_schema"):
                    data["input_schema"] = json.loads(data["input_schema"])
                if data.get("output_schema"):
                    data["output_schema"] = json.loads(data["output_schema"])
                if data.get("config"):
                    data["config"] = json.loads(data["config"])
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
        model_info = await self.get_model(model_id)
        if updates and updates.file_path:
            # 1. Validate file
            file_validation = self.validator.validate_model_file(
                file_path=updates.file_path,
                engine_type=updates.engine_type or model_info.engine_type
            )
            if not file_validation.valid:
                logger.error(f"Model file validation failed: {file_validation.errors}")  
                raise ValueError(f"Model file validation failed: {file_validation.errors}")
            # 2. Store file and delete old file
            if model_info.file_path:
                self.store.delete_model_and_yaml_file(model_info.file_path)
            relative_file_path = self.store.save_model_file(
                temp_file_path=updates.file_path,
                model_id=model_id,
                original_filename=updates.file_name
            )
            if not relative_file_path:
                logger.error(f"Failed to save model file for model ID {model_id}")  
                raise ValueError(f"Failed to save model file for model ID {model_id}.")
            # 3. Update DB record with new file path
            updates.file_path = relative_file_path
            updates.file_name = None # Don't need to store file in DB

        if updates and updates.params_yaml:
            # 1. Validate yaml file
            input_validation, output_validation, config_validation = self.validator.validate_yaml_schema(updates.params_yaml)
            if not input_validation.valid or not output_validation.valid:
                logger.error(f"YAML validation failed: {input_validation.errors} {output_validation.errors}")  
                raise ValueError(f"YAML validation failed: {input_validation.errors} {output_validation.errors}")
            # 2. Store yaml file and delete old file
            if model_info.params_path:
                self.store.delete_yaml_file(model_info.params_path)
            yaml_file_path = self.store.save_yaml_file(
                params_yaml=updates.params_yaml,
                model_id=model_id
            )
            if not yaml_file_path:
                logger.error(f"Failed to save yaml file for model ID {model_id}")  
                raise ValueError(f"Failed to save yaml file for model ID {model_id}.")
            updates.params_path = yaml_file_path
            updates.input_schema = input_validation.data
            updates.output_schema = output_validation.data
            updates.config = config_validation.data
            updates.params_yaml = None # Don't need to store yaml in DB
        update_data = updates.model_dump(exclude_unset=True,exclude_none=True) # Get all fields
        logger.info(f"Update data: {update_data}")
        if not update_data:
            logger.info("No update data provided.")  
            return await self.get_model(model_id) # Return current model if no updates

        update_data["updated_at"] = datetime.utcnow()
        
        rows_updated = self.db.update("models", {"id": model_id}, update_data)
        if rows_updated > 0:
            logger.info(f"Model {model_id} updated successfully.")  
            self._clear_cache_for_model(model_id)
            logger.info(f"Cleared cache for model {model_id}")
            return await self.get_model(model_id)
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


    async def publish_model(self, model_id: str, test_data_config: Optional[Dict[str, Any]] = None) -> DeploymentResult:
            """
            Publishes a model: loads into engine, tests inference, updates status.
            Corresponds to sequence in 4.2.3.
            Args:
                model_id (str): ID of the model to publish.
                test_data_config (Optional[Dict[str, Any]]): Configuration for test inference data.
            Returns:
                DeploymentResult containing success status and message.
            """
            logger.info(f"Attempting to publish model ID: {model_id}")  
            model_info = await self.get_model(model_id)
            logger.info(f"Model info: {model_info}")
            if not model_info:
                return DeploymentResult(success=False, message=f"Model {model_id} not found.", error_code=ErrorCode.MODEL_NOT_FOUND.to_dict() )
            
            if model_info.status == ModelStatusEnum.PUBLISHED:
                return DeploymentResult(success=True, model_id=model_id,  message="Model already published.")

            # 1. Load model into EngineService
            logger.info(f"Loading model {model_id} into engine service...")  
            load_success = self.engine_service.load_model(model_id, model_info)
            if not load_success:
                logger.error(f"Failed to load model {model_id} into engine.")  
                return DeploymentResult(success=False, model_id=model_id,  message="Engine failed to load model.", error_code=ErrorCode.MODEL_LOAD_ERROR.to_dict() )
            logger.info(f"Model {model_id} loaded by EngineService.")  

            # 2. Test Model Inference (using the loaded engine instance via EngineService)
            engine_instance = self.engine_service.get_engine_instance(model_id)
            if not engine_instance: # Should not happen if load_success was true
                return DeploymentResult(success=False, model_id=model_id,  message="Engine instance not found after load.", error_code=ErrorCode.ENGINE_INSTANCE_NOT_FOUND.to_dict() )

            logger.info(f"Performing test inference for model {model_id}...")  
            # Prepare test_inputs based on test_data_config and model_info.input_schema
            sample_test_inputs: List[InferenceInput] = []
            if model_info.input_schema and test_data_config:
                # For now, assume test_data_config IS the InferenceInput.data if simple.
                logger.warning(f"Warning: Actual test data generation from test_data_config is not fully implemented.")  
                if "sample_input_data" in test_data_config: # Example
                    sample_test_inputs.append(InferenceInput(data=test_data_config["sample_input_data"]))
                else: 
                    pass
                

            # if not sample_test_inputs: # If no sample data could be prepared
            #     logger.warning(f"Warning: No sample test inputs provided for model {model_id}. using generated sample inputs.")  
            #     # use model_info.input_schema to generate sample test inputs
            #     input_schema = ModelIODefinitionFile(inputs=model_info.input_schema,outputs=model_info.output_schema)
            #     sample_test_inputs = self._generate_sample_test_inputs(input_schema.inputs)
            #     logger.info(f"Generated sample test inputs: {sample_test_inputs}")
                


            test_result: InferenceResult = engine_instance.test_inference(test_inputs=sample_test_inputs or None)
            
            if not test_result.success:
                logger.error(f"Model {model_id} failed test inference: {test_result.error_message}")  
                self.engine_service.unload_model(model_id) # Unload on test failure
                return DeploymentResult(success=False, model_id=model_id,  message=f"Test inference failed: {test_result.error_message}", error_code=ErrorCode.TEST_INFERENCE_FAILED.to_dict() )
            logger.info(f"Model {model_id} passed test inference.")  

            # 3. Update model status to "published" in DB
            updated_model = await self.update_model(model_id, ModelUpdate(status=ModelStatusEnum.PUBLISHED))
            if updated_model and updated_model.status == ModelStatusEnum.PUBLISHED:
                logger.info(f"Model {model_id} status updated to 'published'.")  
                return DeploymentResult(success=True, model_id=model_id,  message="Model published successfully.")
            else:
                logger.error(f"Failed to update model {model_id} status to 'published' in DB.")  
                self.engine_service.unload_model(model_id) # Unload if DB update fails
                return DeploymentResult(success=False, model_id=model_id,  message="Failed to update model status in database.", error_code=ErrorCode.MODEL_PUBLISH_FAILED.to_dict() )


    async def unpublish_model(self, model_id: str) -> DeploymentResult:
        """
        Unpublishes a model: unloads from engine, updates status to "draft" or "deprecated".
        """
        logger.info(f"Attempting to unpublish model ID: {model_id}")  
        model_info = await self.get_model(model_id)
        if not model_info:
            return DeploymentResult(success=False, message=f"Model {model_id} not found.", error_code=ErrorCode.MODEL_NOT_FOUND.to_dict() )

        # 1. Unload model from EngineService
        self.engine_service.unload_model(model_id) # idempotent
        logger.info(f"Model {model_id} unloaded from engine service (if it was loaded).")  

        # 2. Update model status in DB (e.g., to "draft" or "unpublished")
        new_status = ModelStatusEnum.DRAFT
        updated_model = await self.update_model(model_id, ModelUpdate(status=new_status))
        if updated_model and updated_model.status == new_status:
            logger.info(f"Model {model_id} status updated to '{new_status}'.")  
            return DeploymentResult(success=True, model_id=model_id, message="Model unpublished successfully.")
        else:
            logger.error(f"Failed to update model {model_id} status to '{new_status}' in DB.")  
            return DeploymentResult(success=False, model_id=model_id, message="Failed to update model status in database.", error_code=ErrorCode.MODEL_UNPUBLISH_FAILED.to_dict() )
        
    async def load_published_models(self) -> None:
        """
        Loads all published models into EngineService.
        """
        published_models = await self.list_models(filters={"status": ModelStatusEnum.PUBLISHED})
        for model in published_models:
            self.engine_service.load_model(model.id, model)