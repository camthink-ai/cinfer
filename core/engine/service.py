# cinfer/engine/service.py
from typing import Dict, Any, Optional, List, Type
from threading import RLock # For thread-safe access to shared _engine_instances
import logging
from pydantic import BaseModel
from schemas.engine import ModelIODefinitionFile
from schemas.dynamic_factory import create_dynamic_model_from_definition
logger = logging.getLogger(f"cinfer.{__name__}")

from .base import IEngine, InferenceResult, InferenceInput # Corrected InferenceInput import
from .factory import EngineRegistry, engine_registry as global_engine_registry
from core.config import ConfigManager, get_config_manager
from schemas.models import Model as ModelSchema # To get model info like engine_type

class EngineService:
    """
    Manages active inference engine instances for loaded models.
    As per document section 4.6.3.
    """
    def __init__(self, config_manager: ConfigManager, engine_reg: EngineRegistry):
        self._config_manager: ConfigManager = config_manager
        self._engine_registry: EngineRegistry = engine_reg
        # Stores active engine instances, mapping model_id to its IEngine instance
        self._engine_instances: Dict[str, IEngine] = {}
        self._lock = RLock() # To protect concurrent access to _engine_instances
        self._dynamic_validator_cache: Dict[str, Type[BaseModel]] = {}
        #{ model_id: {"input": DynamicInputModelClass, "output": DynamicOutputModelClass} }

    def get_engine_instance(self, model_id: str) -> Optional[IEngine]:
        """
        Retrieves an active engine instance for a given model_id.
        Args:
            model_id (str): The ID of the model.
        Returns:
            Optional[IEngine]: The active engine instance if the model is loaded, else None.
        """
        with self._lock:
            return self._engine_instances.get(model_id)
        
    def _get_io_definition_from_model(self, model_info: ModelSchema) -> ModelIODefinitionFile:
        """
        Retrieves the IO definition from the model.
        """
        logger.info(f"Getting IO definition from model {model_info.id}.")
        logger.info(f"Model info: {model_info.input_schema}")
        logger.info(f"Model info: {model_info.output_schema}")
        return ModelIODefinitionFile(
            config=model_info.config,
            inputs=model_info.input_schema,
            outputs=model_info.output_schema,
        )
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unloads a model by releasing its engine instance.
        Args:
            model_id (str): The ID of the model to unload.
        Returns:
            bool: True if the model was successfully unloaded or was not loaded.
        """
        with self._lock:
            engine_instance = self._engine_instances.pop(model_id, None)
            if engine_instance:
                engine_instance.release()
                logger.info(f"Model {model_id} unloaded and engine instance released.")
                return True
            logger.info(f"Model {model_id} was not loaded, no action taken.")
            return True # Considered success if not loaded

    def load_model(self, model_id: str, model_info: ModelSchema) -> bool:
        """
        Loads a model into an appropriate engine instance and manages it.
        Args:
            model_id (str): The ID of the model to load.
            model_info (ModelSchema): Pydantic schema containing model details,
                                     including 'file_path', 'engine_type', and 'config'.
        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """
        with self._lock:
            if model_id in self._engine_instances:
                logger.info(f"Model {model_id} is already loaded.")
                return True # Or perhaps re-validate if it's the same version

            engine_type = model_info.engine_type
            # Get engine-specific global configuration from the main config
            # e.g., config.get_config(f"engines.{engine_type}", {})
            # The create_engine in factory.py now takes a general engine_config.
            # The engine's initialize method will use its relevant parts.
            # A more structured approach might be to have a dedicated config section per engine type.
            # For now, let's assume the main engine config has sections like engines.onnx, engines.tensorrt
            
            engine_creation_config = self._config_manager.get_config(f"engines.{engine_type}")
            if engine_creation_config is None:
                logger.warning(f"Warning: No specific configuration found for engine type '{engine_type}'. Using defaults.")
                engine_creation_config = {} # Rely on engine's internal defaults or registry default
            
            # Add general engine configs if not specified type-specific, e.g., max_workers
            engine_creation_config.setdefault("max_workers", self._config_manager.get_config("request.workers_per_model", 2))


            engine_instance = self._engine_registry.create_engine(
                name=engine_type,
                engine_config=engine_creation_config
            )

            if not engine_instance:
                logger.error(f"Failed to create engine instance for type {engine_type} for model {model_id}.")
                return False

            # Model-specific loading configuration (from model_info.config which comes from ModelMetadataBase.config)
            model_load_config = model_info.config or {}

            if engine_instance.load_model(model_info.file_path, model_load_config):
                self._engine_instances[model_id] = engine_instance
                logger.info(f"Model {model_id} (type: {engine_type}) loaded successfully into engine.")
            else:
                logger.error(f"Failed to load model file {model_info.file_path} into engine {engine_type} for model {model_id}.")
                # Clean up the created engine instance if model loading failed
                engine_instance.release()
                return False
            
            # Create dynamic input and output models
            logger.info(f"Creating dynamic input and output models for model {model_id}.")
            try:
                # Get the IO definition from the model
                io_definition: ModelIODefinitionFile = self._get_io_definition_from_model(model_info)

                logger.info(f"IO definition: {io_definition}")
                
                # Create dynamic input and output models
                DynamicInputModel = create_dynamic_model_from_definition(
                    f"InputFor_{model_id.replace('-', '_')}",
                    io_definition.inputs
                )
                DynamicOutputModel = create_dynamic_model_from_definition(
                    f"OutputFor_{model_id.replace('-', '_')}",
                    io_definition.outputs
                )

                # Cache the models
                self._dynamic_validator_cache[model_id] = {
                    "input": DynamicInputModel,
                    "output": DynamicOutputModel
                }
                
                logger.info(f"Dynamic input and output models created for model {model_id}.")
                return True

            except Exception as e:
                logger.error(f"Failed to create dynamic input model for model {model_id}: {e}")
                # Clean up the created engine instance if model loading failed
                self.unload_model(model_id)
                return False

    def get_input_validator(self, model_id: str) -> Type[BaseModel]:
        """
        Retrieves the input validator for a given model_id.
        """
        return self._dynamic_validator_cache[model_id]["input"]
    
    def get_output_validator(self, model_id: str) -> Type[BaseModel]:
        """
        Retrieves the output validator for a given model_id.
        """
        return self._dynamic_validator_cache[model_id]["output"]

    def predict(self, model_id: str, inputs: List[InferenceInput]) -> InferenceResult:
        """
        Performs inference using the loaded model.
        Args:
            model_id (str): The ID of the model to use for inference.
            inputs (List[InferenceInput]): The input data for inference.
        Returns:
            InferenceResult: The result of the inference.
        """
        engine_instance = self.get_engine_instance(model_id) # Thread-safe get
        if not engine_instance:
            return InferenceResult(success=False, error_message=f"Model {model_id} not loaded or engine not available.")
        
        return engine_instance.predict(inputs)

    def get_loaded_model_info(self, model_id: str) -> Optional[Any]: # Should be EngineInfo
        """Gets info about a specific loaded model's engine instance."""
        engine_instance = self.get_engine_instance(model_id)
        if engine_instance:
            return engine_instance.get_info()
        return None

    def get_all_loaded_models_info(self) -> Dict[str, Any]: # Dict[str, EngineInfo]
        """Gets info for all currently loaded models and their engine instances."""
        with self._lock:
            return {mid: eng.get_info() for mid, eng in self._engine_instances.items()}

    def release_all_engines(self):
        """Releases all currently managed engine instances."""
        with self._lock:
            for model_id in list(self._engine_instances.keys()): # Iterate over a copy of keys
                self.unload_model(model_id)
            logger.info("All engine instances have been released.")

# To make EngineService easily accessible, typically it would be instantiated once.
# For now, we define it. Its instantiation will be part of the main application setup.