# cinfer/models/validator.py
from typing import Dict, Any, Optional, List, Tuple
import yaml
import json
from pydantic import ValidationError

from schemas.models import ModelMetadataBase, ValidationResult as SchemaValidationResult
from core.engine.factory import EngineRegistry, engine_registry as global_engine_registry
from core.engine.base import IEngine
from schemas.engine import InferenceInput, InferenceResult # For test_inference signature if needed here
from core.config import ConfigManager # Might be needed for engine configs
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

# Note: The Pydantic schema ValidationResult is in schemas.models.
# We will use that directly. If a different structure is needed specifically
# for validator outputs, we could define a new one. For now, SchemaValidationResult is good.

class ModelValidator:
    """
    Validates model files, metadata, and configurations.
    As per document section 4.2.1, 4.2.2.
    """
    def __init__(self, engine_registry_instance: EngineRegistry, config_manager_instance: Optional[ConfigManager] = None):
        self._engine_registry: EngineRegistry = engine_registry_instance
        self._config_manager: Optional[ConfigManager] = config_manager_instance

    def validate_yaml_schema(self, yaml_content: str) -> Tuple[SchemaValidationResult, SchemaValidationResult]:
        """Validate a YAML schema."""
        try:
            # Parse YAML content
            yaml_data = yaml.safe_load(yaml_content)
            if not isinstance(yaml_data, dict):
                return (
                    SchemaValidationResult(valid=False, errors=["YAML content must be a dictionary"], message="Invalid YAML format"),
                    SchemaValidationResult(valid=False, errors=["YAML content must be a dictionary"], message="Invalid YAML format")
                )

            # Extract input and output schemas
            input_schema = yaml_data.get("inputs", {})
            output_schema = yaml_data.get("outputs", {})
            logger.info(f"Input schema: type: {type(input_schema)}")
            logger.info(f"Output schema: type: {type(output_schema)}")


            # Validate input schema
            input_validation = SchemaValidationResult(
                valid=bool(input_schema),
                errors=None if input_schema else ["Input schema is missing or empty"],
                message="Input schema validation successful" if input_schema else "Input schema validation failed",
                data=input_schema
            )

            # Validate output schema
            output_validation = SchemaValidationResult(
                valid=bool(output_schema),
                errors=None if output_schema else ["Output schema is missing or empty"],
                message="Output schema validation successful" if output_schema else "Output schema validation failed",
                data=output_schema
            )

            logger.info(f"Input schema validation: {input_validation.valid}")
            logger.info(f"Output schema validation: {output_validation.valid}")

            return input_validation, output_validation

        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML format: {str(e)}"
            logger.error(error_msg)
            return (
                SchemaValidationResult(valid=False, errors=[error_msg], message="YAML parsing failed"),
                SchemaValidationResult(valid=False, errors=[error_msg], message="YAML parsing failed")
            )
        except Exception as e:
            error_msg = f"Error validating YAML schema: {str(e)}"
            logger.error(error_msg)
            return (
                SchemaValidationResult(valid=False, errors=[error_msg], message="YAML validation failed"),
                SchemaValidationResult(valid=False, errors=[error_msg], message="YAML validation failed")
            )

    def validate_model_file(self,
                            file_path: str,
                            engine_type: str,
                            engine_specific_config: Optional[Dict[str, Any]] = None
                           ) -> SchemaValidationResult:
        """
        Validates the model file using the appropriate engine's validation capabilities.
        This typically involves static checks on the file format and structure.
        Args:
            file_path (str): Absolute path to the model file.
            engine_type (str): The declared engine type for the model (e.g., "onnx").
            engine_specific_config (Optional[Dict[str, Any]]): Configuration for the engine instance.
                                                               If None, default engine config is used.
        Returns:
            SchemaValidationResult: Result of the file validation.
        """
        engine_class = self._engine_registry.get_engine_class(engine_type)
        if not engine_class:
            return SchemaValidationResult(valid=False, errors=[f"No engine registered for type: {engine_type}"], message="Engine type not supported.")

        # Prepare engine configuration
        # If a global config manager is available, try to get default config for this engine type
        effective_engine_config = {}
        if self._config_manager:
            effective_engine_config = self._config_manager.get_config(f"engines.{engine_type}", {})
        
        if engine_specific_config: # Merge/override with provided specific config
            effective_engine_config.update(engine_specific_config)

        temp_engine: Optional[IEngine] = None
        try:
            # Create a temporary engine instance for validation
            # The factory's create_engine now handles initialization
            temp_engine = self._engine_registry.create_engine(
                name=engine_type,
                engine_config=effective_engine_config
            )
            if not temp_engine:
                 return SchemaValidationResult(valid=False, errors=[f"Failed to create temporary engine for type: {engine_type}"], message="Engine creation failed.")

            # BaseEngine provides validate_model_file. Specific engines can override it.
            if hasattr(temp_engine, 'validate_model_file') and callable(getattr(temp_engine, 'validate_model_file')):
                is_valid_file = temp_engine.validate_model_file(file_path) # This method is on BaseEngine
                if is_valid_file:
                    # Some engines might provide more details upon successful static validation
                    return SchemaValidationResult(valid=True, message=f"Model file '{file_path}' appears valid for engine '{engine_type}'.")
                else:
                    return SchemaValidationResult(valid=False, errors=[f"Model file '{file_path}' failed validation for engine '{engine_type}'."], message="Model file validation failed.")
            else:
                # Fallback if engine doesn't have a specific validate_model_file (should not happen if it inherits BaseEngine)
                import os
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    return SchemaValidationResult(valid=True, message=f"Basic file existence check passed for '{file_path}'. Engine '{engine_type}' has no specific file validation method.")
                else:
                    return SchemaValidationResult(valid=False, errors=[f"Model file '{file_path}' not found or is not a file."], message="Basic file check failed.")

        except Exception as e:
            logger.error(f"Exception during model file validation for {engine_type} with file {file_path}: {e}")
            return SchemaValidationResult(valid=False, errors=[str(e)], message="Exception during model file validation.")
        finally:
            if temp_engine:
                temp_engine.release() # Ensure temporary engine is released


    def validate_model_config(self, model_specific_config: Dict[str, Any], engine_type: str) -> SchemaValidationResult:
        """
        Validates the model-specific configuration.
        This can be engine-dependent. For now, it's a placeholder.
        Args:
            model_specific_config (Dict[str, Any]): The configuration dictionary for the model.
            engine_type (str): The engine type, as validation rules might differ.
        Returns:
            SchemaValidationResult: Result of the config validation.
        """
        # Placeholder: Actual validation logic would depend on the engine_type
        # and expected structure of model_specific_config.
        # For example, an ONNX model might not have much specific config here,
        # but a custom PyTorch model might expect certain keys.
        if not isinstance(model_specific_config, dict):
            return SchemaValidationResult(valid=False, errors=["Model config must be a dictionary."], message="Invalid model config format.")
        
        logger.info(f"Validating model config for engine {engine_type}: {model_specific_config}")
        # Add engine-specific checks here if needed
        # e.g., if engine_type == "my_custom_engine":
        #     if "required_param" not in model_specific_config:
        #         return SchemaValidationResult(valid=False, errors=["Missing 'required_param' in model config."])
        
        return SchemaValidationResult(valid=True, message="Model configuration appears valid (basic check).")


    # Test inference is usually done *after* a model is fully loaded and managed by EngineService.
    # ModelManager would orchestrate this test.
    # If ModelValidator needs to be involved, it would need EngineService and model_id.
    # Based on sequence diagram 4.2.3, ModelMgr calls Engine directly for test_inference.
    # So, we might not need a test_model_inference method directly in ModelValidator,
    # or it could be a helper that EngineService/ModelManager calls.
    # For now, let's assume ModelManager handles the test inference orchestration.