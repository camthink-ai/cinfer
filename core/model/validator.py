# cinfer/models/validator.py
from typing import Dict, Any, Optional, List

from pydantic import ValidationError

from schemas.models import ModelMetadataBase, ValidationResult as SchemaValidationResult
from core.engine.factory import EngineRegistry, engine_registry as global_engine_registry
from core.engine.base import IEngine, InferenceInput, InferenceResult # For test_inference signature if needed here
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

  