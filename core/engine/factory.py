# cinfer/engine/factory.py
from typing import Dict, Type, Optional, List, Any

from .base import IEngine
# Import specific engine implementations that should be discoverable
from .onnx import ONNXEngine
from .dummy import DummyEngine
# from .tensorrt import TensorRTEngine # Example for future
# from .pytorch import PyTorchEngine   # Example for future
import logging

logger = logging.getLogger(f"cinfer.{__name__}")


class EngineRegistry: #
    """
    Manages the registration and creation of inference engine instances.
    Implements a factory pattern for engines.
    """
    def __init__(self):
        self._engines: Dict[str, Type[IEngine]] = {} # Stores name -> engine class
        self._default_engine_name: Optional[str] = None #
        self._instances: Dict[str, IEngine] = {} # Optional: for caching engine instances if needed

        # Auto-register known engines
        self.register_engine("onnx", ONNXEngine) # ONNXEngine.ENGINE_NAME could be used if defined
        self.register_engine("dummy", DummyEngine)
        # self.register_engine("tensorrt", TensorRTEngine)
        # self.register_engine("pytorch", PyTorchEngine)

    def register_engine(self, name: str, engine_class: Type[IEngine]) -> None: #
        """
        Registers an engine class with a given name.
        Args:
            name (str): The unique name for the engine type (e.g., "onnx", "tensorrt").
            engine_class (Type[IEngine]): The class of the engine to register.
        """
        if not issubclass(engine_class, IEngine):
            raise TypeError(f"Engine class {engine_class.__name__} must inherit from IEngine.")
        if name in self._engines:
            logger.warning(f"Warning: Engine with name '{name}' already registered. Overwriting.") 
        self._engines[name] = engine_class
        logger.info(f"Engine '{name}' (class: {engine_class.__name__}) registered.") 

    def unregister_engine(self, name: str) -> None: #
        """
        Unregisters an engine class.
        Args:
            name (str): The name of the engine to unregister.
        """
        if name in self._engines:
            del self._engines[name]
            logger.info(f"Engine '{name}' unregistered.") 
            if name == self._default_engine_name:
                self._default_engine_name = None
        else:
            logger.warning(f"Warning: Engine '{name}' not found for unregistration.") 

    def get_engine_class(self, name: str) -> Optional[Type[IEngine]]:
        """
        Retrieves a registered engine class by name.
        Args:
            name (str): The name of the engine.
        Returns:
            Optional[Type[IEngine]]: The engine class, or None if not found.
        """
        return self._engines.get(name)

    def get_all_engines(self) -> List[str]: #
        """
        Returns a list of names of all registered engines.
        Returns:
            List[str]: List of engine names.
        """
        return list(self._engines.keys())

    def set_default_engine(self, name: str):
        """Sets the default engine name."""
        if name in self._engines:
            self._default_engine_name = name
        else:
            raise ValueError(f"Engine '{name}' not registered. Cannot set as default.")

    def create_engine(self, name: Optional[str] = None, engine_config: Optional[Dict[str, Any]] = None) -> Optional[IEngine]: #
        """
        Creates an instance of the specified (or default) inference engine.
        Args:
            name (Optional[str], optional): The name of the engine to create.
                                           If None, uses the default engine. Defaults to None.
            engine_config (Optional[Dict[str, Any]], optional): Configuration for the engine instance.
                                                              Defaults to None (empty config).
        Returns:
            Optional[IEngine]: An instance of the engine, or None if creation fails.
        """
        engine_name_to_create = name or self._default_engine_name
        if not engine_name_to_create:
            raise ValueError("No engine name specified and no default engine set.")

        engine_class = self.get_engine_class(engine_name_to_create)
        if not engine_class:
            logger.error(f"Error: Engine class for '{engine_name_to_create}' not found.") 
            return None

        try:
            # For AsyncEngine and its children, constructor might take max_workers, queue_size
            # These could come from engine_config or have defaults.
            # The IEngine interface itself doesn't define constructor params.
            # We assume engine_config is for IEngine.initialize() primarily.
            # Specific engine constructors might pull from their own part of engine_config if needed.
            # For ONNXEngine, constructor takes max_workers, queue_size.
            # Let's assume these are top-level in engine_config for now.
            
            # This is a bit tricky because constructors for different engines might vary.
            # The IEngine interface doesn't prescribe a constructor signature.
            # One way is to pass parts of engine_config to constructor if known.
            # For ONNXEngine -> AsyncEngine:
            max_workers = (engine_config or {}).get("max_workers")
            queue_size = (engine_config or {}).get("queue_size")

            if issubclass(engine_class, ONNXEngine): # Specific handling for ONNX constructor
                 engine_instance = engine_class(max_workers=max_workers, queue_size=queue_size)
            else: # Generic instantiation
                 engine_instance = engine_class()
            
            # Initialize the engine with its specific configuration part
            if not engine_instance.initialize(engine_config or {}):
                logger.error(f"Failed to initialize engine instance of '{engine_name_to_create}'.") 
                return None
            
            logger.info(f"Engine instance '{engine_name_to_create}' created and initialized.") 
            return engine_instance
        except Exception as e:
            logger.error(f"Error creating engine '{engine_name_to_create}': {e}") 
            import traceback
            traceback.print_exc()
            return None

    def auto_select_engine(self, model_path: str) -> Optional[IEngine]: #
        """
        Automatically selects and creates an engine instance based on the model file type.
        (Simplified implementation based on file extension).
        Args:
            model_path (str): Path to the model file.
        Returns:
            Optional[IEngine]: An instance of a suitable engine, or None.
        """
        # This is a very basic auto-selection. Real-world might involve inspecting model metadata.
        if model_path.lower().endswith(".onnx"):
            logger.info(f"Auto-selecting ONNX engine for model: {model_path}") 
            # TODO: Get default ONNX engine config from global config manager
            default_onnx_config = {"execution_providers": ["CPUExecutionProvider"]}
            return self.create_engine("onnx", default_onnx_config)
        # Add more rules for .pt, .engine, .tflite etc.
        # elif model_path.lower().endswith(".engine"): # TensorRT
        #     return self.create_engine("tensorrt", {})
        logger.info(f"Could not auto-select an engine for model: {model_path}") 
        return None

# Global instance of the registry
engine_registry = EngineRegistry()
engine_registry.set_default_engine("onnx") # Set ONNX as default if available