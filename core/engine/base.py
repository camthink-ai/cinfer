# cinfer/engine/base.py
import time # For time-based analysis
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from asyncio import Future
from concurrent.futures  import ThreadPoolExecutor
from queue import Queue
from pydantic import BaseModel, Field
from pathlib import Path
from core.processors.base import BaseProcessor
from core.processors.factory import processor_registry 
from schemas.engine import InferenceInput, InferenceOutput, InferenceResult, EngineInfo, ResourceRequirements

logger = logging.getLogger(f"cinfer.{__name__}")

class ResourceTracker: # Placeholder as per document context
    """
    A placeholder for tracking resource usage.
    Actual implementation would monitor CPU, GPU, memory.
    """
    def __init__(self):
        self.used_memory_gb: float = 0.0
        self.used_cpu_cores: float = 0.0

    def update_usage(self, cpu: float, memory: float):
        self.used_cpu_cores = cpu
        self.used_memory_gb = memory

    def get_current_usage(self) -> Dict[str, float]:
        return {"cpu_cores": self.used_cpu_cores, "memory_gb": self.used_memory_gb}


# --- Engine Interfaces and Abstract Classes ---
class IEngine(ABC):
    """
    Interface for AI inference engines.
    Defines a unified standard for interacting with various underlying engine implementations.
    """

    @abstractmethod
    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        """
        Initializes the inference engine with global configurations.
        Args:
            engine_config (Dict[str, Any]): Configuration specific to the engine instance
                                         (e.g., device selection, global settings).
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Loads an AI model into the engine.
        Args:
            model_path (str): Path to the model file.
            model_config (Optional[Dict[str, Any]], optional): Configuration specific to this model
                                                            (e.g., input/output names, optimization flags).
                                                            Defaults to None.
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
        """
        Performs inference on the provided input data using the loaded model.
        Args:
            inputs (List[InferenceInput]): A list of InferenceInput objects.
                                        For batching, this list might contain multiple items.
                                        For single inference, it would be a list with one item.
        Returns:
            InferenceResult: The result of the inference.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self) -> bool:
        """
        Releases resources used by the engine and the loaded model.
        Returns:
            bool: True if resources were released successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> EngineInfo:
        """
        Retrieves information about the current state and configuration of the engine.
        Returns:
            EngineInfo: An object containing details about the engine.
        """
        raise NotImplementedError
        
    @abstractmethod
    def validate_model_file(self, file_path: str) -> bool:
        """Validates the given model file path."""
        pass


class BaseEngine(IEngine):
    """
    Abstract base class providing common functionalities for inference engines.
    Implements IEngine and adds shared utility methods.
    """
    def __init__(self):
        self._model: Optional[Any] = None
        self._engine_config: Dict[str, Any] = {}
        self._model_config: Dict[str, Any] = {}
        self._initialized: bool = False
        self._model_loaded: bool = False
        self._loaded_model_path: Optional[str] = None
        self._resources: ResourceTracker = ResourceTracker()
        self._current_device: Optional[str] = None
        self._processor: Optional[BaseProcessor] = None

    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        self._engine_config = engine_config
        self._initialized = True
        self._current_device = self._engine_config.get("device", "CPU")
        logger.info(f"{self.__class__.__name__} initialized with config: {engine_config}. Device: {self._current_device}")
        return True

    @abstractmethod
    def _load_model_specifico(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """Engine-specific model loading logic."""
        raise NotImplementedError

    def load_model(self, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        if not self._initialized:
            logger.error(f"Error: Engine {self.__class__.__name__} not initialized. Call initialize() first.")
            return False
        
        self._model_config = model_config or {}
        logger.info(f"Loading model from {model_path} with config: {self._model_config}")

        storage_path = Path( self._engine_config.get("models.storage_path", "data/models"))
        model_path = str(storage_path / model_path)

        if not self.validate_model_file(model_path):
            logger.error(f"Model file validation failed for {model_path}")
            return False

        load_success = self._load_model_specifico(model_path, self._model_config)
        if load_success:
            self._model_loaded = True
            self._loaded_model_path = model_path
            logger.info(f"Model {model_path} loaded successfully into {self.__class__.__name__}.")
            # Get the processing strategy from the model config
            strategy_name = model_config.get("processing_strategy")
            if strategy_name:
                processor_class = processor_registry.get_processor_class(strategy_name)
                if not processor_class:
                    logger.error(f"No processor registered for strategy '{strategy_name}'.")
                    return False
                self._processor = processor_class(model_config, self.get_info()) 
                logger.info(f"Processor '{strategy_name}' loaded for the model.")
            else:
                logger.warning("'processing_strategy' not found in model config.")
        else:
            self._model_loaded = False
            self._loaded_model_path = None
            logger.error(f"Failed to load model {model_path} into {self.__class__.__name__}.")
        
        return load_success


    def is_initialized(self) -> bool:
        """Checks if the engine has been initialized."""
        return self._initialized

    def is_model_loaded(self) -> bool:
        """Checks if a model is currently loaded."""
        return self._model_loaded

    def release(self) -> bool:
        self._model = None
        self._model_loaded = False
        self._initialized = False
        self._loaded_model_path = None
        logger.info(f"{self.__class__.__name__} released model and resources.")
        return True

    @abstractmethod
    def _preprocess_input(self, raw_inputs: List[InferenceInput]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _postprocess_output(self, raw_outputs: Any) -> List[InferenceOutput]:
        raise NotImplementedError

    def validate_model_file(self, model_path: str) -> bool:
        import os
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        if not os.path.isfile(model_path):
            logger.error(f"Path is not a file: {model_path}")
            return False
        return True

    def test_inference(self, test_inputs: List[InferenceInput]) -> InferenceResult:
        if not self._model_loaded or not self._model:
            return InferenceResult(success=False, error_message="No model loaded for test inference.")
        
        logger.info(f"Performing test inference on {self.__class__.__name__}...")
        start_time_sec = time.time() # Corrected
        try:
            result = self.predict(test_inputs)
            end_time_sec = time.time() # Corrected
            
            if result.success:
                result.processing_time_ms = (end_time_sec - start_time_sec) * 1000 # Corrected
                logger.info(f"Test inference successful. Time: {result.processing_time_ms:.2f} ms")
            else:
                logger.error(f"Test inference failed: {result.error_message}")
                # If predict sets its own time on failure, use it, else calculate
                if result.processing_time_ms is None:
                     result.processing_time_ms = (end_time_sec - start_time_sec) * 1000 # Corrected
            return result
        except Exception as e:
            end_time_sec = time.time() # Corrected
            logger.error(f"Exception during test inference: {e}")
            return InferenceResult(
                success=False,
                error_message=str(e),
                processing_time_ms=(end_time_sec - start_time_sec) * 1000 # Corrected
            )

    def get_resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            cpu_cores=1,
            memory_gb=1,
            custom={"info": "Override this method for accurate resource requirements."}
        )


class AsyncEngine(BaseEngine):
    def __init__(self, max_workers: Optional[int] = None, queue_size: Optional[int] = None):
        super().__init__()
        if max_workers is None:
            import os
            max_workers = os.cpu_count() or 1
        
        self._task_queue: Queue = Queue(maxsize=queue_size or 0)
        self._worker_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{self.__class__.__name__}_worker")
        self._batch_size: int = 1
        self._max_workers = max_workers

    @abstractmethod
    def _batch_process(self, inputs_batch: List[Any]) -> List[Any]:
        raise NotImplementedError

    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
        if not self._model_loaded:
            return InferenceResult(success=False, error_message="Model not loaded.")
        
        start_time_sec = time.time() # Corrected
        try:
            if self._processor:
                preprocessed_batch = self._processor.preprocess(inputs)
            else:
                preprocessed_batch = self._preprocess_input(inputs)
            raw_outputs_batch = self._batch_process([preprocessed_batch] if not isinstance(preprocessed_batch, list) else preprocessed_batch)
            
            if isinstance(raw_outputs_batch, list) and len(raw_outputs_batch) == 1 and len(inputs) ==1 :
                if self._processor:
                    final_outputs = self._processor.postprocess(raw_outputs_batch[0])
                else:
                    final_outputs = self._postprocess_output(raw_outputs_batch[0])
            else:
                if self._processor:
                    final_outputs = self._processor.postprocess(raw_outputs_batch)
                else:
                    final_outputs = self._postprocess_output(raw_outputs_batch)

            processing_time_ms = (time.time() - start_time_sec) * 1000 # Corrected
            return InferenceResult(success=True, outputs=final_outputs, processing_time_ms=processing_time_ms)
        except Exception as e:
            logger.error(f"Error during prediction in AsyncEngine: {e}")
            return InferenceResult(success=False, error_message=str(e), processing_time_ms=(time.time() - start_time_sec) * 1000) # Corrected

    def async_predict(self, inputs: List[InferenceInput]) -> Future[InferenceResult]:
        if not self._model_loaded:
            from concurrent.futures import Future as ConcreteFuture
            future_result: ConcreteFuture[InferenceResult] = ConcreteFuture()
            future_result.set_result(InferenceResult(success=False, error_message="Model not loaded."))
            return future_result
            
        return self._worker_pool.submit(self.predict, inputs)

    def set_batch_size(self, size: int) -> None:
        if size > 0:
            self._batch_size = size
            logger.info(f"Batch size set to {size} for {self.__class__.__name__}")
        else:
            logger.error(f"Invalid batch size: {size}. Must be positive.")
            pass

    def get_queue_size(self) -> int:
        return self._task_queue.qsize()


    def release(self) -> bool:
        super_released = super().release()
        logger.info(f"Shutting down worker pool for {self.__class__.__name__}...")
        try:
            self._worker_pool.shutdown(wait=True)
            logger.info(f"Worker pool for {self.__class__.__name__} shut down.")
            return super_released
        except Exception as e:
            logger.error(f"Error shutting down worker pool for {self.__class__.__name__}: {e}")
            return False