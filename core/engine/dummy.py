# cinfer/engine/dummy.py

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging 

logger = logging.getLogger(f"cinfer.{__name__}")
try:
    from .base import AsyncEngine, EngineInfo, InferenceInput, InferenceOutput, InferenceResult, ResourceRequirements
except ImportError:
    # If you run this file directly for testing, and base.py is in the same directory, you can do this fallback
    from base import AsyncEngine, EngineInfo, InferenceInput, InferenceOutput, InferenceResult, ResourceRequirements

class DummyEngine(AsyncEngine):
    ENGINE_NAME = "DummyEngine"

    def __init__(self, max_workers: Optional[int] = None, queue_size: Optional[int] = None):
        super().__init__(max_workers=max_workers, queue_size=queue_size)
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: List[Tuple[Optional[int], ...]] = []
        self._input_types: List[str] = []
        self._current_device: Optional[str] = None
        self._model_config: Dict[str, Any] = {} # Used to store virtual model parameters passed from model_config

    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        # super().initialize() will set self._engine_config and self._initialized = True
        super_init_ok = super().initialize(engine_config)
        if not super_init_ok:
            logger.error("DummyEngine: Base class initialization failed.")
            return False
        
        self._current_device = self._engine_config.get("dummy_device", "CPU_dummy") # Get virtual device from engine config
        logger.info(f"DummyEngine initialized. Engine_config: {self._engine_config}. Device set to: {self._current_device}")
        return True

    # _load_model_specifico is called by the base class load_model method
    def _load_model_specifico(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        logger.info(f"DummyEngine: Simulating loading model from '{model_path}' with model_config: {model_config}")
        self._model_config = model_config # Store model config for later use

        # Get virtual model metadata from model_config, or use default values
        self._input_names = self._model_config.get("dummy_input_names", ["input_dummy_tensor"])
        self._output_names = self._model_config.get("dummy_output_names", ["output_dummy_tensor"])
        
        # Default virtual input shape, e.g. (batch_size, channels, height, width)
        # In the configuration, None can be represented as -1 or null for dynamic dimensions
        default_shape_cfg = self._model_config.get("dummy_default_input_shape", [-1, 3, 24, 24]) 
        
        raw_shapes_cfg = self._model_config.get("dummy_input_shapes", [default_shape_cfg] * len(self._input_names))
        # Ensure dummy_input_shapes is a list of lists/tuples
        if not isinstance(raw_shapes_cfg, list) or not all(isinstance(s, (list, tuple)) for s in raw_shapes_cfg):
            logger.warning(f"dummy_input_shapes in model_config is not a list of lists/tuples. Using default shape for all {len(self._input_names)} inputs.")
            raw_shapes_cfg = [default_shape_cfg] * len(self._input_names)
        
        self._input_shapes = []
        for shape_item in raw_shapes_cfg:
            try:
                processed_shape = tuple(None if s is None or s == -1 else int(s) for s in shape_item)
                self._input_shapes.append(processed_shape)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid shape item '{shape_item}' in dummy_input_shapes (Error: {e}). Using default shape '{default_shape_cfg}'.")
                self._input_shapes.append(tuple(None if s is None or s == -1 else int(s) for s in default_shape_cfg))

        self._input_types = self._model_config.get("dummy_input_types", ["tensor(float)"] * len(self._input_names))
        
        # self._loaded_model_path is set by the base class load_model
        # self._model_loaded is also set by the base class load_model after this method returns True
        
        logger.info(f"DummyEngine: Model '{model_path}' considered loaded (simulated).")
        logger.info(f"  Input Names: {self._input_names}")
        logger.info(f"  Input Shapes: {self._input_shapes}")
        logger.info(f"  Input Types: {self._input_types}")
        logger.info(f"  Output Names: {self._output_names}")
        return True

    def _preprocess_input(self, raw_inputs: List[InferenceInput]) -> Dict[str, np.ndarray]:
        logger.debug(f"DummyEngine: Preprocessing {len(raw_inputs)} inputs.")
        if not raw_inputs:
            logger.error("Input list cannot be empty for dummy preprocessing.")
            raise ValueError("Input list cannot be empty for dummy preprocessing.")
        
        if not self._input_names:
            logger.warning("DummyEngine: No input names defined. Cannot preprocess meaningfully.")
            return {}

        # Simplified processing: If there is only one original input and its data is a dictionary, it is assumed that it is provided by name
        if len(raw_inputs) == 1 and isinstance(raw_inputs[0].data, dict):
            input_dict = raw_inputs[0].data
            processed_dict = {}
            all_names_present = True
            for idx, name in enumerate(self._input_names):
                if name in input_dict:
                    # Try to convert to float32 numpy array
                    dtype_str = self._input_types[idx] if idx < len(self._input_types) else 'tensor(float)'
                    # Simulate a simple version of _ort_type_to_numpy_type in ONNXEngine
                    np_dtype = np.float32 # Default
                    if 'int' in dtype_str: np_dtype = np.int64
                    elif 'bool' in dtype_str: np_dtype = np.bool_
                    
                    processed_dict[name] = np.array(input_dict[name], dtype=np_dtype)
                else:
                    all_names_present = False
                    logger.warning(f"Input name '{name}' not found in provided input dict.")
                    break # If any expected input name is missing, fall back to simple processing
            if all_names_present:
                logger.debug("DummyEngine: Preprocessed from a single dict input.")
                return processed_dict

        # Default simple batch processing: Stack all input data under the first input name
        target_input_name = self._input_names[0]
        try:
            batch_data = [inp.data for inp in raw_inputs]
            # Use object type to accommodate different shapes/types of data, or select a common type like float32
            np_batch = np.array(batch_data, dtype=np.float32) 
            logger.debug(f"DummyEngine: Preprocessed to single input '{target_input_name}' with shape {np_batch.shape}.")
            return {target_input_name: np_batch}
        except Exception as e:
            logger.error(f"DummyEngine: Error during simple batch preprocessing: {e}", exc_info=True)
            # Return an empty dict or handle the error as needed
            return {target_input_name: np.array([inp.data for inp in raw_inputs])} # Try the most basic conversion

    def _postprocess_output(self, raw_outputs: List[np.ndarray]) -> List[InferenceOutput]:
        logger.debug(f"DummyEngine: Postprocessing {len(raw_outputs)} raw outputs.")
        processed_outputs: List[InferenceOutput] = []
        for i, output_data in enumerate(raw_outputs):
            # If there is a corresponding name in _output_names, use it, otherwise generate a virtual name
            output_name = self._output_names[i] if i < len(self._output_names) else f"dummy_output_{i}"
            processed_outputs.append(InferenceOutput(data=output_data, metadata={"name": output_name}))
        return processed_outputs

    def _batch_process(self, inputs_batch: List[Dict[str, np.ndarray]]) -> List[List[np.ndarray]]:
        logger.debug(f"DummyEngine: Simulating batch processing for {len(inputs_batch)} request item(s).")
        results_batch: List[List[np.ndarray]] = []

        for single_input_dict in inputs_batch:
            dummy_outputs_for_item: List[np.ndarray] = []
            num_expected_outputs = len(self._output_names) if self._output_names else 1
            
            actual_batch_size = 1 # Default batch size
            if self._input_names and self._input_names[0] in single_input_dict:
                first_input_tensor = single_input_dict[self._input_names[0]]
                if isinstance(first_input_tensor, np.ndarray) and first_input_tensor.ndim > 0:
                    actual_batch_size = first_input_tensor.shape[0] # Infer batch size from actual input
            
            for i in range(num_expected_outputs):
                output_dim = self._model_config.get("dummy_output_dim", 10) # Get virtual output dimension from model config
                output_shape = (actual_batch_size, output_dim)
                
                # Generate random dummy output data
                dummy_output_array = np.random.rand(*output_shape).astype(np.float32)
                logger.debug(f"  Generated dummy output tensor of shape {dummy_output_array.shape} for output slot {i}.")
                dummy_outputs_for_item.append(dummy_output_array)
            results_batch.append(dummy_outputs_for_item)
        return results_batch

    # Note: The predict method in ONNXEngine.py is synchronous.
    # If the AsyncEngine design is to have subclasses implement predict_sync_internal() and be called by AsyncEngine.predict() (asynchronous),
    # then this method should be named predict_sync_internal.
    # To maintain consistent override behavior with the provided ONNXEngine (if it indeed overrides the async predict), this method is temporarily named predict.
    # If your EngineService calls await engine.predict(), and expects the asynchronous behavior of AsyncEngine,
    # then this method should be renamed predict_sync_internal, and AsyncEngine.predict will handle the asynchronous call.
    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
    # def predict_sync_internal(self, inputs: List[InferenceInput]) -> InferenceResult: # 如果遵循 AsyncEngine 设计
        logger.info(f"DummyEngine: Received predict request with {len(inputs)} inputs.")
        if not self._model_loaded:
            logger.error("DummyEngine: Model not loaded for prediction.")
            return InferenceResult(success=False, error_message="DummyEngine: Model not loaded.")
        if not self._initialized:
            logger.error("DummyEngine: Engine not initialized for prediction.")
            return InferenceResult(success=False, error_message="DummyEngine: Engine not initialized.")

        start_time_sec = time.time()
        try:
            if not inputs:
                 logger.warning("DummyEngine: No inputs provided for prediction.")
                 return InferenceResult(success=False, error_message="DummyEngine: No inputs provided.")

            if self._processor:
                preprocessed_data_dict = self._processor.preprocess(inputs)
            else:
                preprocessed_data_dict = self._preprocess_input(inputs)
            if not preprocessed_data_dict and self._input_names: # If expected inputs but preprocessing returns empty
                logger.error("DummyEngine: Preprocessing returned no data, though inputs were expected.")
                return InferenceResult(success=False, error_message="DummyEngine: Preprocessing failed to produce data.")
            logger.debug(f"DummyEngine: Preprocessed data keys: {list(preprocessed_data_dict.keys())}")
            
            # _batch_process expects a list of requests, each request being a dictionary containing named tensors
            # predict usually handles a single "inference request", which may contain a batch of data items (combined by preprocessing)
            raw_outputs_list_of_lists = self._batch_process([preprocessed_data_dict]) 
            
            if not raw_outputs_list_of_lists or not raw_outputs_list_of_lists[0]: # Check if batch processing results are valid
                logger.error("DummyEngine: _batch_process returned unexpected empty or malformed results.")
                raise ValueError("DummyEngine: _batch_process returned invalid results structure.")
            
            raw_outputs_for_this_call = raw_outputs_list_of_lists[0] # Get the outputs for this call (since we only passed one request item)
            logger.debug(f"DummyEngine: Raw outputs from batch_process: {len(raw_outputs_for_this_call)} tensor(s).")

            final_outputs: List[InferenceOutput] = self._postprocess_output(raw_outputs_for_this_call)
            logger.debug(f"DummyEngine: Final postprocessed outputs: {len(final_outputs)}.")

            # Simulate processing delay (if configured)
            simulated_delay_ms = self._model_config.get("dummy_processing_delay_ms", 0)
            if simulated_delay_ms > 0:
                time.sleep(simulated_delay_ms / 1000.0)
            
            processing_time_ms = (time.time() - start_time_sec) * 1000
            logger.info(f"DummyEngine: Prediction successful. Processing time: {processing_time_ms:.2f} ms.")
            return InferenceResult(success=True, outputs=final_outputs, processing_time_ms=processing_time_ms)
        
        except Exception as e:
            logger.error(f"DummyEngine: Error during prediction: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time_sec) * 1000
            return InferenceResult(success=False, error_message=str(e), processing_time_ms=processing_time_ms)

    def get_info(self) -> EngineInfo:
        logger.debug("DummyEngine: Getting engine info.")
        status_str = "uninitialized"
        if self._initialized:
            status_str = "initialized (model not loaded)"
            if self._model_loaded:
                status_str = "ready"
        
        return EngineInfo(
            engine_name=self.ENGINE_NAME,
            engine_version=self._engine_config.get("dummy_version", "1.0.0_dummy"),
            model_loaded=self._model_loaded,
            loaded_model_path=self._loaded_model_path, # Get from base class
            available_devices=["CPU_dummy", "GPU_dummy_simulated"],
            current_device=self._current_device,
            engine_status=status_str,
            additional_info={
                "input_names": self._input_names,
                "output_names": self._output_names,
                "input_shapes": [str(s) for s in self._input_shapes], # Convert shapes to strings for consistency
                "input_types": self._input_types,
                "message": "This is a dummy engine. Behavior can be influenced by 'dummy_*' keys in engine_config and model_config.",
                "loaded_engine_config": self._engine_config,
                "loaded_model_config": self._model_config,
            }
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        logger.debug("DummyEngine: Getting resource requirements.")
        is_gpu_simulated = "GPU" in (self._current_device or "").upper()
        return ResourceRequirements(
            cpu_cores=self._engine_config.get("threads", 1), 
            memory_gb=self._engine_config.get("dummy_memory_gb", 0.05), # Very small value
            gpu_count=1 if is_gpu_simulated else 0,
        )

    def release(self) -> bool:
        current_loaded_path = self._loaded_model_path # Get before super().release() clears it
        logger.info(f"DummyEngine: Releasing model (if any was loaded from '{current_loaded_path}').")
        
        super_released = super().release() 
        
        # Reset DummyEngine-specific state
        self._input_names = []
        self._output_names = []
        self._input_shapes = []
        self._input_types = []
        # _current_device can be reset based on engine_config in initialize, here it can be left unchanged or set to default
        self._current_device = self._engine_config.get("dummy_device", "CPU_dummy") if self._engine_config else "CPU_dummy"
        self._model_config = {}
        
        if super_released:
            logger.info(f"DummyEngine: Model '{current_loaded_path}' and resources released successfully.")
        else:
            # This should not happen unless super().release() has specific logic that returns False
            logger.warning(f"DummyEngine: super().release() returned False for '{current_loaded_path}'. Check base class logic.")
        return super_released