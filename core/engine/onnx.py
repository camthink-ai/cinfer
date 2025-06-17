# cinfer/engine/onnx.py
import time # Corrected
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

try:
    import onnxruntime
except ImportError:
    logger.warning("WARNING: onnxruntime library not found. ONNXEngine will not be available.")
    onnxruntime = None 

from .base import AsyncEngine, EngineInfo, InferenceInput, InferenceOutput, InferenceResult, ResourceRequirements

class ONNXEngine(AsyncEngine):
    ENGINE_NAME = "ONNXRuntime"

    def __init__(self, max_workers: Optional[int] = None, queue_size: Optional[int] = None):
        if onnxruntime is None:
            raise RuntimeError(
                "ONNXRuntime library is not installed. "
                "Please install it to use ONNXEngine (e.g., 'pip install onnxruntime' or 'pip install onnxruntime-gpu')."
            )
        super().__init__(max_workers=max_workers, queue_size=queue_size)
        self._session: Optional[onnxruntime.InferenceSession] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: List[Tuple[Optional[int], ...]] = []
        self._input_types: List[str] = []
        self._session_options: Optional[onnxruntime.SessionOptions] = None


    def _initialize_onnx_runtime(self) -> bool:
        available_providers = onnxruntime.get_available_providers()
        logger.info(f"Available ONNXRuntime providers: {available_providers}")

        # 获取用户配置，如果未配置则根据系统支持自动选择
        providers = self._engine_config.get("execution_providers", None)
        provider_options = self._engine_config.get("provider_options", None)  # optional

        if not providers:
            # 自动选择：优先使用 CUDAExecutionProvider
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider"]
                logger.info("No execution provider specified. Using CUDAExecutionProvider by default.")
            elif "CPUExecutionProvider" in available_providers:
                providers = ["CPUExecutionProvider"]
                logger.info("No execution provider specified. Using CPUExecutionProvider as fallback.")
            else:
                logger.error("Error: No execution providers available on this system.")
                return False
        elif isinstance(providers, str):
            providers = [providers]

        # 过滤掉无效 provider
        valid_providers = []
        for provider in providers:
            if provider in available_providers:
                valid_providers.append(provider)
            else:
                logger.warning(f"Warning: Provider '{provider}' not available. Skipping.")

        if not valid_providers:
            logger.error(
                "Error: No valid ONNXRuntime execution providers configured or available. Defaulting to CPUExecutionProvider if possible.")
            if "CPUExecutionProvider" in available_providers:
                valid_providers = ["CPUExecutionProvider"]
            else:
                return False

        self._engine_config["execution_providers"] = valid_providers
        self._engine_config["provider_options"] = provider_options

        self._session_options = onnxruntime.SessionOptions()
        num_threads = self._engine_config.get("threads", 0)
        # TODO: distinguish between intra_op_num_threads and inter_op_num_threads
        if num_threads > 0:
            self._session_options.intra_op_num_threads = num_threads
            self._session_options.inter_op_num_threads = num_threads

        logger.info(f"ONNXRuntime initialized with providers: {valid_providers}")
        return True

    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        super_init_ok = super().initialize(engine_config)
        if not super_init_ok:
            return False
        return self._initialize_onnx_runtime()

    def _optimize_session(self):
        #skip for now
        logger.info("ONNX session optimization step (if any specific optimizations are applied here).")
        pass

    def _load_model_specifico(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        try:
            self._optimize_session()
            if self._session_options is None: # Should have been set in _initialize_onnx_runtime
                 self._session_options = onnxruntime.SessionOptions()

            #load the model
            self._session = onnxruntime.InferenceSession(
                model_path,
                sess_options=self._session_options,
                providers=self._engine_config["execution_providers"],
                provider_options=self._engine_config.get("provider_options")
            )
            
            #get the input and output names, shapes, and types
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            self._input_shapes = [inp.shape for inp in self._session.get_inputs()]
            self._input_types = [inp.type for inp in self._session.get_inputs()]

            logger.info(f"ONNX Model '{model_path}' loaded.")
            logger.info(f"  Input Names: {self._input_names}")
            logger.info(f"  Input Shapes: {self._input_shapes}")
            logger.info(f"  Input Types: {self._input_types}")
            logger.info(f"  Output Names: {self._output_names}")
            
            self._current_device = self._session.get_providers()[0]
            self._model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading ONNX model {model_path}: {e}")
            self._session = None
            return False

    def _preprocess_input(self, raw_inputs: List[InferenceInput]) -> Dict[str, np.ndarray]:
        if not raw_inputs:
            raise ValueError("Input list cannot be empty for preprocessing.")
        
        if len(self._input_names) == 1:
            batch_data = [inp.data for inp in raw_inputs]
            try:
                np_batch = np.stack(batch_data)
                # Ensure correct dtype based on model input type
                expected_dtype = self._ort_type_to_numpy_type(self._input_types[0])
                if np_batch.dtype != expected_dtype:
                    np_batch = np_batch.astype(expected_dtype)

            except Exception as e:
                if len(raw_inputs) == 1:
                    np_batch = np.array(raw_inputs[0].data, dtype=self._ort_type_to_numpy_type(self._input_types[0]))
                else:
                    raise ValueError(f"Cannot automatically batch inputs for ONNX: {e}. Provide uniform inputs or implement custom batch preprocessing.")
            return {self._input_names[0]: np_batch}
        else:
            if len(raw_inputs) == 1 and isinstance(raw_inputs[0].data, dict):
                 return {
                     name: np.array(raw_inputs[0].data[name], dtype=self._ort_type_to_numpy_type(self._input_types[idx]))
                     for idx, name in enumerate(self._input_names) if name in raw_inputs[0].data
                 }
            raise NotImplementedError(
                "Default ONNX preprocessing for multiple model inputs or complex batching is not implemented. "
                "Please provide specific preprocessing logic or a single input that's a dict."
            )

    def _ort_type_to_numpy_type(self, ort_type_str: str) -> np.dtype:
        type_mapping = {
            'tensor(float16)': np.float16,
            'tensor(float)': np.float32,
            'tensor(double)': np.float64,
            'tensor(int8)': np.int8,
            'tensor(int16)': np.int16,
            'tensor(int32)': np.int32,
            'tensor(int64)': np.int64,
            'tensor(uint8)': np.uint8,
            'tensor(uint16)': np.uint16,
            'tensor(uint32)': np.uint32,
            'tensor(uint64)': np.uint64,
            'tensor(bool)': np.bool_,
        }
        # Attempt to match the full string first
        if ort_type_str in type_mapping:
            return type_mapping[ort_type_str]
        
        # Fallback for partial matches if full string not found (e.g. "float" in "tensor(float)")
        for key_part, np_type in type_mapping.items():
            if key_part.replace("tensor(", "").replace(")", "") in ort_type_str: # e.g. "float" in "tensor(float)"
                return np_type
        
        logger.warning(f"Warning: Unhandled ONNX data type '{ort_type_str}' for NumPy conversion. Defaulting to np.float32.")
        return np.float32


    def _postprocess_output(self, raw_outputs: List[np.ndarray]) -> List[InferenceOutput]:
        if len(raw_outputs) != len(self._output_names):
            raise ValueError("Number of raw outputs does not match expected output names.")
            
        processed_outputs = []
        for i, name in enumerate(self._output_names):
            processed_outputs.append(InferenceOutput(data=raw_outputs[i], metadata={"name": name}))
        return processed_outputs

    def _batch_process(self, inputs_batch: List[Dict[str, np.ndarray]]) -> List[List[np.ndarray]]:
        if not self._session:
            raise RuntimeError("ONNX session not initialized.")

        results_batch = []
        for single_input_dict in inputs_batch:
            ort_inputs = {self._input_names[0]: single_input_dict["images"]}
            raw_outputs = self._session.run(self._output_names, ort_inputs)
            results_batch.append(raw_outputs)
        return results_batch

    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
        if not self._model_loaded or not self._session:
            return InferenceResult(success=False, error_message="ONNX model not loaded.")
        
        start_time_sec = time.time() # Corrected
        try:
            if self._processor:
                preprocessed_data_dict = self._processor.preprocess(inputs)
            else:
                preprocessed_data_dict = self._preprocess_input(inputs)
            raw_outputs_list_of_lists = self._batch_process([preprocessed_data_dict])
            raw_outputs_for_this_call = raw_outputs_list_of_lists[0]
            if self._processor:
                final_outputs = self._processor.postprocess(raw_outputs_for_this_call)
            else:
                final_outputs = self._postprocess_output(raw_outputs_for_this_call)
            processing_time_ms = (time.time() - start_time_sec) * 1000 # Corrected
            return InferenceResult(success=True, outputs=final_outputs, processing_time_ms=processing_time_ms)
        except Exception as e:
            logger.error(f"Error during ONNX prediction: {e}")
            return InferenceResult(success=False, error_message=str(e), processing_time_ms=(time.time() - start_time_sec) * 1000) # Corrected

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_name=self.ENGINE_NAME,
            engine_version=onnxruntime.__version__ if onnxruntime else "N/A",
            model_loaded=self._model_loaded,
            loaded_model_path=self._loaded_model_path,
            available_devices=onnxruntime.get_available_providers() if onnxruntime else [],
            current_device=self._current_device or (self._session.get_providers()[0] if self._session else None),
            engine_status="ready" if self._initialized and self._model_loaded else ("initialized" if self._initialized else "uninitialized"),
            additional_info={
                "input_names": self._input_names,
                "output_names": self._output_names,
                "input_shapes": [str(s) for s in self._input_shapes],
                "session_providers": self._session.get_providers() if self._session else [],
                "engine_config_providers": self._engine_config.get("execution_providers")
            }
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        mem_usage = 0.0
        return ResourceRequirements(
            cpu_cores=self._engine_config.get("threads", 1),
            memory_gb=mem_usage or 0.5,
            gpu_count=1 if any(p in self._engine_config.get("execution_providers", []) for p in ["CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider"]) else 0,
        )

    def release(self) -> bool:
        super_released = super().release()
        if self._session is not None:
            del self._session
            self._session = None
            logger.info(f"{self.ENGINE_NAME} session released.")
        self._input_names = []
        self._output_names = []
        self._input_shapes = []
        self._input_types = []
        return super_released