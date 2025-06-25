import time
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

try:
    import tensorrt as trt
except ImportError:
    logger.warning("WARNING: TensorRT library not found. TensorRTEngine will not be available.")
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    logger.warning("WARNING: pycuda library not found. TensorRTEngine will not be available.")
    cuda = None

from .base import AsyncEngine, EngineInfo, InferenceInput, InferenceOutput, InferenceResult, ResourceRequirements


class TensorRTEngine(AsyncEngine):
    ENGINE_NAME = "TensorRT"

    def __init__(self, max_workers: Optional[int] = None, queue_size: Optional[int] = None):
        super().__init__(max_workers=max_workers, queue_size=queue_size)

        self._trt_runtime: Optional[trt.Runtime] = None
        self._engine: Optional[trt.ICudaEngine] = None
        self._context: Optional[trt.IExecutionContext] = None
        self._host_buffers: Dict[str, np.ndarray] = {}
        self._device_buffers: Dict[str, cuda.DeviceAllocation] = {}
        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, Any] = {}
        self._input_names: List = []
        self._output_names: List = []
        self._stream: Optional[cuda.Stream] = None
        self._model_loaded: bool = False
        self._loaded_model_path: Optional[str] = None

        self._bindings: list = []

    # 测试通过
    def _initialize_tensor_runtime(self) -> bool:
        try:
            if trt is None or cuda is None:
                logger.error("CUDA or TensorRT not available.")
                return False

            self._trt_logger = trt.Logger(trt.Logger.WARNING)
            logger.info("Created TensorRT logger at WARNING level.")

            self._trt_runtime = trt.Runtime(self._trt_logger)
            logger.info("TensorRT Runtime created successfully.")

            # —— 3. 校验并应用设备配置 —— #
            device = self._engine_config.get("device", "GPU")
            if device != "GPU":
                logger.error("TensorRT only supports GPU device.")
                return False
            # 如果你还想切换到非 0 号设备，可以在这里用 cuda.Device(idx).make_context()

            # —— 4. 初始化 CUDA Stream —— #
            self._stream = cuda.Stream(1)
            logger.info("CUDA Stream initialized.")

            logger.info("TensorRT engine runtime initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TensorRT runtime: {e}")
            return False

    # 测试通过
    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        super_init_ok = super().initialize(engine_config)
        if not super_init_ok:
            return False
        return self._initialize_tensor_runtime()

    # 测试通过
    def _load_model_specifico(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """
                反序列化 TensorRT 引擎并分配上下文 & buffers，
                使用 enumerate(self._engine) 来遍历所有 binding。
                """
        if self._trt_runtime is None:
            logger.error("Cannot load model: TRT Runtime not initialized.")
            return False

        try:
            # 1. 反序列化引擎
            engine_data = Path(model_path).read_bytes()
            self._engine = self._trt_runtime.deserialize_cuda_engine(engine_data)
            if self._engine is None:
                logger.error("Failed to deserialize CUDA engine.")
                return False
            logger.info(f"Deserialized TensorRT engine from {model_path}")

            # 2. 创建执行上下文
            self._context = self._engine.create_execution_context()

            # 3. 清空旧状态
            self._host_buffers.clear()
            self._device_buffers.clear()
            self._shapes.clear()
            self._dtypes.clear()
            self._input_names.clear()
            self._output_names.clear()
            self._bindings.clear()

            # 4. 遍历所有 I/O tensor，分配并记录信息
            for i in range(self._engine.num_io_tensors):
                name = self._engine[i]  # binding 名称
                mode = self._engine.get_tensor_mode(name)
                shape = tuple(self._engine.get_tensor_shape(name))
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                size = int(np.prod(shape))

                # 分配内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                # 写回字典
                self._host_buffers[name] = host_mem
                self._device_buffers[name] = device_mem
                self._shapes[name] = shape
                self._dtypes[name] = dtype

                # 塞入 bindings（TensorRT 执行需要的 device 指针列表）
                self._bindings.append(int(device_mem))

                # 记录输入/输出名
                if mode == trt.TensorIOMode.INPUT:
                    self._input_names.append(name)
                else:
                    self._output_names.append(name)

            # 5. 标记成功
            self._model_loaded = True
            self._loaded_model_path = str(model_path)
            self._engine_config["engine_path"] = str(model_path)

            logger.info(
                f"模型加载成功：\n"
                f"- 输入 names:  {self._input_names}\n"
                f"- 输出 names: {self._output_names}\n"
                f"- shapes:      {self._shapes}\n"
                f"- dtypes:      {self._dtypes.keys()}"
            )
            print(
                f"模型加载成功：\n"
                f"- 输入 names: {self._input_names}\n"
                f"- 输出 names: {self._output_names}\n"
                f"- shapes:      {self._shapes}\n"
                f"- dtypes:      {self._dtypes.keys()}"
            )
            return True

        except Exception as e:
            logger.error(f"Exception during engine load: {e}")
            return False

    def _batch_process(self, inputs_batch: List[Any]) -> List[Any]:
        pass

    def _preprocess_input(self, raw_inputs: List[InferenceInput]) -> Any:
        pass

    def _postprocess_output(self, raw_outputs: Any) -> List[InferenceOutput]:
        pass

    def test_inference(self, test_inputs: Optional[List[InferenceInput]] = None) -> InferenceResult:
        pass

    def _preprocess(self, inputs: List[InferenceInput]) -> None:
        # 假设单输入，多输入可自行扩展
        inp = inputs[0]
        name = self._input_names[0]
        arr = np.ascontiguousarray(inp.data.astype(self._dtypes[name]).reshape(self._shapes[name]))
        np.copyto(self._host_buffers[name], arr.ravel())

    def _postprocess(self) -> List[InferenceOutput]:
        outputs: List[InferenceOutput] = []
        for name in self._output_names:
            data = np.array(self._host_buffers[name]).reshape(self._shapes[name])
            outputs.append(InferenceOutput(data=data, metadata={"name": name}))
        return outputs

    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
        if not self._model_loaded or self._engine is None or self._context is None:
            return InferenceResult(success=False, error_message="Model not loaded.")
        start = time.time()
        try:
            # 预处理
            self._preprocess(inputs)
            # 传输输入到设备
            for name in self._input_names:
                cuda.memcpy_htod_async(
                    self._device_buffers[name], self._host_buffers[name], self._stream
                )
            # 推理
            self._context.execute_async_v2(
                bindings=[int(self._device_buffers[n]) for n in (self._input_names + self._output_names)],
                stream_handle=self._stream.handle
            )
            # 拷贝输出到 Host
            for name in self._output_names:
                cuda.memcpy_dtoh_async(
                    self._host_buffers[name], self._device_buffers[name], self._stream
                )
            self._stream.synchronize()
            # 后处理
            outputs = self._postprocess()
            elapsed = (time.time() - start) * 1000
            return InferenceResult(success=True, outputs=outputs, processing_time_ms=elapsed)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return InferenceResult(success=False, error_message=str(e), processing_time_ms=elapsed)

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_name=self.ENGINE_NAME,
            engine_version=trt.__version__,
            model_loaded=self._model_loaded,
            loaded_model_path=self._loaded_model_path,
            available_devices=["GPU"],
            current_device="GPU",
            engine_status="ready" if self._model_loaded else "uninitialized",
            additional_info={}
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            cpu_cores=1,
            memory_gb=0.5,
            gpu_count=1
        )

    def release(self) -> bool:
        super().release()
        # 释放设备内存
        for mem in self._device_buffers.values():
            mem.free()
        self._device_buffers.clear()
        self._host_buffers.clear()
        self._shapes.clear()
        self._dtypes.clear()
        self._input_names.clear()
        self._output_names.clear()
        self._engine = None
        self._context = None
        logger.info(f"{self.ENGINE_NAME} resources released.")
        return True
