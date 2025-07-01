import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import logging

from .base import AsyncEngine, EngineInfo, InferenceInput, InferenceOutput, InferenceResult, ResourceRequirements
from PIL import Image
import requests
import numpy as np

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.warning("WARNING: PaddleOCR library not found. OCREngine will not be available.")

try:
    import paddle
except ImportError:
    logger.warning(
        "WARNING: paddle library not found. OCREngine will not be available.")


class OCREngine(AsyncEngine):
    ENGINE_NAME = "OCREngine"

    def __init__(self, max_workers: Optional[int] = None, queue_size: Optional[int] = None):
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR library is not installed. "
                "Please install it to use OCREngine (e.g., 'pip install paddleocr==3.0.0')."
            )

        if paddle is None:
            raise RuntimeError(
                "PaddlePaddle library is not installed. "
                "Please install it to enable OCREngine (e.g., 'python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/')."
            )

        super().__init__(max_workers=max_workers, queue_size=queue_size)

        self._session = None
        self._model_loaded = False
        self.device = ""

    def _initialize_paddle_runtime(self) -> bool:
        """初始化引擎"""
        try:
            # 设置 Paddle 设备
            if paddle.is_compiled_with_cuda():
                paddle.set_device('gpu')
                logger.info("使用 GPU 进行 OCR 推理")
                print("使用 GPU 进行 OCR 推理")
                self.device = 'gpu'
            else:
                paddle.set_device('cpu')
                logger.info("使用 CPU 进行 OCR 推理")
                print("使用 CPU 进行 OCR 推理")
                self.device = 'cpu'
            return True

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            self._model_loaded = False
            return False

    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        super_init_ok = super().initialize(engine_config)
        if not super_init_ok:
            return False
        return self._initialize_paddle_runtime()

    def _load_model_specifico(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """加载OCR模型并根据关键字识别检测和识别模型路径"""
        import os

        try:
            logger.info(f"正在扫描模型路径: {model_path}")

            # 检查路径是否存在
            if not os.path.exists(model_path):
                raise ValueError(f"模型路径不存在: {model_path}")

            # 获取路径下的所有文件夹
            folders = []
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path):
                    folders.append((item, item_path))

            if not folders:
                raise ValueError(f"路径 {model_path} 下没有找到文件夹")

            logger.info(f"发现 {len(folders)} 个文件夹: {[name for name, _ in folders]}")

            # 根据关键字识别模型路径，优先识别服务级模型
            det_model_path = None
            rec_model_path = None
            cls_model_path = None

            for folder_name, folder_path in folders:
                folder_lower = folder_name.lower()

                # 检测模型路径识别
                if 'det' in folder_lower:
                    det_model_path = folder_path
                    logger.info(f"识别到检测模型: {folder_name} -> {folder_path}")
                # 识别模型路径识别
                elif 'rec' in folder_lower:
                    rec_model_path = folder_path
                    logger.info(f"识别到识别模型: {folder_name} -> {folder_path}")
                # 分类模型路径识别（可选）
                elif 'cls' in folder_lower or 'angle' in folder_lower:
                    cls_model_path = folder_path
                    logger.info(f"识别到分类模型: {folder_name} -> {folder_path}")

            # 验证是否找到了必要的模型
            if det_model_path is None:
                logger.warning("未找到包含'det'关键字的检测模型文件夹")
            if rec_model_path is None:
                logger.warning("未找到包含'rec'关键字的识别模型文件夹")

            self._session = PaddleOCR(
                text_detection_model_dir=det_model_path,
                text_recognition_model_dir=rec_model_path,
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            logger.info(f"PaddleOCR模型加载成功")
            self._model_loaded = True
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self._model_loaded = False
            return False

    def _preprocess_input(self, raw_inputs: List[InferenceInput]) -> Dict[str, List[np.ndarray]]:
        """
        OCR 预处理：从 InferenceInput 中提取图像数据并转换为图像数组
        只支持以下输入格式：
        - dict 包含 'url' 键
        - dict 包含 'image_base64' 或 'b64' 键
        - 直接的 Base64 字符串
        - 直接的 URL 字符串
        """
        images: List[np.ndarray] = []

        for idx, inp in enumerate(raw_inputs):
            data = inp.data
            img = None

            try:
                logger.info(f"🔍 [索引 {idx}] 处理输入数据类型: {type(data)}")
                logger.info(f"🔍 [索引 {idx}] 数据内容预览: {str(data)[:100]}...")

                # 处理 URL (字典格式)
                if isinstance(data, dict) and "url" in data:
                    url = data["url"]
                    logger.info(f"📥 [索引 {idx}] 检测到字典格式的URL: {url}")
                    img = self._download_image_as_array(url, idx)

                # 处理 URL (直接字符串)
                elif isinstance(data, str) and data.lower().startswith("http"):
                    logger.info(f"📥 [索引 {idx}] 检测到字符串格式的URL: {data}")
                    img = self._download_image_as_array(data, idx)

                # 处理 Base64 (字典格式)
                elif isinstance(data, dict) and ("image_base64" in data or "b64" in data):
                    key = "image_base64" if "image_base64" in data else "b64"
                    logger.info(f"📥 [索引 {idx}] 检测到字典格式的Base64，键名: {key}")
                    img = self._decode_base64_to_image(data[key], idx)

                # 处理 Base64 (直接字符串)
                elif isinstance(data, str):
                    logger.info(f"📥 [索引 {idx}] 检测到字符串格式，判断为Base64")
                    b64_str = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
                    img = self._decode_base64_to_image(b64_str, idx)

                else:
                    raise ValueError(f"不支持的输入数据格式 at index {idx}: {type(data)}. 仅支持 URL 和 Base64 格式")

                # 标准化图像格式
                if img is not None:
                    img_normalized = self._normalize_image_channels(img, idx)
                    images.append(img_normalized)
                    logger.info(f"✅ [索引 {idx}] 成功处理，图像尺寸: {img_normalized.shape}")
                else:
                    raise ValueError(f"处理输入 {idx} 失败，无法生成有效图像")

            except Exception as e:
                msg = f"预处理失败 at index {idx}, input={type(data)}, error={e}"
                logger.error(msg, exc_info=True)
                raise ValueError(msg)

        if not images:
            raise ValueError("没有生成有效的图像数组")

        logger.info(f"🎉 预处理完成: {len(images)} 个图像数组")
        return {"images": images}

    def _postprocess_output(self, raw_outputs: List[Any]) -> List[InferenceOutput]:
        """
        OCR后处理：将PaddleOCR的原始输出转换为标准的InferenceOutput格式

        Args:
            raw_outputs: PaddleOCR返回的原始结果列表，可能包含多种格式的数据

        Returns:
            List[InferenceOutput]: 标准化的推理输出列表
        """
        # 为批处理构建结果结构
        batch_results = []

        # 处理单个图像的结果（如果是批处理，这里需要循环处理多个图像）
        image_result = {
            "image_id": 0,  # 图像索引或ID
            "detections": []  # 该图像的所有检测结果
        }

        # 提取识别的文本内容
        if raw_outputs and len(raw_outputs) > 0:
            result_data = raw_outputs[0]
            texts = result_data.get('rec_texts', [])
            scores = result_data.get('rec_scores', [])
            polys = result_data.get('rec_polys', []) or result_data.get('rec_boxes', [])

            for i, text in enumerate(texts):
                confidence = scores[i] if i < len(scores) else 0.0

                # 处理坐标转换为 [x, y, w, h] 格式
                if i < len(polys) and polys[i] is not None:
                    coords = polys[i].tolist() if hasattr(polys[i], 'tolist') else polys[i]

                    # 从多边形坐标计算边界框
                    if coords and len(coords) > 0:
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]

                        # 计算边界框坐标
                        left = min(x_coords)
                        top = min(y_coords)
                        right = max(x_coords)
                        bottom = max(y_coords)

                        # 转换为 [x, y, w, h] 格式（分辨率坐标，int类型）
                        x = int(left)
                        y = int(top)
                        w = int(right - left)
                        h = int(bottom - top)

                        coords = [x, y, w, h]
                    else:
                        coords = [0, 0, 0, 0]
                else:
                    coords = [0, 0, 0, 0]

                # 构建检测结果
                detection: Dict[str, Any] = {
                    "coords": coords,
                    "confidence": round(float(confidence), 2),
                    "text": text
                }

                image_result["detections"].append(detection)

        batch_results.append(image_result)

        return [InferenceOutput(data=batch_results)]

    def _batch_process(self, inputs_batch: List[Dict[str, np.ndarray]]) -> List[List[np.ndarray]]:
        """
        批处理OCR推理

        Args:
            inputs_batch: 批量输入数据，每个元素包含图像数据

        Returns:
            List[Any]: 批量OCR推理结果
        """
        if not self._session:
            raise RuntimeError("Paddle OCR session not initialized.")

        results_batch = []
        for single_input_dict in inputs_batch:
            # 获取图像数据
            images = single_input_dict.get("images")
            if images is None:
                raise ValueError("No 'images' key found in input data.")

            # 执行OCR推理
            ocr_result = self._session.predict(images)
            results_batch.append(ocr_result)

        return results_batch

    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
        if not self._model_loaded or not self._session:
            return InferenceResult(success=False, error_message="Paddle model not loaded.")

        start_time_sec = time.time()  # Corrected

        try:
            # 预处理
            if self._processor:
                logger.info(f"Processor: {self._processor}")
                preprocessed_data_dict = self._processor.preprocess(inputs)
            else:
                logger.info(f"No processor found. Using default preprocess_input.")
                preprocessed_data_dict = self._preprocess_input(inputs)

            logger.info(f"Preprocessed data keys: {list(preprocessed_data_dict.keys())}")

            # 批处理推理
            raw_outputs_list = self._batch_process([preprocessed_data_dict])
            raw_outputs_for_this_call = raw_outputs_list[0]

            # 后处理
            if self._processor:
                final_outputs = self._processor.postprocess(raw_outputs_for_this_call)
            else:
                final_outputs = self._postprocess_output(raw_outputs_for_this_call)

            logger.info(f"Final outputs count: {len(final_outputs)}")
            processing_time_ms = (time.time() - start_time_sec) * 1000

            return InferenceResult(success=True, outputs=final_outputs,processing_time_ms=processing_time_ms)
        except Exception as e:
            logger.error(f"Error during Paddl prediction: {e}")
            return InferenceResult(success=False, error_message=str(e),
                                   processing_time_ms=(time.time() - start_time_sec) * 1000)  # Corrected

    def get_info(self) -> EngineInfo:
        """返回OCR引擎信息"""
        engine_info = EngineInfo(
            engine_name=self.ENGINE_NAME,
            engine_version=paddle.__version__ if paddle else "N/A",
            model_loaded=self._model_loaded,
            loaded_model_path=None,
            available_devices=[],
            current_device=None,
            engine_status="",
            additional_info={
                "session_providers": [],
                "engine_config_providers": "",

                "input_names": {
                  "name": "input_name",
                  "type": "tensor",
                  "shape": [1, 3, 224, 224]
                },
                "output_names": {
                  "name": "output_name",
                  "type": "tensor",
                  "shape": [1, 1000]
                },
                "input_shapes": [],
                "models_type": [],
                "models_labels": [],
                "algorithm": [],

                "desc": []
            }
        )
        logger.info(f"EngineInfo: {engine_info}")
        return engine_info

    def get_resource_requirements(self) -> ResourceRequirements:
        mem_usage = 0.0
        return ResourceRequirements(
            cpu_cores=1,
            memory_gb=0.5,
            gpu_count=1,
        )

    def release(self) -> bool:
        """释放资源"""
        try:
            logger.info("释放 OCR 引擎资源")

            if self._session:
                self._session = None
                logger.info("OCR 会话已清理")

            # 清理 PaddlePaddle 的 GPU 显存缓存
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                logger.info("GPU 缓存已清理")

            self._model_loaded = False
            return True

        except Exception as e:
            logger.error(f"释放资源失败: {e}")
            return False

    def test_inference(self, test_inputs: Optional[List[InferenceInput]] = None) -> InferenceResult:
        """
        测试推理引擎端到端可用性
        """
        logger.info(f"Performing test inference on {self.__class__.__name__}...")
        start_time_sec = time.time()

        try:
            # 创建虚拟图像数据用于测试
            # OCR通常处理BGR格式的图像 (height, width, channels)
            height, width, channels = 640, 640, 3
            logger.debug(f"OCR测试输入维度: [{height}, {width}, {channels}]")

            # 生成随机图像数据 (0-255范围，uint8类型，BGR格式)
            raw_img = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

            # 执行批处理推理测试
            _ = self._batch_process([{"images": raw_img}])

            end_time_sec = (time.time() - start_time_sec) * 1000

            logger.info(f"OCR test inference completed successfully in {end_time_sec:.2f}ms")

            return InferenceResult(success=True, processing_time_ms=end_time_sec)

        except Exception as e:
            end_time_sec = (time.time() - start_time_sec) * 1000
            logger.error(f"Exception during test inference: {e}")
            return InferenceResult(
                success=False, error_message=str(e), processing_time_ms=end_time_sec)


    def _download_image_as_array(self, url: str, idx: int) -> np.ndarray:
        """从 URL 下载图像并直接返回图像数组"""
        try:
            logger.info(f"📡 [索引 {idx}] 正在下载图像: {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            # 检查内容类型
            ct = resp.headers.get("Content-Type", "")
            if not ct.startswith("image/"):
                raise ValueError(f"URL 未返回图像内容: Content-Type={ct}")

            # 解码图像
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("无法解码下载的图像")

            logger.info(f"✅ [索引 {idx}] 成功下载图像: shape={img.shape}")
            return img

        except Exception as e:
            logger.error(f"❌ [索引 {idx}] 从 URL 下载图像失败, url={url}: {e}")
            raise ValueError(f"从 URL 下载图像失败 [索引 {idx}], url={url}: {e}")

    def _decode_base64_to_image(self, b64_str: str, idx: int) -> np.ndarray:
        """将 Base64 字符串解码为图像数组"""
        try:
            logger.info(f"🔓 [索引 {idx}] 正在解码 Base64 图像...")

            # 解码 Base64
            b64_str = b64_str.strip()
            img_bytes = base64.b64decode(b64_str)

            # 尝试用 OpenCV 解码
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                # 回退到 PIL
                logger.info(f"🔄 [索引 {idx}] OpenCV 解码失败，尝试 PIL...")
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            logger.info(f"✅ [索引 {idx}] 成功解码 Base64 图像: shape={img.shape}")
            return img

        except Exception as e:
            logger.error(f"❌ [索引 {idx}] 解码 Base64 图像失败: {e}")
            raise ValueError(f"解码 Base64 图像失败 [索引 {idx}]: {e}")

    def _normalize_image_channels(self, img: np.ndarray, idx: int) -> np.ndarray:
        """标准化图像通道格式为 BGR 三通道"""
        try:
            logger.debug(f"🔧 [索引 {idx}] 标准化图像通道，原始形状: {img.shape}")

            if len(img.shape) == 2:
                # 灰度图转 BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                logger.debug(f"🔄 [索引 {idx}] 灰度图已转换为BGR")
            elif len(img.shape) == 3:
                if img.shape[2] == 1:
                    # 单通道转 BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    logger.debug(f"🔄 [索引 {idx}] 单通道图已转换为BGR")
                elif img.shape[2] == 3:
                    # 已经是三通道，保持不变
                    logger.debug(f"✅ [索引 {idx}] 已是BGR三通道")
                elif img.shape[2] == 4:
                    # RGBA转BGR，丢弃Alpha通道
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    logger.debug(f"🔄 [索引 {idx}] RGBA图已转换为BGR")
                else:
                    # 超过4通道，取前3个通道
                    img = img[:, :, :3]
                    logger.warning(f"⚠️ [索引 {idx}] 多通道图像({img.shape[2]}通道)已截取前3个通道")
            else:
                raise ValueError(f"不支持的图像维度: {img.shape}")

            # 确保最终是三通道BGR格式
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"通道转换后仍非三通道图像，shape={img.shape}")

            logger.debug(f"✅ [索引 {idx}] 通道标准化完成: {img.shape}")
            return img

        except Exception as e:
            logger.error(f"❌ [索引 {idx}] 图像通道标准化失败: {e}")
            raise ValueError(f"图像通道标准化失败 [索引 {idx}]: {e}")