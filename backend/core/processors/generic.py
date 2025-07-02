import base64, io
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from core.processors import BaseProcessor
from core.processors.algorithm import Algorithm, Norm, ChannelType
from schemas.engine import InferenceInput, InferenceOutput, EngineInfo
from utils.filter import Filter

import logging

logger = logging.getLogger(f"cinfer.{__name__}")

try:
    import cv2
except ImportError:
    logger.warning("WARNING: cv2 (OpenCV) library not found. OpenCV functionality will not be available.")
    cv2 = None

try:
    from PIL import Image
except ImportError:
    logger.warning("WARNING: PIL (Pillow) library not found. Image processing functionality will not be available.")
    Image = None

try:
    import requests
except ImportError:
    logger.warning(
        "WARNING: requests library not found. "
        "HTTP functionality will not be available."
    )
    requests = None


class Generic(BaseProcessor):
    def __init__(self, model_config: Dict[str, Any], engine_info: EngineInfo):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) library is not installed. "
                "Please install it to enable OpenCV features "
                "(e.g., 'pip install opencv-python==4.11.0.86' or 'pip install opencv-python')."
            )
        if Image is None:
            raise RuntimeError(
                "Pillow (PIL) library is not installed. "
                "Please install it to enable image processing features "
                "(e.g., 'pip install pillow==11.2.1' or 'pip install pip install pillow')."
            )
        if requests is None:
            raise RuntimeError(
                "No HTTP client available (requests). "
                "Please install one with `pip install requests`."
            )

        super().__init__(model_config, engine_info)

        self._algorithm = Algorithm
        self._engine_info = engine_info.additional_info

        # Extract engine info
        info = engine_info.additional_info
        self.class_names = info.get("models_labels")
        self.MODEL_TYPE = info.get("models_type")
        _, _, h, w = info.get("input_shapes")
        # 最终保存到实例属性
        self.height, self.width = int(h), int(w)

        self.outdata = Filter(self.MODEL_TYPE, self.class_names)

        # State buffers
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45

        self.MASK_COEFFS_DIM = 32
        self._last_d2i_matrices: List[np.ndarray] = []
        self._last_original_shapes: List[Tuple[int, int]] = []
        self._last_file_names: List[str] = []

    def preprocess(self, inputs: List[InferenceInput]) -> Dict[str, Any]:
        """"
        对一批图像进行预处理。
        支持四种输入：
          - 直接的 numpy.ndarray
          - dict 包含 'url' 键
          - dict 包含 'image_base64' 或 'b64' 键
          - 直接的 Base64 字符串
        返回:
          - 批处理张量
          - d2i_matrices 列表
          - original_shapes 列表
        """
        processed_tensors: List[np.ndarray] = []
        d2i_matrices: List[np.ndarray] = []
        original_shapes: List[Tuple[int, int]] = []
        norm_params = Norm.alpha_beta(alpha=1 / 255.0, channel_type=ChannelType.NONE)

        for idx, inp in enumerate(inputs):
            data = inp.data
            metadata = inp.metadata or {}
            self.conf_threshold = metadata.get("conf_threshold", self.conf_threshold)
            self.nms_threshold = metadata.get("nms_threshold", self.nms_threshold)
            try:
                if isinstance(data, np.ndarray):
                    img = data

                elif isinstance(data, dict) and "url" in data:
                    url = data["url"]
                    resp = requests.get(url, timeout=5)
                    resp.raise_for_status()
                    ct = resp.headers.get("Content-Type", "")
                    if not ct.startswith("image/"):
                        raise ValueError(
                            f"URL did not return image content at index={idx}, url={url}, Content-Type={ct}")
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError(f"Failed to decode image from URL at index={idx}, url={url}")

                elif isinstance(data, str) and data.lower().startswith("http"):
                    url = data
                    resp = requests.get(url, timeout=5)
                    resp.raise_for_status()
                    ct = resp.headers.get("Content-Type", "")
                    if not ct.startswith("image/"):
                        raise ValueError(
                            f"URL did not return image content at index={idx}, url={url}, Content-Type={ct}")
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError(f"Failed to decode image from URL at index={idx}, url={url}")

                elif isinstance(data, dict) and ("image_base64" in data or "b64" in data):
                    key = "image_base64" if "image_base64" in data else "b64"
                    img = self._decode_base64_to_image(data[key], idx)

                elif isinstance(data, str):
                    b64_str = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
                    img = self._decode_base64_to_image(b64_str, idx)

                else:
                    raise ValueError(f"Unsupported input data format at index {idx}: {type(data)}")

            except Exception as e:
                msg = f"Preprocess failed at index {idx}, input={repr(inp)}, error={e}"
                logger.error(msg, exc_info=True)
                raise ValueError(msg)

            if img is None or not isinstance(img, np.ndarray):
                msg = f"Decoded image invalid at index {idx}."
                logger.error(msg)
                raise ValueError(msg)

            # 通道检查和转换逻辑
            if len(img.shape) == 2:
                # 灰度图转BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                logger.info(f"Index {idx}: 灰度图已转换为BGR三通道")
            elif len(img.shape) == 3:
                if img.shape[2] == 1:
                    # 单通道转BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    logger.info(f"Index {idx}: 单通道图已转换为BGR三通道")
                elif img.shape[2] == 3:
                    # 已经是三通道，保持不变
                    pass
                elif img.shape[2] == 4:
                    # RGBA转BGR，丢弃Alpha通道
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    logger.info(f"Index {idx}: RGBA图已转换为BGR，丢弃Alpha通道")
                else:
                    # 超过4通道，取前3个通道
                    img = img[:, :, :3]
                    logger.warning(f"Index {idx}: 多通道图像({img.shape[2]}通道)已截取前3个通道")
            else:
                raise ValueError(f"Index {idx}: 不支持的图像维度 {img.shape}")

            # 确保最终是三通道BGR格式
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Index {idx}: 通道转换后仍非三通道图像，shape={img.shape}")

            h, w = img.shape[:2]
            orig_shape = (int(h), int(w))  # (H, W)

            tensor_chw, _, d2i = self._algorithm.preprocess_image(
                image_bgr=img,
                network_input_shape=(self.height, self.width),
                norm=norm_params
            )
            processed_tensors.append(tensor_chw)
            d2i_matrices.append(d2i)
            original_shapes.append(orig_shape)

        # 保存 metadata
        self._last_d2i_matrices = d2i_matrices
        self._last_original_shapes = original_shapes

        # 构造批次张量 NCHW
        batch_tensor = np.stack(processed_tensors, axis=0)
        return {
            "images": batch_tensor
        }

    def postprocess(self, raw_outputs: List[Any]) -> List[InferenceOutput]:
        """
        Robust postprocessing with empty-output protection and detailed error logging.
        """
        predictions_batch = raw_outputs[0]
        mask_protos_batch = raw_outputs[1] if len(raw_outputs) > 1 else None
        batch_size = predictions_batch.shape[0]

        results_data: List[Dict[str, Any]] = []
        for i in range(batch_size):
            cur_preds = predictions_batch[i:i + 1, ...]
            cur_protos = mask_protos_batch[i, ...] if mask_protos_batch is not None else None

            detected_boxes_for_image = Algorithm.postprocess_detections(
                predictions=cur_preds,
                num_classes=len(self.class_names),
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                d2i_matrix=self._last_d2i_matrices[i],
                network_input_shape=(self.height, self.width),
                original_image_shape=self._last_original_shapes[i],
                mask_protos=cur_protos,
                mask_coeffs_dim=self.MASK_COEFFS_DIM
            )

            data = self.outdata.process(detected_boxes_for_image)

            results_data.append({
                "file_name": i,
                "detections": data
            })

        return [InferenceOutput(data=results_data)]

    @staticmethod
    def _decode_base64_to_image(b64_str: str, idx: int) -> Optional[np.ndarray]:
        """
        将 Base64 字符串解码为 OpenCV BGR 图像，遇到 imdecode 失败时尝试 PIL 回退。
        """
        b64_str = b64_str.strip()
        try:
            img_bytes = base64.b64decode(b64_str)
        except Exception as e:
            raise ValueError(f"Invalid Base64 string at index {idx}: {e}")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # 回退使用 PIL 解码
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Failed to decode Base64 image at index {idx}: {e}")
                raise ValueError(f"Failed to decode Base64 image at index {idx}: {e}")
        return img