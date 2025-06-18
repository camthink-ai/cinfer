import base64, io
from typing import Dict, Any, List, Tuple, Optional


import numpy as np
import requests

from core.processors import BaseProcessor
from core.processors.algorithm import Algorithm, Norm, ChannelType
from schemas.engine import InferenceInput, InferenceOutput, EngineInfo

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


class generic(BaseProcessor):
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

        super().__init__(model_config, engine_info)

        self._algorithm = Algorithm

        self.input_size = model_config.get("input_size", (640, 640))  # 默认输入尺寸
        self.conf_threshold = model_config.get("conf_threshold", 0.25)
        self.nms_threshold = model_config.get("nms_threshold", 0.45)
        self.class_names = model_config.get("class_names", [])  # 可选类别名称列表

        self.MODEL_TYPE = "normal"
        self.MODEL_LABELS = model_config.get("MODEL_LABELS", 10)
        self.MASK_COEFFS_DIM = 32  # YOLOv8-Seg模型使用
        self._last_d2i_matrices: List[np.ndarray] = []
        self._last_original_shapes: List[Tuple[int, int]] = []

        self._last_file_names: List[str] = []

        logger.info(f"engine_info: {engine_info}")
        logger.info(f"input_size: {self.input_size}")
        logger.info(f"conf_threshold: {self.conf_threshold}")
        logger.info(f"nms_threshold: {self.nms_threshold}")
        logger.info(f"class_names: {self.class_names}")

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
            img: Optional[np.ndarray] = None

            # 1) numpy 数组
            if isinstance(data, np.ndarray):
                img = data

            # 2) URL 输入（dict 或 字符串）
            elif isinstance(data, dict) and "url" in data:
                resp = requests.get(data["url"])
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif isinstance(data, str) and data.lower().startswith("http"):
                resp = requests.get(data)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            # 3) Base64 输入（dict 或 裸字符串 或 data URL）
            elif isinstance(data, dict) and ("image_base64" in data or "b64" in data):
                key = "image_base64" if "image_base64" in data else "b64"
                img = self._decode_base64_to_image(data[key], idx)
            elif isinstance(data, str):
                # 支持 data URL 前缀
                if data.startswith("data:") and "," in data:
                    _, b64_str = data.split(",", 1)
                else:
                    b64_str = data
                img = self._decode_base64_to_image(b64_str, idx)

            else:
                raise ValueError(f"Unsupported input data format at index {idx}: {type(data)}")

            if img is None:
                raise ValueError(f"Failed to decode image for input index {idx}")

            # 记录原始大小
            h, w = img.shape[:2]
            orig_shape = (int(h), int(w))  # (H, W)

            tensor_chw, _, d2i = self._algorithm.preprocess_image(
                image_bgr=img,
                network_input_shape=self.input_size,
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
            "images": batch_tensor,
            "d2i_matrices": d2i_matrices,
            "original_shapes": original_shapes
        }

    def postprocess(self, raw_outputs: List[Any]) -> List[InferenceOutput]:
        d2i_matrices = self._last_d2i_matrices
        original_shapes = self._last_original_shapes
        predictions_batch_tensor = raw_outputs[0]
        mask_protos_batch_tensor = None
        if self.MODEL_TYPE.lower() == "v8seg" and len(raw_outputs) > 1:
            mask_protos_batch_tensor = raw_outputs[1]

        batch_size = predictions_batch_tensor.shape[0]

        results: List[InferenceOutput] = []
        for i in range(batch_size):
            current_predictions = predictions_batch_tensor[i:i + 1, ...]

            current_mask_protos = None
            if mask_protos_batch_tensor is not None:
                current_mask_protos = mask_protos_batch_tensor[i, ...]  # 形状 [num_coeffs, proto_h, proto_w]

            detected_boxes_for_image = Algorithm.postprocess_detections(
                predictions=current_predictions,
                model_type=self.MODEL_TYPE,
                num_classes=self.MODEL_LABELS,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                d2i_matrix=d2i_matrices[i],
                network_input_shape=self.input_size,
                original_image_shape=original_shapes[i],
                mask_protos=current_mask_protos,
                mask_coeffs_dim=self.MASK_COEFFS_DIM
            )
            detections: List[Dict[str, Any]] = []
            for box in detected_boxes_for_image:
                det = {
                    "file_name": "",
                    "boxes": {
                        "xyxy": list(box.xyxy),
                        "xywh": list(box.xywh),
                        "xyxyn": list(box.xyxyn),
                        "xywhn": list(box.xywhn),
                    },
                    "conf": float(box.confidence),
                    "cls": str(box.class_id),
                    "masks": []
                }

                if getattr(box, "mask", None) is not None:
                    det["masks"] = box.seg.astype(int).tolist()
                detections.append(det)

            results.append(
                InferenceOutput(
                    data=detections,
                )
            )
        return results

    def _decode_base64_to_image(self, b64_str: str, idx: int) -> Optional[np.ndarray]:
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
