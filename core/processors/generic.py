import base64, io
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
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

try:
    import requests
except ImportError:
    logger.warning(
        "WARNING: requests library not found. "
        "HTTP functionality will not be available."
    )
    requests = None


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
        self.input_size = info.get("input_shapes")
        self.class_names = info.get("models_labels", [])
        self.MODEL_TYPE = info.get("models_type")

        # Unpack input dimensions
        # _, _, self.height, self.width = model_config["shapes"]
        try:
            _, _, self.height, self.width = self.input_size
        except Exception:
            raise ValueError(
                f"Invalid input_shapes: {self.input_size}. "
                "Expected a tuple of (batch, channels, height, width)."
            )

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
            img: Optional[np.ndarray] = None
            if isinstance(data, np.ndarray):
                img = data

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

            elif isinstance(data, dict) and ("image_base64" in data or "b64" in data):
                key = "image_base64" if "image_base64" in data else "b64"
                img = self._decode_base64_to_image(data[key], idx)
            elif isinstance(data, str):
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
                num_classes=len(self.class_names),
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                d2i_matrix=d2i_matrices[i],
                network_input_shape=(self.height, self.width),
                original_image_shape=original_shapes[i],
                mask_protos=current_mask_protos,
                mask_coeffs_dim=self.MASK_COEFFS_DIM
            )
            detections: List[Dict[str, Any]] = []
            for box in detected_boxes_for_image:
                cls_name = self.class_names.get(box.class_id, str(box.class_id))
                det = {
                    "file_name": "",
                    "boxes": {
                        "xyxy": list(box.xyxy),
                        "xywh": list(box.xywh),
                        "xyxyn": list(box.xyxyn),
                        "xywhn": list(box.xywhn),
                    },
                    "conf": float(box.confidence),
                    "cls": cls_name,
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
