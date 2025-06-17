from typing import Dict, Any, List, Tuple

import numpy as np

from core.processors import BaseProcessor, algorithm
from core.processors.algorithm import Algorithm, Norm, ChannelType
from schemas.engine import InferenceInput, InferenceOutput, EngineInfo
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

class generic(BaseProcessor):
    def __init__(self, model_config: Dict[str, Any], engine_info: EngineInfo):
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
        """
        对一批图像进行预处理。
        返回: (批处理张量, [每个图像的d2i矩阵列表], [每个图像的原始形状列表])
        """
        logger.info(f"Preprocessing inputs: {inputs}")
        processed_tensors = []
        d2i_matrices = []
        original_shapes = []
        norm_params = Norm.alpha_beta(alpha=1 / 255.0, channel_type=ChannelType.NONE)

        logger.info(f"Processing input: {inputs[0]}")

        for inp in inputs:
            # 假设 data 是 numpy 格式的 BGR 图像
            img = inp.data
            orig_shape = img.shape[:2]  # (H, W)
            tensor_chw, _, d2i = self._algorithm.preprocess_image(
                image_bgr=img,
                network_input_shape=self.input_size,
                norm=norm_params
            )
            processed_tensors.append(tensor_chw)
            d2i_matrices.append(d2i)
            original_shapes.append(orig_shape)

        # 在返回前，把它们保存在实例属性上
        self._last_d2i_matrices = d2i_matrices
        self._last_original_shapes = original_shapes

        batch_tensor = np.stack(processed_tensors, axis=0)  # (N, C, H, W)

        return {
            "images": batch_tensor,
            "d2i_matrices": d2i_matrices,
            "original_shapes": original_shapes
        }

    def postprocess(self, raw_outputs: List[Any]) -> List[InferenceOutput]:
        # 直接从实例属性取回 metadata
        d2i_matrices = self._last_d2i_matrices
        original_shapes = self._last_original_shapes
        predictions_batch_tensor = raw_outputs[0]
        mask_protos_batch_tensor = None
        if self.MODEL_TYPE.lower() == "v8seg" and len(raw_outputs) > 1:
            mask_protos_batch_tensor = raw_outputs[1]

        batch_size = predictions_batch_tensor.shape[0]

        results: List[InferenceOutput] = []
        for i in range(batch_size):
            # 为当前图像提取预测和原型掩码
            # yolo.postprocess_detections 可能期望输入是 [1, num_preds, ...] 或 [num_preds, ...]
            # 这里假设它期望 [1, num_preds, ...] 以保持批处理维度
            current_predictions = predictions_batch_tensor[i:i + 1, ...]

            current_mask_protos = None
            if mask_protos_batch_tensor is not None:
                current_mask_protos = mask_protos_batch_tensor[i, ...]  # 形状 [num_coeffs, proto_h, proto_w]

            # 对批次中的每个项目进行后处理
            # 注意：这里的 yolo.postprocess_detections 是针对单个图像的输出进行处理
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
