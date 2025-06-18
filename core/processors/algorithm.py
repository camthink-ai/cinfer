import onnx
import ast
import numpy as np
from typing import Optional, Tuple, List, Any

import logging

logger = logging.getLogger(f"cinfer.{__name__}")

try:
    import cv2
except ImportError:
    logger.warning("WARNING: cv2 (OpenCV) library not found. OpenCV functionality will not be available.")
    cv2 = None

class NormType:
    NONE = 0
    MEAN_STD = 1
    ALPHA_BETA = 2

class ChannelType:
    NONE = 0
    SWAP_RB = 1

class Norm:
    def __init__(self, type=NormType.NONE, alpha=1.0, beta=0.0,
                 mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                 channel_type=ChannelType.NONE):
        self.type = type
        self.alpha = alpha
        self.beta = beta
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.channel_type = channel_type

    @staticmethod
    def mean_std(mean: Tuple[float, float, float],
                 std: Tuple[float, float, float],
                 alpha: float = 1 / 255.0,
                 channel_type: int = ChannelType.NONE) -> 'Norm':
        return Norm(type=NormType.MEAN_STD, alpha=alpha, mean=mean, std=std, channel_type=channel_type)

    @staticmethod
    def alpha_beta(alpha: float, beta: float = 0.0,
                   channel_type: int = ChannelType.NONE) -> 'Norm':
        return Norm(type=NormType.ALPHA_BETA, alpha=alpha, beta=beta, channel_type=channel_type)

    @staticmethod
    def NoneNorm() -> 'Norm':
        return Norm()

class Box:
    """
    保存四种坐标格式 (xyxy, xywh, xyxyn, xywhn)、置信度、类别、可选掩码
    """
    def __init__(
        self,
        xyxy: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        orig_image_size: Tuple[int, int],
        seg: Optional[np.ndarray] = None
    ):
        self.xyxy = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.seg = seg  # 可选的 (H_box, W_box) 二值/灰度掩码

        # 原图尺寸，用于归一化计算
        self.orig_h, self.orig_w = orig_image_size

        # 计算绝对像素下的 xywh (cx, cy, w, h)
        x1, y1, x2, y2 = self.xyxy
        w_pix = x2 - x1
        h_pix = y2 - y1
        cx = x1 + w_pix / 2
        cy = y1 + h_pix / 2
        self.xywh = (cx, cy, w_pix, h_pix)

        # 归一化下的 xyxy
        x1n = x1 / self.orig_w
        y1n = y1 / self.orig_h
        x2n = x2 / self.orig_w
        y2n = y2 / self.orig_h
        self.xyxyn = (x1n, y1n, x2n, y2n)

        # 归一化下的 xywh
        cxn = cx / self.orig_w
        cyn = cy / self.orig_h
        wnorm = w_pix / self.orig_w
        hnorm = h_pix / self.orig_h
        self.xywhn = (cxn, cyn, wnorm, hnorm)

    def __repr__(self):
        return (
            f"Box(xyxy=({self.xyxy[0]:.1f},{self.xyxy[1]:.1f},"
            f"{self.xyxy[2]:.1f},{self.xyxy[3]:.1f}), "
            f"xywh=({self.xywh[0]:.1f},{self.xywh[1]:.1f},"
            f"{self.xywh[2]:.1f},{self.xywh[3]:.1f}), "
            f"xyxyn=({self.xyxyn[0]:.3f},{self.xyxyn[1]:.3f},"
            f"{self.xyxyn[2]:.3f},{self.xyxyn[3]:.3f}), "
            f"xywhn=({self.xywhn[0]:.3f},{self.xywhn[1]:.3f},"
            f"{self.xywhn[2]:.3f},{self.xywhn[3]:.3f}), "
            f"conf={self.confidence:.2f}, cls={self.class_id}, "
            f"has_mask={self.seg is not None})"
        )


class Algorithm:
    @staticmethod
    def get_model_info(path: str) -> tuple[str, dict[Any, Any] | Any, int | Any, int | Any] | tuple[
        str, dict[Any, Any] | Any, int, int]:
        """
        返回模型类型和类别映射：
          - 模型类型 'v8', 'v8seg' 或 'normal'
          - 类别映射字典 {idx: name, ...}
        """
        model = onnx.load(path)
        graph = model.graph
        meta = {p.key: p.value for p in model.metadata_props}

        version = meta.get('version', '')
        task = meta.get('task', 'detect').lower()
        model_type = 'normal'
        if version.startswith('8'):
            model_type = 'v8seg' if task == 'segment' else 'v8'

        names_str = meta.get('names', '{}')
        try:
            labels = ast.literal_eval(names_str)
        except Exception:
            labels = {}

        init_names = {t.name for t in graph.initializer}
        for inp in graph.input:
            if inp.name in init_names:
                continue
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
            # dims = [N, C, H, W]
            if len(dims) >= 4:
                _, _, h, w = dims
                return model_type, labels, h, w
            break

        return model_type, labels, -1, -1

    @staticmethod
    def get_affine_transform_and_inverse(
            from_shape: Tuple[int, int],  # 高, 宽
            to_shape: Tuple[int, int],  # 高, 宽
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算仿射变换矩阵（及其逆矩阵），用于将`from_shape`映射到`to_shape`，
        同时保持宽高比并进行中心化填充。此逻辑镜像了C++中 AffineMatrix::compute 的实现。
        """
        from_h, from_w = from_shape
        to_h, to_w = to_shape

        scale_x = to_w / float(from_w)
        scale_y = to_h / float(from_h)
        scale = min(scale_x, scale_y)

        # i2d: 图像到目标（网络输入）
        i2d = np.zeros((2, 3), dtype=np.float32)
        i2d[0, 0] = scale
        i2d[0, 1] = 0
        i2d[0, 2] = -scale * from_w * 0.5 + to_w * 0.5 + scale * 0.5 - 0.5
        i2d[1, 0] = 0
        i2d[1, 1] = scale
        i2d[1, 2] = -scale * from_h * 0.5 + to_h * 0.5 + scale * 0.5 - 0.5

        # d2i: 目标（网络输入）到图像
        # 计算 i2d 的逆矩阵
        # 将 i2d 扩展为3x3矩阵以便求逆
        i2d_aug = np.vstack([i2d, [0, 0, 1]])
        d2i_aug = np.linalg.inv(i2d_aug)
        d2i = d2i_aug[:2, :]

        return i2d, d2i

    @staticmethod
    def affine_project(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """将2x3仿射矩阵应用于一个点。"""
        ox = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        oy = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        return ox, oy

    # --- 图像预处理 ---
    def preprocess_image(
            image_bgr: np.ndarray,  # 输入图像，BGR格式 (高, 宽, 通道)
            network_input_shape: Tuple[int, int],  # (高, 宽)
            norm: Norm,
            border_color: Tuple[int, int, int] = (114, 114, 114)  # BGR格式
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        为YOLO推理预处理图像。
        返回:
            - preprocessed_image_chw: (通道, 高, 宽) float32 NumPy数组
            - i2d_matrix: 从原始图像到网络输入的仿射变换矩阵
            - d2i_matrix: 从网络输入到原始图像的仿射变换矩阵
        """
        src_h, src_w = image_bgr.shape[:2]
        net_h, net_w = network_input_shape

        i2d_matrix, d2i_matrix = Algorithm.get_affine_transform_and_inverse((src_h, src_w), network_input_shape)

        # 应用仿射变换
        warped_image = cv2.warpAffine(
            image_bgr, i2d_matrix, (net_w, net_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color
        )

        # 通道交换 (例如，BGR转RGB或RGB转BGR)
        processed_image = warped_image
        if norm.channel_type == ChannelType.SWAP_RB:
            processed_image = processed_image[..., ::-1]

        # 归一化
        processed_image = processed_image.astype(np.float32)
        if norm.type == NormType.ALPHA_BETA:
            processed_image = processed_image * norm.alpha + norm.beta
        elif norm.type == NormType.MEAN_STD:

            processed_image = (processed_image * norm.alpha - norm.mean) / norm.std

        # 从HWC格式转置为CHW格式
        preprocessed_image_chw = processed_image.transpose(2, 0, 1)
        preprocessed_image_chw = np.ascontiguousarray(preprocessed_image_chw)

        return preprocessed_image_chw, i2d_matrix, d2i_matrix

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # 后处理检测
    def postprocess_detections(
            predictions: np.ndarray,
            model_type: str,
            num_classes: int,
            conf_threshold: float,
            nms_threshold: float,
            d2i_matrix: np.ndarray,
            network_input_shape: Tuple[int, int],
            original_image_shape: Tuple[int, int],
            mask_protos: Optional[np.ndarray] = None,
            mask_coeffs_dim: int = 32
    ) -> List[Box]:
        """
        后处理原始网络预测，生成 Box 对象列表。Box 内部自动算出 xywh/xyxyn/xywhn 四种坐标格式，
        并把可选分割掩码存到 seg 字段。
        """

        # ——— 1. YOLOv8 需要先转置 维度 ———
        if model_type.lower() in ["v8", "v8seg"]:
            predictions = predictions.transpose(0, 2, 1)

        batch_preds = predictions[0]  # shape (N_pred, 4+num_classes(+mask_coeff_dim))

        # ——— 2. 计算 class_scores 并筛选置信度高的候选 ——
        if model_type.lower() in ["normal"]:
            # 格式通常是 [cx, cy, w, h, obj_conf, cls1_score, cls2_score, ...]
            obj_conf = batch_preds[:, 4]  # shape (N_pred,)
            class_scores = batch_preds[:, 5: 5 + num_classes] * obj_conf[:, None]
        elif model_type.lower() in ["v8", "v8seg"]:
            # 格式是 [cx, cy, w, h, cls1_score, cls2_score, ... (, mask_coeffs...)]
            class_scores = batch_preds[:, 4: 4 + num_classes]
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        max_scores = np.max(class_scores, axis=1)  # shape (N_pred,)
        candidate_mask = max_scores > conf_threshold

        if not np.any(candidate_mask):
            return []  # 没有大于阈值的框

        cand_preds = batch_preds[candidate_mask]  # shape (N_cand, ...)
        cand_scores = class_scores[candidate_mask]  # shape (N_cand, num_classes)
        cand_max_scores = max_scores[candidate_mask]  # shape (N_cand,)

        # 如果是 v8seg，需要后面解码掩码
        if model_type.lower() == "v8seg":
            # 每行的掩码系数部分：
            # cand_preds[:, 4+num_classes : 4+num_classes+mask_coeffs_dim]
            cand_mask_coeffs = cand_preds[:, 4 + num_classes: 4 + num_classes + mask_coeffs_dim]
        else:
            cand_mask_coeffs = None

        # ——— 3. 从候选预测里取出 network-space 下的 (cx, cy, w, h) ——
        xywh_net = cand_preds[:, :4]  # shape (N_cand, 4)
        cx_net = xywh_net[:, 0]
        cy_net = xywh_net[:, 1]
        w_net = xywh_net[:, 2]
        h_net = xywh_net[:, 3]

        # 计算网络空间下的 (x1, y1, x2, y2)
        x1_net = cx_net - w_net / 2
        y1_net = cy_net - h_net / 2
        x2_net = cx_net + w_net / 2
        y2_net = cy_net + h_net / 2

        nms_boxes_xywh_net = np.stack([x1_net, y1_net, w_net, h_net], axis=1)  # shape (N_cand, 4)
        nms_conf_list = cand_max_scores.tolist()

        if len(nms_boxes_xywh_net) > 0:
            indices = cv2.dnn.NMSBoxes(
                nms_boxes_xywh_net.tolist(),  # list of [x1, y1, w, h]
                nms_conf_list,  # list of confidences
                conf_threshold,
                nms_threshold
            )

            if isinstance(indices, tuple):
                indices = indices[0]
            if hasattr(indices, "ndim") and indices.ndim > 1:
                indices = indices.flatten()
            indices = indices.tolist()
        else:
            indices = []


        final_boxes: List[Box] = []
        if len(indices) > 0:
            net_h, net_w = network_input_shape
            orig_h, orig_w = original_image_shape

            for idx in indices:
                # 网络空间下的坐标
                x1n = x1_net[idx]
                y1n = y1_net[idx]
                x2n = x2_net[idx]
                y2n = y2_net[idx]
                conf = float(cand_max_scores[idx])
                cls_id = int(np.argmax(cand_scores[idx]))

                orig_x1, orig_y1 = Algorithm.affine_project(d2i_matrix, x1n, y1n)
                orig_x2, orig_y2 = Algorithm.affine_project(d2i_matrix, x2n, y2n)

                orig_x1 = float(np.clip(orig_x1, 0, orig_w))
                orig_y1 = float(np.clip(orig_y1, 0, orig_h))
                orig_x2 = float(np.clip(orig_x2, 0, orig_w))
                orig_y2 = float(np.clip(orig_y2, 0, orig_h))

                final_mask_array = None
                if model_type.lower() == "v8seg" and mask_protos is not None:
                    net_box_w = x2n - x1n
                    net_box_h = y2n - y1n

                    proto_num_coeffs, proto_h, proto_w = mask_protos.shape
                    net_h_input, net_w_input = network_input_shape

                    scale_to_proto_x = proto_w / net_w_input
                    scale_to_proto_y = proto_h / net_h_input

                    mask_target_w_proto = int(round(net_box_w * scale_to_proto_x))
                    mask_target_h_proto = int(round(net_box_h * scale_to_proto_y))

                    if mask_target_w_proto > 0 and mask_target_h_proto > 0:
                        box_top_left_x_proto = x1n * scale_to_proto_x
                        box_top_left_y_proto = y1n * scale_to_proto_y

                        instance_mask_proto = np.zeros((mask_target_h_proto, mask_target_w_proto), dtype=np.float32)

                        for dy in range(mask_target_h_proto):
                            for dx in range(mask_target_w_proto):
                                sx = int(round(box_top_left_x_proto + dx))
                                sy = int(round(box_top_left_y_proto + dy))
                                if 0 <= sx < proto_w and 0 <= sy < proto_h:
                                    proto_vector = mask_protos[:, sy, sx]  # shape (proto_num_coeffs,)
                                    coeffs = cand_mask_coeffs[idx]  # shape (proto_num_coeffs,)
                                    instance_mask_proto[dy, dx] = np.dot(coeffs, proto_vector)
                                else:
                                    instance_mask_proto[dy, dx] = -np.inf

                        instance_mask_activated = 1.0 / (1.0 + np.exp(-instance_mask_proto))

                        orig_w_box = int(round(orig_x2 - orig_x1))
                        orig_h_box = int(round(orig_y2 - orig_y1))
                        if orig_w_box > 0 and orig_h_box > 0:
                            final_mask_array = cv2.resize(
                                instance_mask_activated,
                                (orig_w_box, orig_h_box),
                                interpolation=cv2.INTER_LINEAR
                            )
                            final_mask_array = (final_mask_array * 255).astype(np.uint8)
                        else:
                            final_mask_array = np.zeros((0, 0), dtype=np.uint8)

                    else:
                        final_mask_array = np.zeros((0, 0), dtype=np.uint8)

                box_obj = Box(
                    xyxy=(orig_x1, orig_y1, orig_x2, orig_y2),
                    confidence=conf,
                    class_id=cls_id,
                    orig_image_size=(orig_h, orig_w),
                    seg=final_mask_array
                )
                final_boxes.append(box_obj)

        return final_boxes

    @staticmethod
    def hsv2bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = 0.0, 0.0, 0.0
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        elif h_i == 5:
            r, g, b = v, p, q
        else:
            r, g, b = 1.0, 1.0, 1.0

        return int(b * 255), int(g * 255), int(r * 255)

    @staticmethod
    def random_color(obj_id: int) -> Tuple[int, int, int]:
        """
        给定一个类别 ID，返回一个“随机但固定”的 BGR 颜色。
        1. 先把 obj_id 转成整数（如果它本身是字符串数字的话）。
        2. 通过位移、异或、取模得到 HSV 的 h_plane、s_plane（都在 0.00~0.99 之间）。
        3. 亮度 v_plane 固定为 1.0。
        4. 最后把 (h_plane,s_plane,1.0) 转成 BGR 三通道并返回。
        """
        # 如果传入的是字符串，把它先转成 int；否则保持原样
        if isinstance(obj_id, str):
            try:
                obj_id = int(obj_id)
            except ValueError:
                raise ValueError(f"无法把类别 ID '{obj_id}' 转成整数，请确认 class_id 是数字形式的字符串或整数。")

        # 如果传入的还是非整数（比如 float），也直接转换为 int
        if not isinstance(obj_id, int):
            obj_id = int(obj_id)

        # 下面的位运算只能对整数执行
        h_plane = (((obj_id << 2) ^ 0x937151) % 100) / 100.0
        s_plane = (((obj_id << 3) ^ 0x315793) % 100) / 100.0  # 饱和度
        # 如果想让饱和度更高，可以改成下面这一行：
        # s_plane = ((((obj_id << 3) ^ 0x315793) % 50) + 50) / 100.0

        return Algorithm.hsv2bgr(h_plane, s_plane, 1.0)

    def draw_detections(image: np.ndarray,
                        boxes: list,
                        class_names: Optional[list] = None,
                        mask_alpha: float = 0.5,
                        draw_contour: bool = True):
        img = image.copy()
        h_img, w_img = img.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            color = Algorithm.random_color(box.class_id)
            # 1) 画 bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 2) 写 label
            label = class_names[box.class_id] if class_names and 0 <= box.class_id < len(
                class_names) else f"Cls {box.class_id}"
            label += f" {box.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - lh - baseline), (x1 + lw, y1), color, cv2.FILLED)
            cv2.putText(img, label, (x1, y1 - baseline // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 3) 叠加 mask
            if box.seg is not None and box.seg.size > 0:
                box_w, box_h = x2 - x1, y2 - y1
                seg = cv2.resize((box.seg > 128).astype(np.uint8) * 255,
                                 (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                roi = img[y1:y2, x1:x2].copy()
                overlay = np.zeros_like(roi, dtype=np.uint8)
                overlay[seg > 128] = color
                blended = cv2.addWeighted(roi, 1 - mask_alpha, overlay, mask_alpha, 0)
                img[y1:y2, x1:x2] = blended

                if draw_contour:
                    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img[y1:y2, x1:x2], contours, -1, color, 1)

        return img
