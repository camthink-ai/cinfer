import onnx
import ast
import numpy as np
from typing import Optional, Tuple, List, Any, Union

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
    边界框类，包含边界框的位置信息、置信度、类别ID和可选的分割掩码。
    """
    left: int  # 左上角x坐标
    top: int   # 左上角y坐标
    right: int # 右下角x坐标
    bottom: int # 右下角y坐标
    confidence: float # 置信度
    class_id: int # 类别ID
    masks: Optional[Union[np.ndarray, List[List[Tuple[int, int]]]]]
    mask_pixels: List[List[int]]
    points: List[List]
    def __init__(self, left: int, top: int, right: int, bottom: int,
                 confidence: float, class_id: int, masks: Optional[Union[np.ndarray, List[List[Tuple[int, int]]]]] = None, mask_pixels: List[List[int]] = None,  points: List[List] = None):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.confidence = confidence
        self.class_id = int(class_id)

        self.masks = masks
        self.mask_pixels = mask_pixels
        self.points = points
    def __repr__(self):
        return (f"Box(l={self.left:.1f}, t={self.top:.1f}, r={self.right:.1f}, b={self.bottom:.1f}, "
                f"conf={self.confidence:.2f}, cls={self.class_id}, masks={self.masks is not None}), mask_pixels{self.mask_pixels is not None}, points={self.points is not None})")

class Algorithm:
    @staticmethod
    def get_model_info(path: str) -> tuple[str, dict[Any, Any] | Any, int | Any, int | Any] | tuple[str, dict[Any, Any] | Any, int, int]:
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
    @staticmethod
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
    @staticmethod
    def postprocess_detections(
            predictions: np.ndarray,
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
        后处理原始网络预测，生成 Box 对象列表。支持:
        - 普通目标检测
        - YOLOv8实例分割(带mask_protos)
        - YOLOv8姿态检测(带关键点)
        """
        #新增
        # —— 1. 自适应转置 —— #
        # predictions: (B, M, D) 或 (B, D, M)，M >> D 时说明维度反了
        b, m, d = predictions.shape
        if m < d:
            predictions = predictions.transpose(0, 2, 1)

        batch_preds = predictions[0]  # (N_pred, D)
        total_dims = batch_preds.shape[1]

        # —— 2. 推断模型分支 —— #
        base_no_extra = 4 + num_classes
        extra = total_dims - base_no_extra

        is_seg = mask_protos is not None
        is_normal = (not is_seg) and (extra == 1)
        is_pose = (not is_seg) and (extra > 1 and extra % 3 == 0)
        num_keypoints = extra // 3 if is_pose else 0

        # —— 3. 切出 obj_conf / class_scores / base_dim —— #
        if is_normal:
            # [x,y,w,h, obj_conf, cls1..clsC]
            obj_conf = batch_preds[:, 4]
            class_scores = batch_preds[:, 5:5 + num_classes] * obj_conf[:, None]
            base_dim = 5 + num_classes
        else:
            # [x,y,w,h, cls1..clsC, (mask_coeffs), (keypoints...)]
            obj_conf = None
            class_scores = batch_preds[:, 4:4 + num_classes]
            base_dim = 4 + num_classes

        # —— 4. 筛选候选框 —— #
        max_scores = np.max(class_scores, axis=1)
        candidate_mask = max_scores > conf_threshold
        if not np.any(candidate_mask):
            return []

        cand_preds = batch_preds[candidate_mask]
        cand_scores = class_scores[candidate_mask]
        cand_max_scores = max_scores[candidate_mask]

        # —— 5. 分割系数 & 关键点 —— #
        cand_mask_coeffs = (
            cand_preds[:, base_no_extra:base_no_extra + mask_coeffs_dim]
            if is_seg else None
        )

        if is_pose:
            kpt_raw = cand_preds[:, base_dim:base_dim + 3 * num_keypoints]
            kpt_preds = kpt_raw.reshape(-1, num_keypoints, 3)
        else:
            kpt_preds = np.zeros((0, num_keypoints, 3), dtype=float)

        # —— 6. NMS 准备 —— #
        xywh_net = cand_preds[:, :4]
        cx, cy, w, h = xywh_net.T
        x1_net = cx - w / 2
        y1_net = cy - h / 2
        nms_boxes = np.stack([x1_net, y1_net, w, h], axis=1).tolist()
        raw = cv2.dnn.NMSBoxes(
            nms_boxes,
            cand_max_scores.tolist(),
            conf_threshold,
            nms_threshold
        )
        # 兼容多种返回格式
        if isinstance(raw, tuple): raw = raw[0]
        if hasattr(raw, "ndim") and raw.ndim > 1:
            raw = raw.flatten()
        indices = raw.tolist()

        # —— 7. 构造最终 Box —— #
        final_boxes: List[Box] = []
        net_h, net_w = network_input_shape
        orig_h, orig_w = original_image_shape

        for idx in indices:
            x1n, y1n = x1_net[idx], y1_net[idx]
            x2n, y2n = x1n + w[idx], y1n + h[idx]
            conf = round(float(cand_max_scores[idx]), 2)
            cls_id = int(np.argmax(cand_scores[idx]))

            # 投影回原图并 clip、转 int
            ox1, oy1 = Algorithm.affine_project(d2i_matrix, x1n, y1n)
            ox2, oy2 = Algorithm.affine_project(d2i_matrix, x2n, y2n)
            ox1, oy1 = int(np.clip(ox1, 0, orig_w)), int(np.clip(oy1, 0, orig_h))
            ox2, oy2 = int(np.clip(ox2, 0, orig_w)), int(np.clip(oy2, 0, orig_h))


            # 生成关键点
            kps: List[List[float]] = []
            if is_pose:
                for i_k, (kx, ky, kc) in enumerate(kpt_preds[idx]):
                    px, py = Algorithm.affine_project(d2i_matrix, float(kx), float(ky))
                    px, py = float(np.clip(px, 0, orig_w)), float(np.clip(py, 0, orig_h))
                    kps.append([round(px, 2), round(py, 2), i_k + 1, round(float(kc), 2)])

            # 生成分割 mask 多边形
            masks: List[List[Tuple[int, int]]] = []
            mask_pixels: List[List[int]] = []
            if is_seg and cand_mask_coeffs is not None:
                net_box_w = x2n - x1n
                net_box_h = y2n - y1n

                proto_num, proto_h, proto_w = mask_protos.shape
                scale_x = proto_w / net_w
                scale_y = proto_h / net_h
                mask_w = int(round((x2n - x1n) * scale_x))
                mask_h = int(round((y2n - y1n) * scale_y))
                if mask_w > 0 and mask_h > 0:
                    instance = np.zeros((mask_h, mask_w), dtype=np.float32)
                    for yy in range(mask_h):
                        for xx in range(mask_w):
                            sx = int(round(x1n * scale_x + xx))
                            sy = int(round(y1n * scale_y + yy))
                            if 0 <= sx < proto_w and 0 <= sy < proto_h:
                                proto_vec = mask_protos[:, sy, sx]
                                coeffs = cand_mask_coeffs[idx]
                                instance[yy, xx] = np.dot(coeffs, proto_vec)
                    mask_act = 1.0 / (1.0 + np.exp(-instance))
                    mask_img = cv2.resize(
                        (mask_act * 255).astype(np.uint8),
                        (ox2 - ox1, oy2 - oy1),
                        interpolation=cv2.INTER_LINEAR
                    )
                    _, bin_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

                    # ——— 提取轮廓多边形 ———
                    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        pts = cnt.reshape(-1, 2)
                        masks.append([(int(x + ox1), int(y + oy1)) for x, y in pts])

                    # ——— 提取所有掩码像素点 ———
                    ys, xs = np.where(bin_mask > 0)  # 相对框内坐标
                    for y, x in zip(ys, xs):
                        mask_pixels.append([int(x + ox1), int(y + oy1)])

            final_boxes.append(
                Box(
                    left=ox1, top=oy1,
                    right=(ox2 - ox1), bottom=(oy2 - oy1),
                    confidence=conf, class_id=cls_id,
                    points=kps, masks=masks, mask_pixels=mask_pixels
                )
            )

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

    @staticmethod
    def draw_detections(
            image: np.ndarray,
            boxes: List,
            class_names: Optional[List[str]] = None,
            mask_alpha: float = 0.5,
            draw_contour: bool = True
    ) -> np.ndarray:
        """
        在原图上绘制检测结果，包括：
         - bbox
         - 类别和置信度标签
         - 基于 box.mask_pixels 的语义分割掩码叠加
         - 可选轮廓线
        """
        img = image.copy()
        h_img, w_img = img.shape[:2]

        for box in boxes:
            # 1) 计算并裁剪 bbox 坐标
            x1 = max(0, int(round(box.left)))
            y1 = max(0, int(round(box.top)))
            x2 = min(w_img, int(round(box.left + box.right)))
            y2 = min(h_img, int(round(box.top + box.bottom)))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            # 2) 随机颜色 & 画 bbox
            color = Algorithm.random_color(box.class_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 3) 写类别+置信度标签
            label = (class_names[box.class_id] if class_names and 0 <= box.class_id < len(class_names)
                     else f"Cls {box.class_id}")
            label += f" {box.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - lh - baseline), (x1 + lw, y1), color, cv2.FILLED)
            cv2.putText(img, label, (x1, y1 - baseline // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 4) 叠加语义分割掩码（基于 mask_pixels）
            if hasattr(box, "mask_pixels") and box.mask_pixels:
                w_box, h_box = x2 - x1, y2 - y1
                # 在 ROI 大小上创建二值 mask
                roi_mask = np.zeros((h_box, w_box), dtype=np.uint8)
                for x, y in box.mask_pixels:
                    if x1 <= x < x2 and y1 <= y < y2:
                        roi_mask[y - y1, x - x1] = 255

                # 叠加彩色遮罩
                roi = img[y1:y2, x1:x2].copy()
                overlay = np.zeros_like(roi, dtype=np.uint8)
                overlay[roi_mask > 0] = color
                blended = cv2.addWeighted(roi, 1 - mask_alpha, overlay, mask_alpha, 0)
                img[y1:y2, x1:x2] = blended

                # 可选：绘制轮廓线
                if draw_contour:
                    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img[y1:y2, x1:x2], contours, -1, color, 1)

            if hasattr(box, "masks") and box.masks:
                # 在 ROI 区域画多边形
                sub_img = img[y1:y2, x1:x2]
                for contour in box.masks:
                    # contour 是一系列绝对坐标 [x, y]
                    pts = np.array([[int(x) - x1, int(y) - y1]for x, y in contour], dtype=np.int32)
                    # 画轮廓线
                    cv2.polylines(sub_img, [pts], isClosed=True, color=color, thickness=1)
                img[y1:y2, x1:x2] = sub_img

        return img
