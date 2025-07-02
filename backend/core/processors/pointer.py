import base64, io
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from core.processors import BaseProcessor
from core.processors.algorithm import Algorithm, Norm, ChannelType
from schemas.engine import InferenceInput, InferenceOutput, EngineInfo
from utils.filter import Filter

from core.processors.post_ptr import PointerReader

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


class Pointer(BaseProcessor):
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

        self._reader = PointerReader()

        self._scale_min = 0.0
        self._scale_max = 60.0
        self._img = None
        self.show_image = False

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

            self._scale_min = metadata.get("scale_min")
            self._scale_max = metadata.get("scale_max")
            self._img = metadata.get("img", None)
            self.show_image = metadata.get("show_img", None)

            # img: Optional[np.ndarray] = None
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

    def postprocess(self, raw_outputs: List[Any], show_image: bool = False) -> List[InferenceOutput]:
        """
        Robust postprocessing with empty-output protection and detailed error logging.
        """
        predictions_batch = raw_outputs[0]
        mask_protos_batch = raw_outputs[1] if len(raw_outputs) > 1 else None
        batch_size = predictions_batch.shape[0]

        """优化后的指针读数算法"""
        # scale_ellipses = []
        # pointer_lines = []
        # img = cv2.imread(self._img)

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

            '''
            # Step 1: 分类和预处理轮廓
            for box in detected_boxes_for_image:
                # 合并所有分段为单个轮廓
                all_pts = [np.array(seg, dtype=np.float32).reshape(-1, 2) for seg in box.masks]
                contour = np.vstack(all_pts)

                if box.class_id == 1 and len(contour) >= 5:
                    # 表盘轮廓：拟合椭圆
                    ellipse = cv2.fitEllipse(contour.astype(np.int32).reshape(-1, 1, 2))
                    scale_ellipses.append({'ellipse': ellipse, 'contour': contour})
                elif box.class_id == 0 and len(contour) >= 2:
                    # 指针轮廓：拟合直线
                    line_params = cv2.fitLine(contour.astype(np.int32).reshape(-1, 1, 2),
                                              cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                    pointer_lines.append({'line': line_params, 'contour': contour})

            if not scale_ellipses or not pointer_lines:
                raise RuntimeError("未检测到有效的表盘或指针，无法计算读数")

            # 使用第一对表盘和指针
            scale = scale_ellipses[0]
            pointer = pointer_lines[0]

            # 获取表盘中心和指针尖端 (浮点坐标)
            (cx, cy), (ellipse_a, ellipse_b), ellipse_angle = scale['ellipse']

            # 找指针尖端：离表盘中心最远的点
            pts = pointer['contour']
            dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            tip_idx = np.argmax(dists)
            px, py = pts[tip_idx]

            # 计算刻度范围 (零点和满刻度点)
            dial_contour = scale['contour']
            r_mean = (ellipse_a + ellipse_b) / 4.0  # 平均半径

            # 筛选刻度圈附近的点并计算角度
            dx = dial_contour[:, 0] - cx
            dy = dial_contour[:, 1] - cy
            distances = np.hypot(dx, dy)
            ring_mask = (distances > r_mean * 0.9) & (distances < r_mean * 1.1)

            ring_angles = np.arctan2(dy[ring_mask], dx[ring_mask]) % (2 * np.pi)
            ring_angles_sorted = np.sort(ring_angles)

            # 找最大间隙确定刻度起止点
            angle_diffs = np.diff(ring_angles_sorted)
            wrap_around_diff = ring_angles_sorted[0] + 2 * np.pi - ring_angles_sorted[-1]
            all_diffs = np.append(angle_diffs, wrap_around_diff)

            max_gap_idx = np.argmax(all_diffs)
            angle_zero = ring_angles_sorted[(max_gap_idx + 1) % len(ring_angles_sorted)]
            angle_full = ring_angles_sorted[max_gap_idx]

            # 计算指针角度
            angle_tip = np.arctan2(py - cy, px - cx) % (2 * np.pi)

            # 计算读数
            total_sweep = (angle_full - angle_zero) % (2 * np.pi)
            pointer_offset = (angle_tip - angle_zero) % (2 * np.pi)

            if np.isclose(total_sweep, 0.0, atol=1e-10):
                reading = self._scale_min
                print(reading)
            else:
                fraction = pointer_offset / total_sweep
                reading = self._scale_min + fraction * (self._scale_max - self._scale_min)
                reading = np.clip(reading, self._scale_min, self._scale_max)


            # 可视化绘图
            if img is not None:
                # 转换为整数坐标用于绘图
                cx_int, cy_int = int(round(cx)), int(round(cy))
                px_int, py_int = int(round(px)), int(round(py))

                # 计算零点和满刻度点的像素坐标
                zero_x = cx + r_mean * np.cos(angle_zero)
                zero_y = cy + r_mean * np.sin(angle_zero)
                full_x = cx + r_mean * np.cos(angle_full)
                full_y = cy + r_mean * np.sin(angle_full)

                zero_int = (int(round(zero_x)), int(round(zero_y)))
                full_int = (int(round(full_x)), int(round(full_y)))

                # 绘制椭圆和轮廓
                axes = (int(round(ellipse_a / 2)), int(round(ellipse_b / 2)))
                cv2.ellipse(img, (cx_int, cy_int), axes, ellipse_angle, 0, 360, (0, 255, 0), 2)
                cv2.drawContours(img, [scale['contour'].astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [pointer['contour'].astype(np.int32).reshape(-1, 1, 2)], -1, (255, 0, 0), 1)

                # 标记关键点
                cv2.circle(img, (cx_int, cy_int), 6, (0, 0, 128), -1)  # 中心点 (深红色)
                cv2.circle(img, (px_int, py_int), 6, (0, 0, 255), -1)  # 指针尖端 (红色)
                cv2.circle(img, zero_int, 6, (0, 255, 255), -1)  # 零点 (黄色)
                cv2.circle(img, full_int, 6, (255, 255, 0), -1)  # 满刻度点 (青色)

                # 添加文字标注
                cv2.putText(img, f"Reading: {reading:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"Range: {self._scale_min}-{self._scale_max}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 窗口展示 (可选择是否显示)
                if self.show_image:
                    cv2.namedWindow('Pointer Reading Result', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Pointer Reading Result', 800, 800)
                    cv2.imshow('Pointer Reading Result', img)

                    # 等待按键输入，按任意键关闭窗口
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()

                    # 如果按下 'q' 键，设置标志不再显示后续图像
                    if key == ord('q'):
                        self.show_image = False
                        logger.info("用户按下 'q' 键，已关闭图像显示")
            '''

            readings = self._reader.read_gauge(detected_boxes_for_image, self._scale_min, self._scale_max, self.show_image, self._img)

            results_data.append({
                "file_name": i,
                "detections": [ {"reading": f"{r:.2f}"} for r in readings ]
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
