from typing import List, Any, Optional

import cv2
import numpy as np


class PointerReader:
    """
    指针仪表读数识别器
    """

    def __init__(self):
        pass

    def read_gauge(self, boxes: List[Any], scale_min: float, scale_max: float,
                   show_image: bool = False, img_path: Optional[str] = None) -> List[float]:
        """
        指针仪表读数核心算法（多表返回多读数）
        Returns:
            List[float]: 每个表的读数列表
        """
        # Step 1: 分类和预处理轮廓
        scale_ellipses = []
        pointer_lines = []

        for box in boxes:
            all_pts = [np.array(seg, dtype=np.float32).reshape(-1, 2) for seg in box.masks]
            contour = np.vstack(all_pts)

            if box.class_id == 1 and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour.astype(np.int32).reshape(-1, 1, 2))
                scale_ellipses.append({'ellipse': ellipse, 'contour': contour})
            elif box.class_id == 0 and len(contour) >= 2:
                line_params = cv2.fitLine(contour.astype(np.int32).reshape(-1, 1, 2),
                                          cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                pointer_lines.append({'line': line_params, 'contour': contour})

        if not scale_ellipses or not pointer_lines:
            raise RuntimeError("未检测到有效的表盘或指针，无法计算读数")

        readings: List[float] = []

        # Step 2: 对每一对 scale/pointer 进行计算
        # 这里简单地 zip，如果你有更复杂的匹配逻辑可以替换
        for scale, pointer in zip(scale_ellipses, pointer_lines):
            # 表盘中心 & 椭圆参数
            (cx, cy), (ellipse_a, ellipse_b), ellipse_angle = scale['ellipse']

            # 指针尖端：离中心最远的点
            pts = pointer['contour']
            dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            tip_idx = np.argmax(dists)
            px, py = pts[tip_idx]

            # 找刻度环上的点
            dial_contour = scale['contour']
            r_mean = (ellipse_a + ellipse_b) / 4.0
            dx = dial_contour[:, 0] - cx
            dy = dial_contour[:, 1] - cy
            distances = np.hypot(dx, dy)
            mask_ring = (distances > r_mean * 0.9) & (distances < r_mean * 1.1)

            angles = np.arctan2(dy[mask_ring], dx[mask_ring]) % (2 * np.pi)
            angles_sorted = np.sort(angles)
            diffs = np.diff(angles_sorted)
            wrap = angles_sorted[0] + 2 * np.pi - angles_sorted[-1]
            all_diffs = np.append(diffs, wrap)
            idx_gap = np.argmax(all_diffs)
            angle_zero = angles_sorted[(idx_gap + 1) % len(angles_sorted)]
            angle_full = angles_sorted[idx_gap]

            # 指针角度
            angle_tip = np.arctan2(py - cy, px - cx) % (2 * np.pi)
            total = (angle_full - angle_zero) % (2 * np.pi)
            offset = (angle_tip - angle_zero) % (2 * np.pi)

            if np.isclose(total, 0.0, atol=1e-10):
                reading = scale_min
            else:
                frac = offset / total
                reading = scale_min + frac * (scale_max - scale_min)
                reading = float(np.clip(reading, scale_min, scale_max))

            readings.append(reading)

            # 可视化每个表的读数和关键点
            if show_image and img_path:
                self._visualize_result(
                    img_path, cx, cy, px, py, ellipse_a, ellipse_b, ellipse_angle,
                    r_mean, angle_zero, angle_full, reading, scale_min, scale_max,
                    scale, pointer, show_image
                )

        return readings

    def _visualize_result(self, img_path: str, cx: float, cy: float, px: float, py: float,
                          ellipse_a: float, ellipse_b: float, ellipse_angle: float,
                          r_mean: float, angle_zero: float, angle_full: float,
                          reading: float, scale_min: float, scale_max: float,
                          scale: dict, pointer: dict, show_image: bool) -> None:
        """
        可视化读数结果
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法加载图像文件 {img_path}")
                return

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
            cv2.putText(img, f"Range: {scale_min}-{scale_max}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 窗口展示
            if show_image:
                cv2.namedWindow('Pointer Reading Result', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Pointer Reading Result', 800, 800)
                cv2.imshow('Pointer Reading Result', img)

                # 等待按键输入，按任意键关闭窗口
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()

                if key == ord('q'):
                    print("用户按下 'q' 键，关闭图像显示")

        except Exception as e:
            print(f"可视化过程中发生错误: {str(e)}")