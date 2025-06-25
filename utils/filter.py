from typing import List, Dict, Any
from core.processors.algorithm import Box

class Filter:
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]

    def __init__(self, model_type: str, class_names: Dict[int, str]):
        """
        model_type: "目标识别" / "姿态识别" / "实例分割" / "语义分割"
        class_names: 类别 id → 名称 映射
        """
        self.model_type = model_type
        self.class_names = class_names

    def process(self, boxes: List[Box]) -> Any:
        if self.model_type == "语义分割":
            return self._semantic(boxes)
        else:
            return self._standard(boxes)

    def _standard(self, boxes: List[Box]) -> List[Dict[str, Any]]:
        """目标识别 / 姿态 / 实例分割 统一输出"""
        out = []
        for box in boxes:
            cls = self.class_names.get(box.class_id, str(box.class_id))
            entry: Dict[str, Any] = {
                "box": [box.left, box.top, box.right, box.bottom],
                "conf": float(box.confidence),
                "cls": cls
            }
            if box.masks:
                entry["masks"] = [
                    [int(x), int(y)]
                    for contour in box.masks
                    for x, y in contour
                ]
            if box.points:
                entry["points"] = [
                    [int(x), int(y), int(idx), float(c)]
                    for (x, y, idx, c) in box.points
                ]
                entry["skeleton"] = Filter.SKELETON
            out.append(entry)
        return out

    def _semantic(self, boxes: List[Box]) -> Dict[str, List[List[int]]]:
        """
        语义分割：按类别聚合所有 mask_pixels
        返回：{ cls_name: [[x,y], ...], ... }
        """
        semantic_pixels: Dict[str, List[List[int]]] = {}
        for box in boxes:
            cls_name = self.class_names.get(box.class_id, str(box.class_id))
            if not box.mask_pixels:
                continue
            semantic_pixels.setdefault(cls_name, []).extend(
                [[int(x), int(y)] for x, y in box.mask_pixels]
            )
        return semantic_pixels