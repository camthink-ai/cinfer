from enum import Enum, auto
from typing import Dict, List, Any


class ModelOutputDescriber:
    """
    引擎通用工具类：
    """
    # 中文说明文案
    _CN_DESCRIPTIONS: Dict[str, str] = {
        '目标识别': (
            "[目标检测]\n"
            "输出为若干 字典 对象, 每个 字典 包含：\n"
            "  • box: (目标框) \n"
            "     - List[left, top, right, bottom] （边界框坐标，单位像素）"
            "  • conf （置信度，0~1）\n"
            "  • cls （标签）\n"
        ),
        '姿态识别': (
            "[姿态识别]\n"
            "输出为若干 字典 对象, 每个 字典 包含：\n"
            "  • box: (目标框) \n"
            "     - List[left, top, right, bottom] （边界框坐标，单位像素）"
            "  • conf （置信度，0~1）\n"
            "  • cls （标签固定为 'person'）\n"
            "  • points：关键点列表 \n"
            "    – List[List[xi, yi, cls, ci]], xi, yi 为关键点坐标; cls为标签; ci为该关键点置信度;\n"
            "  • skeleton: 姿态关系列表提示信息\n"
            "    – List[List[cls, cls]], cls为关键点标签\n"
            "可通过 skeleton 关系列表在原图上根据 points 绘制每个实例的精确轮廓。"
        ),
        '实例分割': (
            "[实例分割]\n"
            "输出为若干 字典 对象, 每个 字典 包含：\n"
            "  • box: (目标框) \n"
            "     - List[left, top, right, bottom] （边界框坐标，单位像素）"
            "  • conf （置信度，0~1）\n"
            "  • cls （类别 ID）\n"
            "  • masks：分割掩码，可为：\n"
            "     – List[List[x, y]]，轮廓多边形坐标\n"
            "可通过 masks 在原图上绘制每个实例的精确轮廓。"
        ),
        '语义分割': (
            "[语义分割]\n"
            "此模式下输出按类别聚合的像素坐标：\n"
            "  • detections: Dict[str, List[List[int]]]\n"
            "      – 键为类别名称（cls），\n"
            "      – 值为该类别所有像素坐标列表，每个坐标为 [x, y]。\n"
            "直接输出全图级别的语义掩码坐标。"
        ),
    }

    @staticmethod
    def get_description(model_type: str) -> str:
        """
        根据 model_type（英文或中文）返回对应的输出字段说明文本。
        支持的类型：
          - 目标检测
          - 姿态识别
          - 实例分割
          - 语义分割

        :param model_type: 英文或中文的模型类型标识
        :return: 说明文本
        """
        desc = ModelOutputDescriber._CN_DESCRIPTIONS.get(model_type)
        if desc is None:
            supported = "', '".join(ModelOutputDescriber._CN_DESCRIPTIONS.keys())
            return (
                f"未知的模型类型 '{model_type}'。\n"
                f"支持的类型有：'{supported}'。"
            )
        return desc