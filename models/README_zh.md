# 模型目录说明

英文说明请参阅 [English version](README.md)。

`models` 目录提供了可直接导入的 ONNX 示例模型，每个子目录都包含模型权重以及用于指导 Cinfer 推理流程的 `params.yaml`/`params.yml` 配置。只需将子目录复制到持久化的模型存储路径（默认 `/app/data/models`），或通过 API/前端界面注册即可使用。

- `object-detection`：COCO 风格的 80 类通用检测模型（`80-target-detection-yolo-11n.onnx`），适合通用场景，使用 `generic` 处理流程。
- `generic-target-detection`：面向数字（0–9）的轻量级检测模型（`meter_v5i-6.onnx`），常用于仪表或数字屏幕识别，同样采用 `generic` 处理策略。
- `pose-estimation`：人体姿态估计示例（`yolov8n-pose.onnx`），输出关键点及骨架信息，沿用通用的 `generic` 处理流程并补充姿态特定的数据。
- `instance-seg`：指针式仪表读数模型（`yolov8s-seg-20240816.onnx`），使用 `pointer` 后处理计算指针角度，需要额外传入 `scale_min`、`scale_max` 等元数据。

可根据需要复制任意模型目录到 `/app/data/models` 等持久化存储位置，或在 Cinfer 后端/API 中注册后快速体验。
