# Model Samples

For the Chinese description, see the [Chinese version](README_zh.md).

The `models` directory ships with ready-to-import ONNX examples that cover several common computer vision tasks. Each subfolder contains the exported weights together with a `params.yaml`/`params.yml` configuration describing how Cinfer should load and post-process the model.

- `object-detection`: COCO-style, 80-class object detector packaged as `80-target-detection-yolo-11n.onnx`, ideal for broad coverage scenarios. Uses the generic processing pipeline.
- `generic-target-detection`: Lightweight digit detector (`0`–`9`) via `meter_v5i-6.onnx`, useful for meters or screens showing numeric values. Also relies on the generic processing strategy.
- `pose-estimation`: Human pose estimation example (`yolov8n-pose.onnx`) that outputs keypoints and skeletal connections. Uses the generic pipeline while furnishing pose-specific result data.
- `instance-seg`: Pointer-style gauge reader (`yolov8s-seg-20240816.onnx`) leveraging the `pointer` post-processing strategy to combine segmentation masks with pointer angle calculations. Requires additional metadata such as `scale_min` and `scale_max`.

Copy any folder into your persistent model storage (default `/app/data/models`) or register it through the Cinfer backend/API to get started quickly. For a full translation, visit `README_zh.md`.
