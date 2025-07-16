import base64
import io
import time
from typing import Any, Dict, List, Optional

import cv2
import logging

from .base import AsyncEngine, EngineInfo, InferenceInput, InferenceOutput, InferenceResult, ResourceRequirements
from PIL import Image
import requests
import numpy as np

logger = logging.getLogger(f"cinfer.{__name__}")

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.warning("WARNING: PaddleOCR library not found. OCREngine will not be available.")

try:
    import paddle
except ImportError:
    logger.warning(
        "WARNING: paddle library not found. OCREngine will not be available.")


class OCREngine(AsyncEngine):
    ENGINE_NAME = "OCREngine"

    def __init__(self, max_workers: Optional[int] = None, queue_size: Optional[int] = None):
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR library is not installed. "
                "Please install it to use OCREngine (e.g., 'pip install paddleocr==3.0.0')."
            )

        if paddle is None:
            raise RuntimeError(
                "PaddlePaddle library is not installed. "
                "Please install it to enable OCREngine (e.g., 'python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/')."
            )

        super().__init__(max_workers=max_workers, queue_size=queue_size)

        self._session = None
        self._model_loaded = False
        self.device = ""

    def _initialize_paddle_runtime(self) -> bool:
        """initialize paddle runtime"""
        try:
            # set device
            if paddle.is_compiled_with_cuda():
                paddle.set_device('gpu')
                logger.info("use GPU for OCR inference")
                self.device = 'gpu'
            else:
                paddle.set_device('cpu')
                logger.info("use CPU for OCR inference")
                self.device = 'cpu'
            return True

        except Exception as e:
            logger.error(f"initialize failed: {e}")
            self._model_loaded = False
            return False

    def initialize(self, engine_config: Dict[str, Any]) -> bool:
        super_init_ok = super().initialize(engine_config)
        if not super_init_ok:
            return False
        return self._initialize_paddle_runtime()

    def _load_model_specifico(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """load OCR model and identify detection and recognition model paths based on keywords"""
        import os

        try:
            logger.info(f"scanning model path: {model_path}")

            # check if the path exists
            if not os.path.exists(model_path):
                raise ValueError(f"model path not found: {model_path}")

            # get all folders in the path
            folders = []
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path):
                    folders.append((item, item_path))

            if not folders:
                raise ValueError(f"no folders found in path: {model_path}")

            logger.info(f"found {len(folders)} folders: {[name for name, _ in folders]}")

            # identify model path based on keywords, prioritize server-level model
            det_model_path = None
            rec_model_path = None
            cls_model_path = None

            for folder_name, folder_path in folders:
                folder_lower = folder_name.lower()

                # detection model path recognition
                if 'det' in folder_lower:
                    det_model_path = folder_path
                    logger.info(f"detected detection model: {folder_name} -> {folder_path}")
                # recognition model path recognition
                elif 'rec' in folder_lower:
                    rec_model_path = folder_path
                    logger.info(f"detected recognition model: {folder_name} -> {folder_path}")
                # classification model path recognition (optional)
                elif 'cls' in folder_lower or 'angle' in folder_lower:
                    cls_model_path = folder_path
                    logger.info(f"detected classification model: {folder_name} -> {folder_path}")

            # verify if necessary models are found
            if det_model_path is None:
                logger.warning("no detection model found with 'det' keyword")
            if rec_model_path is None:
                logger.warning("no recognition model found with 'rec' keyword")

            self._session = PaddleOCR(
                text_detection_model_dir=det_model_path,
                text_recognition_model_dir=rec_model_path,
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            logger.info(f"PaddleOCR model loaded successfully")
            self._model_loaded = True
            return True

        except Exception as e:
            logger.error(f"model loading failed: {e}", exc_info=True)
            self._model_loaded = False
            return False

    def _preprocess_input(self, raw_inputs: List[InferenceInput]) -> Dict[str, List[np.ndarray]]:
        """
        OCR preprocess: extract image data from InferenceInput and convert to image array
        only support the following input formats:
        - dict contains 'url' key
        - dict contains 'image_base64' or 'b64' key
        - direct Base64 string
        - direct URL string
        """
        images: List[np.ndarray] = []

        for idx, inp in enumerate(raw_inputs):
            data = inp.data
            img = None

            try:
                logger.info(f" [index {idx}] processing input data type: {type(data)}")
                logger.info(f" [index {idx}] data content preview: {str(data)[:100]}...")

                # handle URL (dict format)
                if isinstance(data, dict) and "url" in data:
                    url = data["url"]
                    logger.info(f" [index {idx}] detected dict format URL: {url}")
                    img = self._download_image_as_array(url, idx)

                # handle URL (direct string)
                elif isinstance(data, str) and data.lower().startswith("http"):
                    logger.info(f" [index {idx}] detected string format URL: {data}")
                    img = self._download_image_as_array(data, idx)

                # handle Base64 (dict format)
                elif isinstance(data, dict) and ("image_base64" in data or "b64" in data):
                    key = "image_base64" if "image_base64" in data else "b64"
                    logger.info(f" [index {idx}] detected dict format Base64, key: {key}")
                    img = self._decode_base64_to_image(data[key], idx)

                # handle Base64 (direct string)
                elif isinstance(data, str):
                    logger.info(f" [index {idx}] detected string format, judged as Base64")
                    b64_str = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
                    img = self._decode_base64_to_image(b64_str, idx)

                else:
                    raise ValueError(f"unsupported input data format at index {idx}: {type(data)}. only support URL and Base64 format")

                # normalize image format
                if img is not None:
                    img_normalized = self._normalize_image_channels(img, idx)
                    images.append(img_normalized)
                    logger.info(f" [index {idx}] successfully processed, image size: {img_normalized.shape}")
                else:
                    raise ValueError(f"failed to process input {idx}, unable to generate valid image")

            except Exception as e:
                msg = f"failed to preprocess at index {idx}, input={type(data)}, error={e}"
                logger.error(msg, exc_info=True)
                raise ValueError(msg)

        if not images:
            raise ValueError("no valid image arrays generated")

        logger.info(f"🎉 preprocess completed: {len(images)} image arrays")
        return {"images": images}

    def _postprocess_output(self, raw_outputs: List[Any]) -> List[InferenceOutput]:
        """
        OCR postprocess: convert the original output of PaddleOCR to the standard InferenceOutput format

        Args:
            raw_outputs: the original output of PaddleOCR, which may contain multiple formats of data

        Returns:
            List[InferenceOutput]: the standardized inference output list
        """
        batch_results = []

        # Iterate through the results for each image in the batch
        for image_idx, image_results in enumerate(raw_outputs):

            image_output = {
                "image_id": image_idx,
                "detections": []
            }

            # The result for a single image might be wrapped in an extra list, handle both cases
            if image_results and isinstance(image_results, list) and len(image_results) > 0:
                # If there are detections for this image
                detections_list = image_results[0] if isinstance(image_results[0], list) and isinstance(image_results[0][0], list) else image_results

                for detection_line in detections_list:
                    if detection_line is None:
                        continue

                    # Extract box, text, and confidence from the tuple
                    box, (text, confidence) = detection_line

                    # box is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Convert polygon to a bounding box [x_min, y_min, width, height]
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]

                    left = int(min(x_coords))
                    top = int(min(y_coords))
                    right = int(max(x_coords))
                    bottom = int(max(y_coords))

                    bbox = [left, top, right - left, bottom - top]

                    detection_dict = {
                        "box": bbox,
                        "conf": round(float(confidence), 4),
                        "cls": text
                    }
                    image_output["detections"].append(detection_dict)

            batch_results.append(image_output)

        # Wrap the final list of results for all images into a single InferenceOutput
        return [InferenceOutput(data=batch_results)]

    def _batch_process(self, inputs_batch: List[Dict[str, np.ndarray]]) -> List[List[np.ndarray]]:
        """
        batch OCR inference

        Args:
            inputs_batch: batch input data, each element contains image data

        Returns:
            List[Any]: batch OCR inference results
        """
        if not self._session:
            raise RuntimeError("Paddle OCR session not initialized.")

        results_batch = []
        # iterate through all requests (usually only one)
        for single_input_dict in inputs_batch:
            # get the list of all images in this request
            images_list = single_input_dict.get("images")
            if images_list is None or not isinstance(images_list, list):
                raise ValueError("Input data must contain a list of images under the 'images' key.")

            results_for_this_request = []
            logger.info(f"Processing a batch of {len(images_list)} images individually...")

            # --- core modification: iterate through the image list, perform OCR on each image individually ---
            for image_array in images_list:
                # call ocr method for a single image
                # try catch for PaddleOCR 2.7.3 in x86 CPU: could not execute a primitive
                try:
                    ocr_result = self._session.ocr(image_array, cls=False)
                except Exception as e:
                    logger.warning(f"Error during OCR processing: {e}")
                    load_success = self._load_model_specifico(self._loaded_model_path, self._model_config)
                    if not load_success:
                        logger.error(f"Failed to reload OCR model: {e}")
                        raise RuntimeError("Failed to reload OCR model.")
                    ocr_result = self._session.ocr(image_array, cls=False)
                results_for_this_request.append(ocr_result)

            # add the results of all images in this request as a whole to the final batch processing results
            results_batch.append(results_for_this_request)

        return results_batch

    def predict(self, inputs: List[InferenceInput]) -> InferenceResult:
        if not self._model_loaded or not self._session:
            return InferenceResult(success=False, error_message="Paddle model not loaded.")

        start_time_sec = time.time()  # Corrected

        try:
            # preprocess
            if self._processor:
                logger.info(f"Processor: {self._processor}")
                preprocessed_data_dict = self._processor.preprocess(inputs)
            else:
                logger.info(f"No processor found. Using default preprocess_input.")
                preprocessed_data_dict = self._preprocess_input(inputs)

            logger.info(f"Preprocessed data keys: {list(preprocessed_data_dict.keys())}")

            # batch processing inference
            raw_outputs_list = self._batch_process([preprocessed_data_dict])
            raw_outputs_for_this_call = raw_outputs_list[0]

            # postprocess
            if self._processor:
                final_outputs = self._processor.postprocess(raw_outputs_for_this_call)
            else:
                final_outputs = self._postprocess_output(raw_outputs_for_this_call)

            logger.info(f"Final outputs count: {len(final_outputs)}")
            processing_time_ms = (time.time() - start_time_sec) * 1000

            return InferenceResult(success=True, outputs=final_outputs,processing_time_ms=processing_time_ms)
        except Exception as e:
            logger.error(f"Error during Paddl prediction: {e}")
            return InferenceResult(success=False, error_message=str(e),
                                   processing_time_ms=(time.time() - start_time_sec) * 1000)  # Corrected

    def get_info(self) -> EngineInfo:
        """return OCR engine information"""
        engine_info = EngineInfo(
            engine_name=self.ENGINE_NAME,
            engine_version=paddle.__version__ if paddle else "N/A",
            model_loaded=self._model_loaded,
            loaded_model_path=None,
            available_devices=[],
            current_device=None,
            engine_status="",
            additional_info={
                "session_providers": [],
                "engine_config_providers": "",

                "input_names": {
                  "name": "input_name",
                  "type": "tensor",
                  "shape": [1, 3, 224, 224]
                },
                "output_names": {
                  "name": "output_name",
                  "type": "tensor",
                  "shape": [1, 1000]
                },
                "input_shapes": [],
                "models_type": [],
                "models_labels": [],
                "algorithm": [],

                "desc": []
            }
        )
        logger.info(f"EngineInfo: {engine_info}")
        return engine_info

    def get_resource_requirements(self) -> ResourceRequirements:
        mem_usage = 0.0
        return ResourceRequirements(
            cpu_cores=1,
            memory_gb=0.5,
            gpu_count=1,
        )

    def release(self) -> bool:
        """release resources"""
        try:
            logger.info("release OCR engine resources")

            if self._session:
                self._session = None
                logger.info("OCR session cleaned")

            # clean up the GPU memory cache of PaddlePaddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                logger.info("GPU cache cleaned")

            self._model_loaded = False
            return True

        except Exception as e:
            logger.error(f"failed to release resources: {e}")
            return False

    def test_inference(self, test_inputs: Optional[List[InferenceInput]] = None) -> InferenceResult:
        """
        test the end-to-end availability of the inference engine
        """
        logger.info(f"Performing test inference on {self.__class__.__name__}...")
        start_time_sec = time.time()

        try:
            # create virtual image data for testing
            # OCR usually processes images in BGR format (height, width, channels)
            height, width, channels = 640, 640, 3
            logger.debug(f"OCR test input dimension: [{height}, {width}, {channels}]")

            # generate random image data (0-255 range, uint8 type, BGR format)
            raw_img = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

            # execute batch processing inference test
            _ = self._batch_process([{"images": [raw_img]}])

            end_time_sec = (time.time() - start_time_sec) * 1000

            logger.info(f"OCR test inference completed successfully in {end_time_sec:.2f}ms")

            return InferenceResult(success=True, processing_time_ms=end_time_sec)

        except Exception as e:
            end_time_sec = (time.time() - start_time_sec) * 1000
            logger.error(f"Exception during test inference: {e}")
            return InferenceResult(
                success=False, error_message=str(e), processing_time_ms=end_time_sec)


    def _download_image_as_array(self, url: str, idx: int) -> np.ndarray:
        """download image from URL and return image array directly"""
        try:
            logger.info(f" [index {idx}] downloading image: {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            # check content type
            ct = resp.headers.get("Content-Type", "")
            if not ct.startswith("image/"):
                raise ValueError(f"URL did not return image content: Content-Type={ct}")

            # decode image
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("failed to decode downloaded image")

            logger.info(f"[index {idx}] successfully downloaded image: shape={img.shape}")
            return img

        except Exception as e:
            logger.error(f"[index {idx}] failed to download image from URL, url={url}: {e}")
            raise ValueError(f"failed to download image from URL [index {idx}], url={url}: {e}")

    def _decode_base64_to_image(self, b64_str: str, idx: int) -> np.ndarray:
        """decode Base64 string to image array"""
        try:
            logger.info(f"[index {idx}] decoding Base64 image...")

            # decode Base64
            b64_str = b64_str.strip()
            img_bytes = base64.b64decode(b64_str)

            # try to decode with OpenCV
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                # fallback to PIL
                logger.info(f"[index {idx}] OpenCV decoding failed, trying PIL...")
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            logger.info(f"[index {idx}] successfully decoded Base64 image: shape={img.shape}")
            return img

        except Exception as e:
            logger.error(f"[index {idx}] failed to decode Base64 image: {e}")
            raise ValueError(f"failed to decode Base64 image [index {idx}]: {e}")

    def _normalize_image_channels(self, img: np.ndarray, idx: int) -> np.ndarray:
        """normalize image channels to BGR three channels"""
        try:
            logger.debug(f"[index {idx}] normalizing image channels, original shape: {img.shape}")

            if len(img.shape) == 2:
                # grayscale image to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                logger.debug(f"[index {idx}] grayscale image converted to BGR")
            elif len(img.shape) == 3:
                if img.shape[2] == 1:
                    # single channel image to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    logger.debug(f"[index {idx}] single channel image converted to BGR")
                elif img.shape[2] == 3:
                    # already BGR three channels, keep unchanged
                    logger.debug(f"[index {idx}] already BGR three channels")
                elif img.shape[2] == 4:
                    # RGBA to BGR, discard Alpha channel
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    logger.debug(f"[index {idx}] RGBA image converted to BGR")
                else:
                    # more than 4 channels, truncate to the first 3 channels
                    img = img[:, :, :3]
                    logger.warning(f"[index {idx}] multi-channel image ({img.shape[2]} channels) has been truncated to the first 3 channels")
            else:
                raise ValueError(f"unsupported image dimension: {img.shape}")

            # ensure the final is a three-channel BGR format
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"channels converted but still not a three-channel image, shape={img.shape}")

            logger.debug(f"✅ [index {idx}] channel normalization completed: {img.shape}")
            return img

        except Exception as e:
            logger.error(f"❌ [index {idx}] failed to normalize image channels: {e}")
            raise ValueError(f"failed to normalize image channels [index {idx}]: {e}")
