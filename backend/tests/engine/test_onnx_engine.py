import pytest
import numpy as np
import os
from typing import Dict, Any
from PIL import Image
# path may need to be adjusted according to your actual project structure
from core.engine.onnx import ONNXEngine, onnxruntime # Import onnxruntime to check availability
from schemas.engine import InferenceInput, InferenceResult


EXPECTED_DUMMY_OUTPUT_SHAPE = (1, 25200, 15)
# create a dummy onnx model and put it in tests/assets/dummy_model.onnx
#'input_names': ['images'], 'output_names': ['output0'], 'input_shapes': ['[1, 3, 640, 640]'], 'session_providers': ['CPUExecutionProvider'], 'engine_config_providers': ['CPUExecutionProvider']
@pytest.fixture
def onnx_engine_config_cpu() -> Dict[str, Any]:
    """Provides a basic CPU configuration for ONNXEngine."""
    return {"execution_providers": ["CPUExecutionProvider"], "threads": 1}

@pytest.fixture(scope="module") # module scope, create once per test module
def dummy_onnx_model_path(tmp_path_factory):
    if onnxruntime is None:
        pytest.skip("onnxruntime not installed, skipping real ONNX model tests")

    return "tests/assets/dummy_model.onnx"

@pytest.fixture
def onnx_engine_real(onnx_engine_config_cpu) -> ONNXEngine: # Assuming onnx_engine_config_cpu is defined
    if onnxruntime is None:
        pytest.skip("onnxruntime not installed")
    engine = ONNXEngine()
    initialized = engine.initialize(onnx_engine_config_cpu)
    assert initialized
    return engine

# --- Test Cases with Real Model ---
@pytest.mark.skipif(onnxruntime is None, reason="onnxruntime not installed")
def test_onnx_engine_load_real_model(onnx_engine_real: ONNXEngine, dummy_onnx_model_path: str):
    model_config = {} # Or provide specific config if your dummy model needs it
    loaded = onnx_engine_real.load_model(dummy_onnx_model_path, model_config)
    assert loaded
    assert onnx_engine_real.is_model_loaded()
    info = onnx_engine_real.get_info()
    assert info.loaded_model_path == dummy_onnx_model_path
    assert "images" in info.additional_info["input_names"]
    assert "output0" in info.additional_info["output_names"]
    assert info.additional_info["input_shapes"] == ['[1, 3, 640, 640]']
    assert info.additional_info["session_providers"] == ["CPUExecutionProvider"]
    assert info.additional_info["engine_config_providers"] == ["CPUExecutionProvider"]

@pytest.mark.skipif(onnxruntime is None, reason="onnxruntime not installed")
def test_onnx_engine_predict_real_model_with_image(
    onnx_engine_real: ONNXEngine,
    dummy_onnx_model_path: str, 
    tmp_path # pytest built-in fixture, used to create temporary files and directories
):
    #prepare a sample image
    sample_image_path = "tests/assets/sample.jpg"


    #load the model
    model_config = {} 
    loaded = onnx_engine_real.load_model(dummy_onnx_model_path, model_config)
    assert loaded
    assert onnx_engine_real.is_model_loaded()

    #preprocess the image
    try:
        img = Image.open(sample_image_path).convert('RGB') 
        img_resized = img.resize((640, 640)) # resize to 640x640
        img_np = np.array(img_resized, dtype=np.float32) #convert to float32
        
        #normalize to 0-1 range
        img_np = img_np / 255.0 
        
        #adjust the dimension order: HWC -> CHW
        img_chw = np.transpose(img_np, (2, 0, 1)) 

        print(f"Preprocessed image shape: {img_chw.shape}, dtype: {img_chw.dtype}")

    except Exception as e:
        pytest.fail(f"Failed during image preprocessing: {e}")

    #create InferenceInput
    inputs_for_predict = [InferenceInput(data=img_chw)]

    #predict
    result = onnx_engine_real.predict(inputs_for_predict)

    #print result
    print(f"Prediction result: {result}")

    #assert
    assert result.success, f"Prediction failed: {result.error_message}"
    assert result.outputs is not None, "Outputs should not be None on success"
    
    assert len(result.outputs) == 1, f"Expected 1 output, got {len(result.outputs)}"
    
    output_data = result.outputs[0].data
    assert isinstance(output_data, np.ndarray), "Output data should be a NumPy array"

    print(f"Actual model output_data shape: {output_data.shape}")


    assert output_data.shape == EXPECTED_DUMMY_OUTPUT_SHAPE, \
        f"Output shape mismatch. Expected {EXPECTED_DUMMY_OUTPUT_SHAPE}, got {output_data.shape}"


    assert output_data.dtype == np.float32, f"Expected float32 output, got {output_data.dtype}"

    print(f"Prediction output shape: {output_data.shape} - Test PASSED (not checking for identity)")
    
    
    
   