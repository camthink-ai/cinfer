import pytest
import numpy as np
from typing import List, Dict, Any

from core.engine.dummy import DummyEngine
from schemas.engine import InferenceInput, InferenceResult, EngineInfo, ResourceRequirements

@pytest.fixture
def dummy_engine_config() -> Dict[str, Any]:
    return {"dummy_device": "CPU_test", "dummy_version": "0.1-test"}

@pytest.fixture
def dummy_model_config_simple() -> Dict[str, Any]:
    return {
        "dummy_input_names": ["image"],
        "dummy_output_names": ["scores"],
        "dummy_input_shapes": [[-1, 3, 64, 64]], # Batch, C, H, W; -1 for dynamic batch
        "dummy_input_types": ["tensor(float)"],
        "dummy_output_dim": 5, # e.g., 5 classes
        "dummy_processing_delay_ms": 10 # Simulate 10ms delay
    }

@pytest.fixture
def dummy_engine(dummy_engine_config: Dict[str, Any]) -> DummyEngine:
    engine = DummyEngine()
    initialized = engine.initialize(dummy_engine_config)
    assert initialized, "DummyEngine failed to initialize"
    return engine

def test_dummy_engine_initialization(dummy_engine: DummyEngine, dummy_engine_config: Dict[str, Any]):
    """Test engine initialization and basic info."""
    assert dummy_engine.is_initialized()
    info = dummy_engine.get_info()
    assert info.engine_name == "DummyEngine"
    assert info.engine_version == dummy_engine_config["dummy_version"]
    assert not info.model_loaded
    assert info.current_device == dummy_engine_config["dummy_device"]

def test_dummy_engine_load_model(dummy_engine: DummyEngine, dummy_model_config_simple: Dict[str, Any]):
    """Test model loading simulation."""
    model_path = "data/models/sample.file"
    loaded = dummy_engine.load_model(model_path, dummy_model_config_simple)
    assert loaded
    assert dummy_engine.is_model_loaded()

    info = dummy_engine.get_info()
    assert info.model_loaded
    assert info.loaded_model_path == model_path
    assert info.additional_info["input_names"] == dummy_model_config_simple["dummy_input_names"]
    assert info.additional_info["output_names"] == dummy_model_config_simple["dummy_output_names"]

def test_dummy_engine_predict(dummy_engine: DummyEngine, dummy_model_config_simple: Dict[str, Any]):
    """Test predict method simulation."""
    model_path = "data/models/sample.file"
    dummy_engine.load_model(model_path, dummy_model_config_simple)
    assert dummy_engine.is_model_loaded()

    # Create dummy input data
    # Assuming first input shape is something like (batch, C, H, W)
    # For dummy, let's create a batch of 2
    batch_size = 2
    input_data_shape = (batch_size, 3, 64, 64) 
    dummy_np_input = np.random.rand(*input_data_shape).astype(np.float32)
    
    # DummyEngine's _preprocess_input current simple implementation expects data for the first input name
    # or a dict if only one raw_input is provided.
    # Let's provide it as a list of InferenceInput, where each data is the tensor for one item in the batch.
    # However, the dummy_preprocess is very simple, so we can also just pass the batch directly.
    # For this test, let's assume inputs is a list of InferenceInput,
    # and our dummy_preprocess can handle creating a batch from it for a single named input.
    
    # Option 1: Single InferenceInput with batched data (if preprocess supports it well)
    # inputs = [InferenceInput(data=dummy_np_input)]

    # Option 2: Multiple InferenceInputs, each an item (more typical for AsyncEngine predict)
    # and let preprocess stack them (current dummy preprocess is simple here)
    # For DummyEngine's current preprocess, let's create a single InferenceInput containing the batch
    # if the model expects a single named input.
    # If the dummy input name is "image":
    # inputs = [InferenceInput(data={"image": dummy_np_input})] # This fits the dict path in preprocess
    # OR if preprocess simply takes the data:
    inputs = [InferenceInput(data=dummy_np_input[i]) for i in range(batch_size)]


    result = dummy_engine.predict(inputs) # This calls the synchronous predict

    assert result.success
    assert result.outputs is not None
    assert len(result.outputs) == len(dummy_model_config_simple["dummy_output_names"])
    
    first_output = result.outputs[0]
    assert first_output.metadata["name"] == dummy_model_config_simple["dummy_output_names"][0]
    assert isinstance(first_output.data, np.ndarray)
    # Check output shape: (batch_size, dummy_output_dim)
    assert first_output.data.shape == (batch_size, dummy_model_config_simple["dummy_output_dim"])
    assert result.processing_time_ms is not None and result.processing_time_ms >= dummy_model_config_simple["dummy_processing_delay_ms"]

def test_dummy_engine_release(dummy_engine: DummyEngine, dummy_model_config_simple: Dict[str, Any]):
    """Test releasing the model."""
    model_path = "data/models/sample.file"
    dummy_engine.load_model(model_path, dummy_model_config_simple)
    assert dummy_engine.is_model_loaded()

    released = dummy_engine.release()
    assert released
    assert not dummy_engine.is_model_loaded()
    info = dummy_engine.get_info()
    assert not info.model_loaded
    assert info.loaded_model_path is None # Base class release clears this

def test_dummy_engine_resource_requirements(dummy_engine: DummyEngine):
    """Test resource requirements reporting."""
    reqs = dummy_engine.get_resource_requirements()
    assert isinstance(reqs, ResourceRequirements)
    assert reqs.cpu_cores >= 0
    assert reqs.memory_gb >= 0
    assert reqs.gpu_count >= 0