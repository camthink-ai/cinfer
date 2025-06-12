# cinfer/core/processors/base.py
from typing import List, Dict, Any
from abc import ABC, abstractmethod


from schemas.engine import InferenceInput, InferenceOutput

class BaseProcessor(ABC):
    """
    Abstract base class for model-specific pre-processing and post-processing logic.
    """
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initializes the processor with model-specific configuration.

        Args:
            model_config (Dict[str, Any]): The 'config' dictionary from the model's metadata,
                                           which may contain processor-specific settings.
        """
        self.model_config = model_config
        # You can extract specific parameters here, e.g.,
        # self.resize_dim = model_config.get("resize_dim", (224, 224))

    @abstractmethod
    def preprocess(self, inputs: List[InferenceInput]) -> Dict[str, Any]:
        """
        Preprocesses raw input data into the format expected by the model engine.

        Args:
            inputs (List[InferenceInput]): A list of raw inputs from the user request.

        Returns:
            Dict[str, Any]: A dictionary where keys are input tensor names and
                            values are the processed data (e.g., a NumPy array).
        """
        pass

    @abstractmethod
    def postprocess(self, raw_outputs: List[Any]) -> List[InferenceOutput]:
        """
        Postprocesses the raw model output into a user-friendly format.

        Args:
            raw_outputs (List[Any]): A list of raw outputs from the model engine (e.g., list of NumPy arrays).

        Returns:
            List[InferenceOutput]: A list of structured inference outputs.
        """
        pass