# cinfer/core/processors/factory.py
from typing import Type, Dict, Optional
from .base import BaseProcessor
from .generic import generic

# processor register
PROCESSOR_REGISTRY = {
    "generic": generic,
    # "yolo_v8_detection": YOLOv8Processor,
}

class ProcessorRegistry:
    """A registry for model processing strategies."""
    def __init__(self):
        self._processors: Dict[str, Type[BaseProcessor]] = {}
        for name, processor_class in PROCESSOR_REGISTRY.items():
            self.register(name, processor_class)

    def register(self, name: str, processor_class: Type[BaseProcessor]):
        """Registers a processor class with a given name."""
        if name in self._processors:
            # Handle duplicates if necessary (e.g., log a warning)
            pass
        self._processors[name] = processor_class

    def get_processor_class(self, name: str) -> Optional[Type[BaseProcessor]]:
        """Retrieves a processor class by its registered name."""
        return self._processors.get(name)

# Create a global instance of the registry that can be imported elsewhere
processor_registry = ProcessorRegistry()
