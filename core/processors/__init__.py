# cinfer/core/processors/__init__.py
from .base import BaseProcessor
from .factory import processor_registry

__all__ = ["BaseProcessor", "processor_registry"]