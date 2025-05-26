# cinfer/request/queue_manager.py
import threading
from typing import Dict, Optional, Any, List, Tuple

from .queue import RequestQueue
from .worker import WorkerPool
from schemas.request import InferenceRequest, QueueStatus
from core.engine.service import EngineService
from core.config import get_config_manager # For default queue/worker settings
from core.engine.base import InferenceResult # For type hinting

class QueueManager:
    """
    Manages model-specific request queues and their associated worker pools.
    As per document section 4.3.1.
    """
    def __init__(self, engine_service: EngineService):
        self._engine_service: EngineService = engine_service
        self._config = get_config_manager()
        
        # model_id -> RequestQueue instance
        self._queues: Dict[str, RequestQueue] = {}
        # model_id -> WorkerPool instance
        self._worker_pools: Dict[str, WorkerPool] = {}
        
        self._lock = threading.RLock() # To protect access to _queues and _worker_pools

        self.default_max_queue_size: int = self._config.get_config("request.queue_size", 100)
        self.default_workers_per_model: int = self._config.get_config("request.workers_per_model", 2)
        self.default_sync_request_timeout_sec: float = self._config.get_config("request.timeout_ms", 5000) / 1000.0


    def shutdown_all(self):
        """
        Shuts down all worker pools and request queues.
        """
        #TODO
        pass