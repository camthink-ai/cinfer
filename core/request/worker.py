# cinfer/request/worker.py
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable, Any, List

from core.engine.service import EngineService
from schemas.request import InferenceRequest
from .queue import RequestQueue # The RequestQueue for a specific model
from core.engine.base import InferenceResult # For type hinting

class WorkerPool:
    """
    Manages a dedicated pool of worker threads for processing inference requests
    for a specific model from its RequestQueue.
    As per document section 4.3.1.
    """
    def __init__(self,
                 model_id: str,
                 num_workers: int,
                 request_queue: RequestQueue,
                 engine_service: EngineService,
                 metrics_collector: Optional[Any] = None): # Placeholder for MetricsCollector
        self.model_id = model_id
        self.num_workers = max(1, num_workers) # Ensure at least one worker
        self._request_queue: RequestQueue = request_queue
        self._engine_service: EngineService = engine_service
        self._metrics_collector = metrics_collector # Placeholder

        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix=f"CinferWorker-{self.model_id}"
        )
        self._worker_futures: List[Future] = []
        self._stop_event = threading.Event() # Event to signal workers to stop
    