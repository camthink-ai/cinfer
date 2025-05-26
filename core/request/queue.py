# cinfer/request/queue.py
import queue # Thread-safe PriorityQueue
from typing import Optional, Any, Tuple, Dict
import uuid
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

from schemas.request import InferenceRequest
from core.engine.base import InferenceResult # For type hinting results

# Store for results of async requests, mapping request_id to Future[InferenceResult] or InferenceResult itself
# This needs to be thread-safe if accessed by multiple components.
# A more robust solution might involve a dedicated result backend (e.g., Redis) for distributed systems.
# For a single-process app with threads, a thread-safe dict or a dict protected by a lock is okay.


class RequestQueue:
    """
    Implements a priority queue for inference requests for a single model.
    Uses a thread-safe PriorityQueue.
    As per document section 4.3.1.
    """
    def __init__(self, model_id: str, max_queue_size: Optional[int] = None):
        self.model_id = model_id
        # queue.PriorityQueue: items are (priority, timestamp, request_id, request_object)
        # Lower number for priority means higher priority in queue.PriorityQueue.
        # Our InferenceRequest uses higher number = higher priority, so we'll negate it.
        self._queue: queue.PriorityQueue[Tuple[int, float, str, InferenceRequest]] = \
            queue.PriorityQueue(maxsize=max_queue_size or 0) # 0 for unlimited
        
        self._entry_timestamps: Dict[str, float] = {} # request_id -> enqueue_timestamp

        # For enqueue_and_wait: maps request_id to a Future that will hold the InferenceResult
        self._pending_sync_futures: Dict[str, Future[InferenceResult]] = {}
        # For enqueue_async: maps request_id to InferenceResult (once processed)
        self._processed_async_results: Dict[str, InferenceResult] = {}
        # TODO: Add a mechanism to clean up old _processed_async_results after TTL.

   