# cinfer/request/queue.py
import queue # Thread-safe PriorityQueue
from typing import Optional, Any, Tuple, Dict
import uuid
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

from schemas.request import InferenceRequest
from core.engine.base import InferenceResult # For type hinting results
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

# Store for results of async requests, mapping request_id to Future[InferenceResult] or InferenceResult itself
# This needs to be thread-safe if accessed by multiple components.
# A more robust solution might involve a dedicated result backend (e.g., Redis) for distributed systems.
# For a single-process app with threads, a thread-safe dict or a dict protected by a lock is okay.
# Let's assume QueueManager will manage this store for now.

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

    def _get_queue_item(self, request: InferenceRequest) -> Tuple[int, float, str, InferenceRequest]:
        """Wraps InferenceRequest for storage in PriorityQueue."""
        # PriorityQueue retrieves lowest valued item first.
        # We want higher numerical priority in request to be processed first.
        # So, store (-priority, entry_time, request_id, request)
        return (-request.priority, time.time(), request.id, request)

    def enqueue_and_wait(self, request: InferenceRequest, timeout_sec: Optional[float] = None) -> InferenceResult:
        """
        Enqueues a request and waits for its result.
        Args:
            request (InferenceRequest): The request to process.
            timeout_sec (Optional[float]): Max seconds to wait for the result.
                                          If None, waits indefinitely (or until request.timeout_ms).
        Returns:
            InferenceResult: The result of the inference.
        Raises:
            queue.Full: If the queue is full and blocking is False (not used here, as put is blocking).
            TimeoutError: If waiting for the result times out.
        """
        if self.is_full(): # Check before creating future if queue has a max size
            return InferenceResult(success=False, error_message=f"Queue for model {self.model_id} is full.")

        request_id = request.id
        future_result: Future[InferenceResult] = Future()
        self._pending_sync_futures[request_id] = future_result
        
        try:
            queue_item = self._get_queue_item(request)
            self._queue.put(queue_item) # This can block if queue is full and has maxsize
            self._entry_timestamps[request_id] = queue_item[1] # Store entry time
            logger.info(f"Request {request_id} enqueued for sync processing (model: {self.model_id}). Size: {self.size()}") # Use logging
        except queue.Full:
            self._pending_sync_futures.pop(request_id, None)
            return InferenceResult(success=False, error_message=f"Queue for model {self.model_id} is full (on put).")
        except Exception as e:
            self._pending_sync_futures.pop(request_id, None)
            return InferenceResult(success=False, error_message=f"Failed to enqueue request {request_id}: {e}")

        try:
            # Wait for the future to be resolved by the worker
            effective_timeout = timeout_sec
            if request.timeout_ms and (timeout_sec is None or request.timeout_ms / 1000 < timeout_sec):
                effective_timeout = request.timeout_ms / 1000
            
            result = future_result.result(timeout=effective_timeout)
            return result
        except FutureTimeoutError:
            # print(f"Timeout waiting for result of request {request_id} (model: {self.model_id}).") # Use logging
            # The request might still be in the queue or being processed.
            # It needs a cancellation mechanism if possible, or just timeout the wait.
            return InferenceResult(success=False, error_message=f"Timeout waiting for inference result for request {request_id}.")
        except Exception as e:
            # print(f"Error waiting for result of request {request_id}: {e}") # Use logging
            return InferenceResult(success=False, error_message=f"Error obtaining result for request {request_id}: {e}")
        finally:
            self._pending_sync_futures.pop(request_id, None)
            self._entry_timestamps.pop(request_id, None)


    def enqueue_async(self, request: InferenceRequest) -> str:
        """
        Enqueues a request for asynchronous processing.
        Args:
            request (InferenceRequest): The request to process.
        Returns:
            str: The request_id.
        Raises:
            queue.Full: If the queue is full.
        """
        if self.is_full():
            raise queue.Full(f"Queue for model {self.model_id} is full.")
            
        request_id = request.id
        queue_item = self._get_queue_item(request)
        self._queue.put_nowait(queue_item) # Use put_nowait for async to raise Full immediately
        self._entry_timestamps[request_id] = queue_item[1]
        logger.info(f"Request {request_id} enqueued for async processing (model: {self.model_id}). Size: {self.size()}") # Use logging
        return request_id

    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        """
        Dequeues the highest priority request. Called by WorkerPool.
        Returns:
            Optional[InferenceRequest]: The dequeued request, or None if timeout/empty.
        """
        try:
            _prio, _ts, _req_id, request_obj = self._queue.get(block=block, timeout=timeout)
            # self._entry_timestamps.pop(_req_id, None) # Keep until processed for timeout checks by manager
            return request_obj
        except queue.Empty:
            return None

    def task_done(self):
        """Signals that a formerly enqueued task is complete."""
        self._queue.task_done()

    def resolve_request(self, request_id: str, result: InferenceResult):
        """
        Called by a worker after processing a request to set its result.
        Resolves the future for synchronous waits or stores result for async polling.
        """
        # For synchronous requests waiting on a Future
        future = self._pending_sync_futures.pop(request_id, None)
        if future and not future.done():
            future.set_result(result)
        else: # Must be an async request, store its result
            self._processed_async_results[request_id] = result
        
        self._entry_timestamps.pop(request_id, None) # Clean up entry timestamp

    def get_async_result(self, request_id: str) -> Optional[InferenceResult]:
        """
        Retrieves the result of a completed asynchronous request.
        Returns:
            Optional[InferenceResult]: The result if ready, else None.
            (Could also return a specific status like "pending" or "not_found")
        """
        # TODO: Add TTL cleanup for _processed_async_results
        return self._processed_async_results.get(request_id)
        
    def remove_async_result(self, request_id: str) -> bool:
        """Removes a stored async result after it has been fetched."""
        if request_id in self._processed_async_results:
            del self._processed_async_results[request_id]
            return True
        return False

    def size(self) -> int:
        """Returns the current number of items in the queue."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.empty()

    def is_full(self) -> bool:
        """Checks if the queue is full (if maxsize was set)."""
        if self._queue.maxsize > 0:
            return self._queue.full()
        return False # Unlimited queue is never full

    def get_request_entry_time(self, request_id: str) -> Optional[float]:
        """Gets the timestamp when a request was enqueued."""
        return self._entry_timestamps.get(request_id)