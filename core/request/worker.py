# cinfer/request/worker.py
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable, Any, List

from core.engine.service import EngineService
from schemas.request import InferenceRequest
from .queue import RequestQueue # The RequestQueue for a specific model
from core.engine.base import InferenceResult # For type hinting
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

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
        
        logger.info(f"WorkerPool for model '{self.model_id}' initialized with {self.num_workers} worker(s).") # Use logging

    def _process_request_task(self, request: InferenceRequest):
        """
        The actual task of processing a single inference request.
        This is what's submitted to the ThreadPoolExecutor by the main worker loop.
        """
        logger.info(f"Worker (model: {self.model_id}, req: {request.id}): Starting processing.") # Use logging
        result: Optional[InferenceResult] = None
        try:
            # Perform inference using the EngineService
            # The request.inputs is InferenceRequestData, which has input_list: List[InferenceInput]
            result = self._engine_service.predict(model_id=request.model_id, inputs=request.inputs.input_list)
            
            if self._metrics_collector and result:
                # self._metrics_collector.record_inference_time(self.model_id, result.processing_time_ms) # Example
                # self._metrics_collector.increment_processed_requests(self.model_id, result.success)
                pass

        except Exception as e:
            logger.error(f"Worker (model: {self.model_id}, req: {request.id}): Error during inference: {e}") # Use logging
            result = InferenceResult(success=False, error_message=f"Unhandled exception during inference: {e}")
        finally:
            if result is None: # Should not happen if try/except covers all paths
                result = InferenceResult(success=False, error_message="Unknown error during processing, result not set.")
            
            # Resolve the request in the queue (sets future for sync waits, stores for async)
            self._request_queue.resolve_request(request.id, result)
            self._request_queue.task_done() # Signal to PriorityQueue for join()
            logger.info(f"Worker (model: {self.model_id}, req: {request.id}): Finished processing. Success: {result.success}") # Use logging

    def _worker_loop(self):
        """
        The main loop for each worker thread.
        Continuously dequeues requests and submits them for processing.
        """
        logger.info(f"Worker thread started for model '{self.model_id}' (Thread: {threading.current_thread().name})") # Use logging
        while not self._stop_event.is_set():
            request: Optional[InferenceRequest] = None
            try:
                # Dequeue with a timeout to allow checking _stop_event periodically
                request = self._request_queue.dequeue(block=True, timeout=0.5) # Timeout in seconds
            except queue.Empty: # From queue.PriorityQueue, not used here directly but conceptually
                # Timeout occurred, loop again to check stop_event
                continue
            
            if self._stop_event.is_set(): # Check event again after dequeue attempt
                if request: # If a request was fetched just before stop, re-queue or handle
                    # For simplicity, we'll ignore it, or it could be re-queued if persistence is desired.
                    # print(f"Worker (model: {self.model_id}): Stop event set, ignoring dequeued request {request.id}") # Use logging
                    self._request_queue.task_done() # Still need to call task_done
                break

            if request:
                # Instead of self._executor.submit for _process_request_task,
                # the worker loop *is* the task executor for items from its dedicated queue.
                # The ThreadPoolExecutor is used to run these _worker_loop instances.
                self._process_request_task(request)
            # else:
                # This branch is reached if dequeue times out and returns None

        # print(f"Worker thread stopped for model '{self.model_id}' (Thread: {threading.current_thread().name})") # Use logging


    def start(self):
        """Starts the worker threads in the pool."""
        if self._worker_futures:
            logger.info(f"WorkerPool for model '{self.model_id}' already started or starting.") # Use logging
            return

        self._stop_event.clear()
        for _ in range(self.num_workers):
            # Each worker runs the _worker_loop, which fetches from the shared _request_queue
            future = self._executor.submit(self._worker_loop)
            self._worker_futures.append(future)
        logger.info(f"WorkerPool for model '{self.model_id}' started with {self.num_workers} worker threads.") # Use logging

    def stop(self, wait: bool = True):
        """
        Signals all worker threads to stop and shuts down the executor.
        Args:
            wait (bool): If True, waits for all worker threads to complete.
        """
        logger.info(f"Stopping WorkerPool for model '{self.model_id}'...") # Use logging
        self._stop_event.set() # Signal all worker_loop instances to exit
        
        # Wait for all submitted _worker_loop tasks to complete.
        # The timeout in dequeue within _worker_loop allows them to see _stop_event.
        self._executor.shutdown(wait=wait) # This waits for futures to complete
        self._worker_futures.clear()
        logger.info(f"WorkerPool for model '{self.model_id}' stopped.") # Use logging

    def adjust_workers(self, new_num_workers: int) -> bool:
        """
        Adjusts the number of active worker threads.
        Note: ThreadPoolExecutor doesn't directly support dynamic resizing of max_workers
        after initialization in a simple way that reduces active threads cleanly.
        A common approach is to create a new executor, which is complex to manage gracefully
        with ongoing tasks.
        This implementation will only allow increasing workers if the current executor
        was created with a smaller number than its max_workers capacity (not how TPE works),
        or by stopping and starting with a new executor (which QueueManager would handle).

        For now, this method will be a placeholder or simplified.
        QueueManager will likely handle this by stopping this pool and starting a new one.
        """
        new_num_workers = max(1, new_num_workers)
        if new_num_workers == self.num_workers:
            return True # No change needed

        logger.info(f"Adjusting workers for model '{self.model_id}' from {self.num_workers} to {new_num_workers} is complex " # Use logging
        f"with ThreadPoolExecutor. Typically involves pool replacement by QueueManager.")
        # To truly adjust, QueueManager would stop this pool and create a new one.
        # This method can signal the desired number, and QueueManager can act.
        self.num_workers = new_num_workers # Update desired count for QueueManager to see
        return False # Indicate that a restart by QueueManager is likely needed