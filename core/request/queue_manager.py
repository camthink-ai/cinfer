# cinfer/request/queue_manager.py
import threading
from typing import Dict, Optional, Any, List, Tuple

from .queue import RequestQueue
from .worker import WorkerPool
from schemas.request import InferenceRequest, QueueStatus
from core.engine.service import EngineService
from core.config import get_config_manager # For default queue/worker settings
from core.engine.base import InferenceResult # For type hinting
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

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


    def _get_or_create_queue_and_pool(self, model_id: str) -> Tuple[Optional[RequestQueue], Optional[WorkerPool]]:
        """
        Retrieves or creates the RequestQueue and WorkerPool for a given model_id.
        This should be called when a model is loaded/published.
        """
        with self._lock:
            if model_id not in self._queues:
                # print(f"QueueManager: Creating new queue and worker pool for model {model_id}.") # Use logging
                
                # Check if model is actually known/loadable by EngineService (conceptual check)
                # model_info = self._engine_service.get_loaded_model_info(model_id) # This checks if *already* loaded by ES
                # if not model_info: # If EngineService doesn't know about it / can't load it, no queue.
                #     print(f"QueueManager: Model {model_id} not found or not loadable by EngineService. Cannot create queue/pool.")
                #     return None, None
                
                # Create RequestQueue
                max_q_size_cfg = self._config.get_config(f"models.{model_id}.queue_size") # Model-specific config
                queue_max_size = max_q_size_cfg if isinstance(max_q_size_cfg, int) else self.default_max_queue_size
                self._queues[model_id] = RequestQueue(model_id=model_id, max_queue_size=queue_max_size)

                # Create and start WorkerPool
                workers_cfg = self._config.get_config(f"models.{model_id}.workers") # Model-specific config
                num_workers = workers_cfg if isinstance(workers_cfg, int) else self.default_workers_per_model
                
                pool = WorkerPool(
                    model_id=model_id,
                    num_workers=num_workers,
                    request_queue=self._queues[model_id],
                    engine_service=self._engine_service
                    # metrics_collector can be passed here if available
                )
                pool.start()
                self._worker_pools[model_id] = pool
            
            return self._queues.get(model_id), self._worker_pools.get(model_id)

    def enqueue_and_wait(self, request: InferenceRequest) -> InferenceResult:
        """
        Enqueues a request for synchronous processing and waits for the result.
        """
        # Ensure queue and pool exist for this model (e.g., when model is published/loaded)
        # This method assumes the model IS active and has a queue.
        # ModelManager.publish_model should call something like self.activate_model_queue(model_id)
        
        queue, _ = self._get_or_create_queue_and_pool(request.model_id) # Lazy creation for now
        if not queue:
            return InferenceResult(success=False, error_message=f"No processing queue available for model {request.model_id}. Model might not be loaded/published.")

        # Use a default timeout for the wait if not specified by request
        request_specific_timeout_sec = (request.timeout_ms / 1000.0) if request.timeout_ms else None
        wait_timeout_sec = request_specific_timeout_sec or self.default_sync_request_timeout_sec
        
        return queue.enqueue_and_wait(request, timeout_sec=wait_timeout_sec)

    def enqueue_async(self, request: InferenceRequest) -> str:
        """
        Enqueues a request for asynchronous processing.
        Returns request_id.
        Raises queue.Full if the specific model queue is full.
        """
        queue, _ = self._get_or_create_queue_and_pool(request.model_id) # Lazy creation
        if not queue:
            # This case implies an issue with model readiness or configuration.
            raise RuntimeError(f"No processing queue available for model {request.model_id}.")
            
        return queue.enqueue_async(request)

    def get_async_result(self, model_id: str, request_id: str) -> Optional[InferenceResult]:
        """
        Retrieves the result of a completed asynchronous request.
        """
        with self._lock:
            queue = self._queues.get(model_id)
        if queue:
            result = queue.get_async_result(request_id)
            if result is not None: # If result is found (not pending)
                queue.remove_async_result(request_id) # Clean up after fetching
            return result
        return None # Or a specific "not found / pending" status

    def get_queue_status(self, model_id: str) -> Optional[QueueStatus]:
        """Gets the status of the queue and worker pool for a specific model."""
        with self._lock:
            queue = self._queues.get(model_id)
            pool = self._worker_pools.get(model_id)

        if queue and pool:
            return QueueStatus(
                model_id=model_id,
                queue_size=queue.size(),
                active_workers=pool.num_workers, # This is desired workers; active threads in TPE is harder to get simply
                max_workers=pool.num_workers,   # Or a configured max if different
                pending_requests=queue.size() # Approximation
            )
        return None

    def adjust_workers(self, model_id: str, new_count: int) -> bool:
        """
        Adjusts the number of worker threads for a given model's pool.
        This typically involves stopping the current pool and starting a new one.
        """
        with self._lock:
            if model_id not in self._queues or model_id not in self._worker_pools:
                logger.error(f"QueueManager: Cannot adjust workers. No pool for model {model_id}.") # Use logging
                return False

            current_pool = self._worker_pools[model_id]
            current_queue = self._queues[model_id] # Keep the existing queue with its pending items

            if current_pool.num_workers == new_count:
                return True # No change needed

            logger.info(f"QueueManager: Adjusting workers for model {model_id} from {current_pool.num_workers} to {new_count}.") # Use logging
            current_pool.stop(wait=True) # Stop and wait for current workers

            # Create and start a new WorkerPool with the new count, re-using the existing queue
            new_pool = WorkerPool(
                model_id=model_id,
                num_workers=new_count,
                request_queue=current_queue, # Re-use the existing queue
                engine_service=self._engine_service
            )
            new_pool.start()
            self._worker_pools[model_id] = new_pool # Replace with the new pool
            # print(f"QueueManager: Workers for model {model_id} adjusted to {new_count}.") # Use logging
            return True

    def activate_model_queue(self, model_id: str, num_workers: Optional[int] = None, max_q_size: Optional[int] = None):
        """
        Explicitly creates/activates the queue and worker pool for a model.
        Called by ModelManager when a model is published/loaded.
        """
        with self._lock:
            if model_id in self._queues:
                logger.info(f"QueueManager: Queue for model {model_id} already active.") # Use logging
                # Optionally adjust workers if num_workers is different
                if num_workers is not None and self._worker_pools[model_id].num_workers != num_workers:
                    self.adjust_workers(model_id, num_workers)
                return

            logger.info(f"QueueManager: Activating queue and worker pool for model {model_id}.") # Use logging
            
            effective_max_q_size = max_q_size if max_q_size is not None else self.default_max_queue_size
            self._queues[model_id] = RequestQueue(model_id=model_id, max_queue_size=effective_max_q_size)

            effective_num_workers = num_workers if num_workers is not None else self.default_workers_per_model
            pool = WorkerPool(
                model_id=model_id,
                num_workers=effective_num_workers,
                request_queue=self._queues[model_id],
                engine_service=self._engine_service
            )
            pool.start()
            self._worker_pools[model_id] = pool

    def deactivate_model_queue(self, model_id: str):
        """
        Deactivates and removes the queue and worker pool for a model.
        Called by ModelManager when a model is unpublished/deleted.
        """
        with self._lock:
            logger.info(f"QueueManager: Deactivating queue and worker pool for model {model_id}.") # Use logging
            pool = self._worker_pools.pop(model_id, None)
            if pool:
                pool.stop(wait=True) # Wait for workers to finish processing existing queue items
            
            queue = self._queues.pop(model_id, None)
            if queue:
                # Handle any remaining items in the queue if necessary (e.g., log, reject)
                # For now, they are just discarded with the queue object.
                logger.info(f"QueueManager: Queue for model {model_id} removed. Remaining items: {queue.size()}") # Use logging
                pass
            logger.info(f"QueueManager: Queue and pool for model {model_id} deactivated.") # Use logging

    def shutdown_all(self):
        """Shuts down all worker pools and clears queues."""
        logger.info("QueueManager: Shutting down all worker pools...") # Use logging
        with self._lock:
            for model_id in list(self._worker_pools.keys()):
                self.deactivate_model_queue(model_id)
        logger.info("QueueManager: All worker pools shut down.") # Use logging