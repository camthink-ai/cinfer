# cinfer/request/processor.py
from typing import Dict, Any, Optional, List, Tuple, Type
import time
from .queue_manager import QueueManager
from core.model.manager import ModelManager # To validate model_id and its status
from schemas.request import InferenceRequest, HealthStatus, QueueStatus # Pydantic models
from schemas.engine import InferenceResult, InferenceInput # For return types
from schemas.request import InferenceRequestData
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

class RequestProcessor:
    """
    Main entry point for processing inference requests.
    Coordinates with ModelManager and QueueManager.
    As per document section 4.3.1.
    """
    def __init__(self,
                 queue_manager: QueueManager,
                 model_manager: ModelManager,
                 metrics_collector: Optional[Any] = None): # Placeholder for MetricsCollector
        self._queue_manager: QueueManager = queue_manager
        self._model_manager: ModelManager = model_manager
        self._metrics_collector = metrics_collector # Placeholder
        logger.info("RequestProcessor initialized.") 

    async def process_request(self, request_payload: Dict[str, Any]) -> InferenceResult:
        """
        Processes a synchronous inference request.
        Validates the request, checks model status, and enqueues for processing.
        Args:
            request_payload (Dict[str, Any]): The raw dictionary payload for the inference request.
                                             Expected to match InferenceRequest schema.
        Returns:
            InferenceResult: The result of the inference.
        """
        request_id_for_metric: Optional[str] = None
        start_time = time.time() # For overall processing time metric, if needed outside engine

        try:
            # 1. Parse and validate the incoming payload into an InferenceRequest object
            # Pydantic will raise ValidationError if payload is malformed
            logger.info(f"Request payload: {request_payload}")
            inference_req = InferenceRequest(**request_payload)
            request_id_for_metric = inference_req.id
            
            # Further custom validation on the request object itself
            validation_res = inference_req.validate_request()
            if not validation_res.valid:
                logger.error(f"Request validation failed for {inference_req.id}: {validation_res.errors}") # Use logging
                return InferenceResult(success=False, error_message=f"Request validation failed: {validation_res.errors}")

        except Exception as e: # Covers Pydantic ValidationError and others
            logger.error(f"Failed to parse or validate request payload: {e}") # Use logging
            # For Pydantic errors, e.errors() gives more detail.
            return InferenceResult(success=False, error_message=f"Invalid request payload: {e}")

        # 2. Check model existence and status using ModelManager
        model = await self._model_manager.get_model(inference_req.model_id) # Assuming async ModelManager
        if not model:
            logger.error(f"Model {inference_req.model_id} not found for request {inference_req.id}.") # Use logging
            return InferenceResult(success=False, error_message=f"Model '{inference_req.model_id}' not found.")
        
        # Ensure model is in a servable state (e.g., "published")
        # Assuming model.status is a string like "published"
        if model.status != "published": # Or use ModelStatus.PUBLISHED if it's an enum
            logger.error(f"Model {model.id} is not in a published state (status: {model.status}). Cannot process request {inference_req.id}.") # Use logging
            return InferenceResult(success=False, error_message=f"Model '{model.id}' is not published (status: {model.status}).")

        # 3. Enqueue the request for synchronous processing (waits for result)
        logger.info(f"Processing request {inference_req.id} for model {inference_req.model_id} synchronously...") # Use logging
        if self._metrics_collector:
            # self._metrics_collector.increment_request_received(model_id=inference_req.model_id)
            pass
        
        result = self._queue_manager.enqueue_and_wait(inference_req)

        if self._metrics_collector:
            # overall_time_ms = (time.time() - start_time) * 1000
            # self._metrics_collector.record_request_latency(request_id_for_metric, overall_time_ms, result.success)
            pass
            
        return result

    async def process_request_async(self, request_payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Processes an asynchronous inference request.
        Validates and enqueues the request, returns a request ID.
        Args:
            request_payload (Dict[str, Any]): The raw dictionary payload.
        Returns:
            Tuple[Optional[str], Optional[str]]: (request_id, None) on success, 
                                                 (None, error_message) on failure.
        """
        try:
            inference_req = InferenceRequest(**request_payload)
            validation_res = inference_req.validate_request()
            if not validation_res.valid:
                return None, f"Request validation failed: {validation_res.errors}"
        except Exception as e:
            return None, f"Invalid request payload: {e}"

        model = await self._model_manager.get_model(inference_req.model_id)
        if not model:
            return None, f"Model '{inference_req.model_id}' not found."
        if model.status != "published": # Or use ModelStatus.PUBLISHED
            return None, f"Model '{model.id}' is not published (status: {model.status})."

        try:
            logger.info(f"Processing request {inference_req.id} for model {inference_req.model_id} asynchronously...") # Use logging
            if self._metrics_collector:
                # self._metrics_collector.increment_request_received(model_id=inference_req.model_id, async_req=True)
                pass
            request_id = self._queue_manager.enqueue_async(inference_req)
            return request_id, None
        except Exception as e: # Catches queue.Full or other runtime errors from enqueue_async
            logger.error(f"Failed to enqueue async request {inference_req.id}: {e}") # Use logging
            return None, str(e)

    async def get_async_request_result(self, model_id: str, request_id: str) -> Optional[InferenceResult]:
        """
        Retrieves the result of a previously submitted asynchronous request.
        Args:
            model_id (str): The model ID the request was for.
            request_id (str): The ID of the asynchronous request.
        Returns:
            Optional[InferenceResult]: The result if available, else None (or specific status).
        """
        logger.info(f"Fetching async result for request ID {request_id}, model {model_id}...") # Use logging
        result = self._queue_manager.get_async_result(model_id, request_id)
        if result and self._metrics_collector:
            # self._metrics_collector.record_async_result_fetched(request_id)
            pass
        return result


    async def process_batch(self, requests_payload: List[Dict[str, Any]]) -> List[InferenceResult]:
        """
        Processes a batch of inference requests.
        (Placeholder - marked for future iteration in document 2.2.2)
        """
        logger.info("Batch processing is not yet fully implemented.") # Use logging
        # This would involve:
        # 1. Parsing each item in requests_payload into InferenceRequest.
        # 2. Validating each request and model status.
        # 3. Potentially enqueuing them individually or using a specific batch queue mechanism if developed.
        # 4. Aggregating results.
        # For now, returning error or processing sequentially as a basic fallback.
        results = []
        for req_payload in requests_payload:
            # This is a naive sequential processing, not true batch optimization.
            result = await self.process_request(req_payload) 
            results.append(result)
        return results

    async def health_check(self) -> HealthStatus:
        """
        Performs a health check of the request processing system.
        Checks queue manager status and potentially other components.
        """
        logger.info("Performing health check...") # Use logging
        # For a more comprehensive health check, you might query:
        # - Each model's queue status from QueueManager
        # - EngineService status (e.g., loaded models, any errors)
        # - Database connectivity
        
        component_statuses: List[Dict[str, Any]] = []
        overall_ok = True

        # Example: Check status of a few model queues if QueueManager exposes a way to list active models
        # active_models = self._queue_manager.get_active_model_ids() # Needs to be implemented in QueueManager
        # for model_id in active_models[:3]: # Check a few
        #     q_status = self._queue_manager.get_queue_status(model_id)
        #     if q_status:
        #         component_statuses.append({
        #             "component": f"Queue-{model_id}",
        #             "status": "OK", # Basic check, could be more detailed
        #             "details": q_status.model_dump()
        #         })
        #     else:
        #         component_statuses.append({"component": f"Queue-{model_id}", "status": "UNKNOWN"})
        #         overall_ok = False # If a managed queue is not found, it might be an issue

        # For now, a simple health check
        # A more robust check would ping dependencies or check internal states.
        if self._queue_manager: # Basic check that queue_manager exists
            status_message = "Request processor and queue manager are operational."
            # Could add more details, e.g., number of models with active queues
            # num_active_queues = len(self._queue_manager._queues) # Accessing protected for example
            # status_message += f" Managing {num_active_queues} model queues."
        else:
            status_message = "Queue manager not available."
            overall_ok = False
            
        return HealthStatus(
            status="OK" if overall_ok else "ERROR",
            message=status_message,
            component_statuses=component_statuses
        )
    
    async def convert_inputs_to_inference_list(self, model_input_schema: Type[Any]) -> InferenceRequestData:
        """
        Convert a Pydantic model to a list of dictionaries.
        """
        input_list = list()
        model_schema = model_input_schema.model_dump()
        items = list(model_schema.items())
        first_value = items[0][1]
        rest_items = dict(items[1:])

        logger.info(f"First value: {first_value}")
        logger.info(f"Rest items: {rest_items}")

        input_schema = InferenceInput(
            data=first_value,
            metadata=rest_items
        )

        logger.info(f"Input schema: {input_schema}")
        input_list.append(input_schema)
         
        return InferenceRequestData(
            input_list=input_list
        )