# cinfer/api/internal/system.py
import logging
from typing import Dict, Any, List, Annotated, Union

from fastapi import APIRouter, Depends, HTTPException, Header

from core.config import ConfigManager
from core.request.processor import RequestProcessor
from core.request.queue_manager import QueueManager
from core.engine.service import EngineService
from schemas.request import HealthStatus, QueueStatus, SystemStatus
from schemas.common import UnifiedAPIResponse
from schemas.common import SystemMetrics
from api.dependencies import get_config, get_request_proc, get_queue_mgr, get_engine_svc, get_db_service, get_system_monitor, get_internal_auth_result # Dependency getters
from api.dependencies import require_admin_user 
from core.database.base import DatabaseService
from monitoring.collector import SystemMonitor
from schemas.auth import AuthResult
from schemas.common import SystemInfo
from utils.errors import ErrorCode
from utils.exceptions import APIError
from fastapi import status


router = APIRouter()

logger = logging.getLogger(f"cinfer.{__name__}")


@router.get(
        "/status", 
        response_model=UnifiedAPIResponse[SystemStatus],
        response_model_exclude_none=True, 
        summary="System Status"
)
async def get_system_status(
    db: DatabaseService = Depends(get_db_service)
):
    """
    Provides the overall health status of the inference system.
    """
    logger.info("Performing system status check (admin).")
    #I need to check if the system has a registered administrator
    admin_user = db.find_one("users", {"is_admin": True})
    if not admin_user:
        return UnifiedAPIResponse(
            success=True,
            message="System not initialized.",
            data=SystemStatus(init=False)
        )
    else:
        return UnifiedAPIResponse(
            success=True,
            message="System initialized.",
            data=SystemStatus(init=True)
        )


@router.get(
        "/metrics", 
        response_model=UnifiedAPIResponse[List[SystemMetrics]], 
        response_model_exclude_none=True, 
        summary="System Metrics"
)
async def get_system_metrics(
    db: DatabaseService = Depends(get_db_service),
    collector: SystemMonitor = Depends(get_system_monitor),
    auth_result: AuthResult = Depends(get_internal_auth_result)
):
    """
    Provides the system metrics.
    """
    try:
        metrics = collector.read_metrics()
    except Exception as e:
        logger.error(f"Error reading metrics: {e}", exc_info=True)
        raise APIError(
            error=ErrorCode.COMMON_INTERNAL_ERROR
        )
    return UnifiedAPIResponse(
        success=True,
        message="System metrics collected successfully.",
        data=metrics
    )

@router.get(
        "/info", 
        response_model=UnifiedAPIResponse[SystemInfo], 
        response_model_exclude_none=True, 
        summary="System Info"
)
async def get_system_info(
    db: DatabaseService = Depends(get_db_service),
    collector: SystemMonitor = Depends(get_system_monitor),
    auth_result: AuthResult = Depends(get_internal_auth_result)
):
    """
    Provides the system info.
    """
    info = collector.collect_system_info()
    return UnifiedAPIResponse(
        success=True,
        message="System info collected successfully.",
        data=info
    )
