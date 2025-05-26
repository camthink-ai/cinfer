# cinfer/api/internal/system.py
import logging
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException

from core.config import ConfigManager
from core.request.processor import RequestProcessor
from core.request.queue_manager import QueueManager
from core.engine.service import EngineService
from schemas.request import HealthStatus, QueueStatus, SystemStatus
from api.dependencies import get_config, get_request_proc, get_queue_mgr, get_engine_svc, get_db_service # Dependency getters
from api.dependencies import require_admin_user 
from core.database.base import DatabaseService
router = APIRouter()

logger = logging.getLogger(f"cinfer.{__name__}")


@router.get("/status", response_model=SystemStatus, summary="System Status")
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
        return SystemStatus(init=False, message="System not initialized.")
    else:
        return SystemStatus(init=True, message="System initialized.")

