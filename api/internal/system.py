# cinfer/api/internal/system.py
import logging
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException

from core.config import ConfigManager
from core.request.processor import RequestProcessor
from core.request.queue_manager import QueueManager
from core.engine.service import EngineService
from schemas.request import HealthStatus, QueueStatus
from api.dependencies import get_config, get_request_proc, get_queue_mgr, get_engine_svc # Dependency getters
from api.dependencies import require_admin_user # Use the new dependency
router = APIRouter(dependencies=[Depends(require_admin_user)])

logger = logging.getLogger(f"cinfer.{__name__}")


