# main.py
import logging
import threading
from contextlib import asynccontextmanager
from filelock import FileLock
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.exceptions import APIError
from schemas.common import UnifiedAPIResponse
from fastapi.exceptions import RequestValidationError
from utils.errors import ErrorCode
# Core and services
from core.config import get_config_manager, ConfigManager
from core.logging import setup_logging
from core.database import DatabaseFactory, DatabaseService

from core.engine.factory import engine_registry, EngineRegistry
from core.engine.service import EngineService

from core.model.model_store import ModelStore
from core.model.validator import ModelValidator
from core.model.manager import ModelManager

from core.auth.rate_limiter import RateLimiter
from core.auth.token import TokenService
from core.auth.service import AuthService

from core.request.queue_manager import QueueManager
from core.request.processor import RequestProcessor

from monitoring.collector import SystemMonitor
# API Routers
from api.internal.auth import router as internal_auth_router
from api.internal.system import router as internal_system_router
from api.internal.models import router as internal_models_router
from api.internal.tokens import router as internal_tokens_router
from api.openapi.models import router as openapi_models_router


# --- Global State / App Context  ---
logger = logging.getLogger(f"cinfer.{__name__}")


# --- FastAPI Application Instance ---
config_manager = get_config_manager()
system_config = config_manager.get_config("system", {"name": "CamThink AI Inference Platform", "version": "1.0.0"})

app = FastAPI(
    title=system_config["name"],
    description="A lightweight, high-performance visual AI inference service system.",
    version=system_config["version"] 
)


# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    # 1. Initialize Configuration
    config_manager = get_config_manager() 
    app.state.config = config_manager
    
    # 2. Setup Logging
    setup_logging()
    logger.info("Logging initialized.")

    # 3. Initialize Database Service
    db_config = config_manager.get_config("database", {"type": "sqlite", "path": "data/cinfer.db"})
    db_service = DatabaseFactory.create_database(db_config)

    if not db_service.connect():
        logger.critical("Failed to connect to the database. Application cannot start.")
    else:
        logger.info("Database service connected.")

    app.state.db = db_service

    # 4. Initialize Engine Service (uses EngineRegistry which is global)
    engine_service = EngineService(config_manager=config_manager, engine_reg=engine_registry)
    app.state.engine_service = engine_service
    logger.info("EngineService initialized.")

    # 5. Initialize Model Management Components
    model_store = ModelStore(config_manager=config_manager)
    #app.state.model_store = model_store
    model_validator = ModelValidator(engine_registry_instance=engine_registry, config_manager_instance=config_manager)
    #app.state.model_validator = model_validator
    model_manager = ModelManager(
        db_service=db_service,
        engine_service=engine_service,
        model_store=model_store,
        model_validator=model_validator
    )
    app.state.model_manager = model_manager
    logger.info("ModelManager initialized.")

    # 6. Initialize Auth Components
    rate_limiter = RateLimiter() # Uses global config_manager
    token_service = TokenService(db_service=db_service) # Uses global config_manager via security utils
    app.state.token_service = token_service
    auth_service = AuthService(
        token_service=token_service,
        rate_limiter=rate_limiter
    )
    app.state.auth_service = auth_service
    logger.info("AuthService initialized.")

    # 7. Initialize Request Processing Components
    queue_manager = QueueManager(engine_service=engine_service) # Uses global config_manager
    app.state.queue_manager = queue_manager
    request_processor = RequestProcessor(
        queue_manager=queue_manager,
        model_manager=model_manager
        # metrics_collector can be added here
    )
    app.state.request_processor = request_processor
    logger.info("RequestProcessor initialized.")

    # 8. Initialize System Monitor
    lock_file = "system_monitor.lock"
    app.state.monitor_lock = FileLock(lock_file, timeout=1)  # 1秒超时
    system_monitor = SystemMonitor(db_service=db_service)
    app.state.system_monitor = system_monitor
    
    def start_monitor_with_lock():
        try:
            app.state.monitor_lock.acquire(blocking=False)
            logger.info("Acquired lock for system monitoring. Starting monitor thread.")
            system_monitor.run_continuous()
        except TimeoutError:
            logger.info("Another process is already running the system monitor.")
        except Exception as e:
            logger.error(f"Error in monitor thread: {e}")
    
    threading.Thread(target=start_monitor_with_lock, daemon=True).start()
    
    # TODO: Load published models into EngineService/QueueManager from DB
    # logger.info("Pre-loading published models...")


# --- Application Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    # Shutdown
    logger.info("Application shutdown...")
    # Release resources in reverse order of initialization or based on dependency
    queue_manager = getattr(app.state, "queue_manager", None)
    if queue_manager:
        queue_manager.shutdown_all()
        logger.info("QueueManager shut down.")
        
    engine_service = getattr(app.state, "engine_service", None)
    if engine_service:
        engine_service.release_all_engines()
        logger.info("EngineService engines released.")

    db_service = getattr(app.state, "db_service", None)
    if db_service:
        db_service.disconnect()
        logger.info("Database service disconnected.")
    
    logger.info("Application shutdown complete.")


# --- Middleware ---
# CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Configure allowed origins (e.g., ["http://localhost:3000"] for React UI)
    allow_credentials=True,
    allow_methods=["*"], # Or specify ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"], # Or specify specific headers
)

@app.middleware("http")
async def spa_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # if 404 and not api route, return index.html
    if response.status_code == 404 and not request.url.path.startswith("/api"):
        return FileResponse("static/index.html")
    
    return response

# --- Global Exception Handler (Example) ---

@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content=UnifiedAPIResponse(
            success=False,
            error_code=exc.error_code,
            message=exc.detail,
            error_details=exc.details,
            data=None
        ).model_dump(exclude_none=True)
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log the exception
    logger = logging.getLogger(f"cinfer.{__name__}")
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=UnifiedAPIResponse(
            success=False,
            error_code=ErrorCode.COMMON_INTERNAL_ERROR.code,
            message=ErrorCode.COMMON_INTERNAL_ERROR.message,
            data=None,
            error_details=None
        ).model_dump(exclude_none=True)
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Log the HTTP exception if desired, or let FastAPI handle its default logging
   return JSONResponse(
        status_code=exc.status_code,
        content=UnifiedAPIResponse(
            success=False,
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            data=None,
            error_details=None
        ).model_dump(exclude_none=True)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        loc = error["loc"]
        field_name = ".".join(str(item) for item in loc if item != "body")
        
        errors.append({
            "field": field_name,
            "message": error["msg"],
            "error_type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content=UnifiedAPIResponse(
            success=False,
            error_code=ErrorCode.COMMON_VALIDATION_ERROR.code,
            message=ErrorCode.COMMON_VALIDATION_ERROR.message,
            data=None,
            error_details={"errors": errors}
        ).model_dump(exclude_none=True)
    )

# --- API Routers ---
app.include_router(internal_auth_router, prefix="/api/v1/internal/auth", tags=["Internal - Auth"])
app.include_router(internal_system_router, prefix="/api/v1/internal/system", tags=["Internal - System"])
app.include_router(internal_models_router, prefix="/api/v1/internal/models", tags=["Internal - Models"])
app.include_router(internal_tokens_router, prefix="/api/v1/internal/tokens", tags=["Internal - Tokens"])

app.include_router(openapi_models_router, prefix="/api/v1/models", tags=["OpenAPI - Models"])





# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to Cinfer API (Version {app.version})"}

# To run this application (after saving as main.py in project root):
# Ensure all dependencies (FastAPI, Uvicorn, Passlib, python-jose, PyYAML, etc.) are installed.
# pip install fastapi uvicorn passlib[bcrypt] python-jose[cryptography] PyYAML
# Then run: uvicorn main:app --reload
