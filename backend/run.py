# run.py
import sys
import os
import uvicorn
import argparse
import logging # Import logging to ensure our setup is respected

# --- Add these lines to adjust sys.path ---
# Get the absolute path of the directory containing run.py (which is the project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add the project root to the Python path if it's not already there.
# This ensures that packages like 'core', 'api', 'schemas' can be found directly.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

# Corrected import based on your directory structure:
from core.config import get_config_manager, ConfigManager
from core.logging import setup_logging

def main():
    # Initialize ConfigManager first to read server settings
    config: ConfigManager = get_config_manager()

    # Setup logging based on config before Uvicorn initializes fully
    setup_logging()
    logger = logging.getLogger(f"cinfer.{__name__}") # Keep app logger name consistent if desired

    # Initialize the database

    from scripts.init_db import initialize_database
    initialize_database()


    parser = argparse.ArgumentParser(description="Run the Cinfer FastAPI application.")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Run in production mode (multiple workers, no reload).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to (overrides config).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server to (overrides config).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (overrides config, used in --prod mode).",
    )

    args = parser.parse_args()

    default_host = config.get_config("server.host", "127.0.0.1")
    default_port = config.get_config("server.port", 8000)
    default_workers = config.get_config("server.workers", 1)

    host_to_use = args.host if args.host is not None else default_host
    port_to_use = args.port if args.port is not None else default_port
    
    run_config = {
        "app": "main:app", # This refers to main.py in the project root, which is correct
        "host": host_to_use,
        "port": port_to_use,
    }

    if args.prod:
        workers_to_use = args.workers if args.workers is not None else default_workers
        run_config["workers"] = max(1, workers_to_use)
        run_config["reload"] = False
        logger.info(
            f"Starting server in PRODUCTION mode on {host_to_use}:{port_to_use} "
            f"with {run_config['workers']} worker(s)."
        )
    else: # Development mode
        run_config["workers"] = 1
        run_config["reload"] = True
        # Adjust reload_dirs to watch your top-level package directories
        # If 'main.py' contains your app and imports from 'core', 'api', etc.,
        # watching these directories makes sense.
        run_config["reload_dirs"] = [".", "core", "api", "schemas", "utils", "models", "engine", "request", "auth"] # Watch relevant app code dirs
        logger.info(
            f"Starting server in DEVELOPMENT mode on {host_to_use}:{port_to_use} "
            f"with auto-reload enabled."
        )

    uvicorn.run(**run_config)

if __name__ == "__main__":
    main()