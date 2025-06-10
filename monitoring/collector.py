#! /usr/bin/env python3

import os
import sys
import time
import logging
import json
import fcntl  
from typing import Dict, Any, List, Optional, Tuple 
from pathlib import Path
from datetime import datetime, timedelta
from core.config import get_config_manager
import platform
import psutil
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from core.database.base import DatabaseService 
from core.database import DatabaseFactory     
logger = logging.getLogger(f"cinfer.{__name__}") 

DEFAULT_COLLECTION_INTERVAL = 300  # seconds
DEFAULT_OUTPUT_FILE = Path("data/monitoring/metrics.json")

class SystemMonitor:
    """
    Collect system metrics, including CPU, memory, GPU (if available),
    and model deployment status and access token statistics from the database.
    """
    
    def __init__(self, db_service: Optional[DatabaseService] = None, output_file: Path = DEFAULT_OUTPUT_FILE):
        """
        Initialize the system monitor.

        Args:
            db_service (Optional[DatabaseService]): An optional, initialized DatabaseService instance.
                                                    If provided, the monitor will use this instance for database queries.
                                                    The caller is responsible for managing the lifecycle of this instance (connect/disconnect).
            output_file (Path): The path to the metrics output file.
        """
        self.metrics: Dict[str, Any] = {}
        self.db_service: Optional[DatabaseService] = db_service
        self.output_file: Path = output_file
        
        # Ensure the output directory exists
        if not self.output_file.parent.exists():
            try:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Metrics output directory created: {self.output_file.parent}")
            except Exception as e:
                logger.error(f"Failed to create metrics output directory {self.output_file.parent}: {e}")

    # --- Metrics collection methods ---
    def collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU usage metrics"""
        try:
            cpu_freq_info = psutil.cpu_freq()
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1), # Use a shorter interval to avoid blocking
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_freq_current": cpu_freq_info.current if cpu_freq_info else None,
                "load_avg_1m_5m_15m": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}", exc_info=True)
            return {"error": str(e)}
    
    def collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory usage metrics"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "virtual_total_gb": round(memory.total / (1024**3), 2),
                "virtual_available_gb": round(memory.available / (1024**3), 2),
                "virtual_used_gb": round(memory.used / (1024**3), 2),
                "virtual_memory_used": memory.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "swap_memory_used": swap.percent,
            }
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}", exc_info=True)
            return {"error": str(e)}
    
    def collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Collect GPU usage metrics (if GPUtil is available)"""
        if not HAS_GPU:
            return [{"load_percent": "N/A"}]
            
        gpu_metrics_list: List[Dict[str, Any]] = []
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return [{"load_percent": "N/A"}]
            for i, gpu in enumerate(gpus):
                gpu_metrics_list.append({
                    "id": gpu.id, # Use the GPU ID provided by GPUtil
                    "name": gpu.name,
                    "load_percent": round(gpu.load * 100, 2),
                    "memory_total_mb": round(gpu.memoryTotal, 2),
                    "memory_used_mb": round(gpu.memoryUsed, 2),
                    "memory_free_mb": round(gpu.memoryFree, 2),
                    "memory_percent_used": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2) if gpu.memoryTotal > 0 else 0,
                    "temperature_celsius": gpu.temperature
                })
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}", exc_info=True)
            return [{"error": str(e)}]
        return gpu_metrics_list
    
    def collect_model_status(self) -> Dict[str, Any]:
        """Collect model deployment statistics from the database"""
        result = {
            "published_count": 0,
            "unpublished_count": 0, # "unpublished" includes "draft", "archived", etc.
            "total_count": 0
        }
        
        if not self.db_service:
            logger.warning("Database service not provided to SystemMonitor; cannot collect model status.")
            result["status"] = "db_service_not_available"
            return result
        
        if not getattr(self.db_service, 'conn', None): # Check internal connection status (if applicable)
            logger.warning("Database service is not connected; cannot collect model status.")
            result["status"] = "db_not_connected" # Assuming we don't automatically reconnect here
            return result
            
        try:
            # Get total count
            result["total_count"] = self.db_service.count("models")
            
            # Get published count
            result["published_count"] = self.db_service.count("models", {"status": "published"})
            result["unpublished_count"] = result["total_count"] - result["published_count"]
            
        except Exception as e:
            logger.error(f"Error collecting model status from database: {e}", exc_info=True)
            result["status"] = "query_error"
            result["error_message"] = str(e)
            
        return result
    
    def collect_access_token_stats(self) -> Dict[str, Any]: 
        """Collect access token statistics from the database"""
        result = {
            "total_count": 0,
            "active_count": 0
        }
        if not self.db_service: 
            logger.warning("Database service not provided to SystemMonitor; cannot collect access token stats.")
            result["status"] = "db_service_not_available"
            return result

        if not getattr(self.db_service, 'conn', None):
            logger.warning("Database service is not connected; cannot collect access token stats.")
            result["status"] = "db_not_connected"
            return result

        try:
            result["total_count"] = self.db_service.count("access_tokens")
            result["active_count"] = self.db_service.count("access_tokens", {"status": "active"})

        except Exception as e:
            logger.error(f"Error collecting access token stats from database: {e}", exc_info=True)
            result["status"] = "query_error"
            result["error_message"] = str(e)
        return result
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all defined system metrics"""
        timestamp = int(datetime.now().timestamp() * 1000) #milliseconds
        
        self.metrics = {
            "timestamp": timestamp,
            "cpu": self.collect_cpu_metrics(),
            "memory": self.collect_memory_metrics(),
            "gpu": self.collect_gpu_metrics()
        }
        logger.info("Collected all metrics.")
        return self.metrics
    
    
    def collect_os_info(self) -> Dict[str, Any]:
        """Collect OS info"""
        info = platform.freedesktop_os_release()
        os_info = {
            "os_name": info["NAME"],
            "os_version": info["VERSION"]
        }
        return os_info
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system info"""
        config_manager = get_config_manager()
        system_config = config_manager.get_config("system", {"name": "CamThink AI Inference Platform", "version": "1.0.0"})
        
        gpu_name_list = ["CPU"] #default CPU
        gpu_metrics = self.collect_gpu_metrics()
        gpu_usage = gpu_metrics[0].get("load_percent", "N/A") if gpu_metrics else "N/A"
        if gpu_usage != "N/A":
            gpu_name_list.extend([gpu["name"] for gpu in gpu_metrics])

        
        system_info = {
            "system_name": "Neo Edge NG4500", #default for now
            "hardware_acceleration": gpu_name_list,  
            "os_info": self.collect_os_info(),
            "software_name": system_config["name"],
            "software_version": system_config["version"],
            "models_stats": self.collect_model_status(), 
            "access_tokens_stats": self.collect_access_token_stats()
        }
        logger.info(f"Collected system info: {system_info}")
        return system_info
    
    def save_metrics(self, metrics_to_save: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save metrics to a JSON file with file locking to prevent conflicts.
        If metrics_to_save is provided, save it; otherwise save self.metrics.
        """
        if metrics_to_save is None:
            metrics_to_save = self.metrics
        
        if not metrics_to_save:
            logger.warning("No metrics to save.")
            return False

        try:
            if not self.output_file.parent.exists():
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_file, 'a+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    f.seek(0)
                    content = f.read()
                    
                    existing_metrics_list: List[Dict[str, Any]] = []
                    if content.strip():
                        try:
                            existing_metrics_list = json.loads(content)
                            if not isinstance(existing_metrics_list, list):
                                logger.warning(f"Metrics file {self.output_file} did not contain a list. Resetting.")
                                existing_metrics_list = []
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON from {self.output_file}. File might be corrupted. Resetting metrics list.")
                            existing_metrics_list = []
                    
                    existing_metrics_list.append(metrics_to_save)
        
                    max_records = int(24 * 60 * 60 / DEFAULT_COLLECTION_INTERVAL)  # lastest 24h
                    if len(existing_metrics_list) > max_records:
                        existing_metrics_list = existing_metrics_list[-max_records:]
                    
                
                    f.truncate(0)
                    f.seek(0)
                    json.dump(existing_metrics_list, f, indent=4)
                    
                    logger.info(f"Metrics saved to {self.output_file}")
                    return True
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
        except Exception as e:
            logger.error(f"Error saving metrics to {self.output_file}: {e}", exc_info=True)
            return False

    #read the lastest 24h metrics
    def read_metrics(self) -> List[Dict[str, Any]]:
        """Read metrics from the JSON file with file locking to prevent conflicts"""
        metrics_list = []
        
        try:
            if not self.output_file.exists():
                logger.warning(f"Metrics file {self.output_file} does not exist.")
                return []
            
            with open(self.output_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read()
                    if content.strip():
                        metrics_list = json.loads(content)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.output_file}. File might be corrupted.")
        except Exception as e:
            logger.error(f"Error reading metrics from {self.output_file}: {e}", exc_info=True)
        
        return metrics_list
    
    
    # run() method can be kept
    def run_continuous(self, interval: int = DEFAULT_COLLECTION_INTERVAL):
        """Run the monitor continuously at a specified interval and save metrics."""
        logger.info(f"Starting continuous system monitoring (interval: {interval}s). Output to: {self.output_file}")
        try:
            while True:
                try:
                    current_metrics = self.collect_all_metrics()

                    timestamp = current_metrics.get("timestamp", "N/A")
                    cpu_usage = current_metrics.get("cpu", {}).get("cpu_percent", "N/A")
                    mem_usage = current_metrics.get("memory", {}).get("virtual_memory_used", "N/A")
                    gpu_metrics = current_metrics.get("gpu", [{"load_percent": "N/A"}])
                    #default 0
                    gpu_usage = gpu_metrics[0].get("load_percent", "N/A") if gpu_metrics else "N/A"

                    view_metrics = {
                        "timestamp": timestamp,
                        "cpu_usage": cpu_usage,
                        "mem_usage": mem_usage,
                        "gpu_usage": gpu_usage
                    }


                    save_success = self.save_metrics(view_metrics)
                    if save_success:
                        logger.info(f"View metrics: {view_metrics}")
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Continuous monitoring stopped by user.")
        except Exception as e:
            logger.error(f"Fatal error in continuous monitoring loop: {e}", exc_info=True)
        finally:
            if self.db_service and hasattr(self.db_service, 'is_internally_managed') and self.db_service.is_internally_managed:
                 self.db_service.disconnect()
                 logger.info("Database connection closed by SystemMonitor.")



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running SystemMonitor as a standalone script (for continuous collection or testing).")
    
    # When running independently, the script itself is responsible for database configuration and connection
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "db" / "cinfer.db"
    
    current_db_config: Dict[str, Any] = {"type": "sqlite", "path": str(db_path)}
    logger.info(f"Standalone run using DB config: {current_db_config}")
    try:
        db_service_instance = DatabaseFactory.create_database(current_db_config)
        if not db_service_instance.connect(): 
            logger.error("Failed to connect to the database for standalone monitor run. Exiting.")
            sys.exit(1) 
        logger.info("Database connected successfully for standalone monitor run.")
      
        monitor_instance = SystemMonitor(db_service=db_service_instance) 
        
        monitor_instance.run_continuous(interval=DEFAULT_COLLECTION_INTERVAL)

    except Exception as e:
        logger.error(f"An error occurred during standalone monitor execution: {e}", exc_info=True)
    finally:
        if db_service_instance and getattr(db_service_instance, 'conn', None):
            db_service_instance.disconnect()
            logger.info("Database connection closed after standalone monitor run.")