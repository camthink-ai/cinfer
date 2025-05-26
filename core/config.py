# cinfer/core/config.py

import yaml
import os
from typing import Any, Dict, Callable, List
import logging
logger = logging.getLogger(f"cinfer.{__name__}")
# for placeholder, actually can import more robust schema validation library, like Pydantic
class ValidationResult:
    def __init__(self, valid: bool, errors: List[str] = None):
        self.valid = valid
        self.errors = errors or []

    def __bool__(self):
        return self.valid

class ConfigValidator:
    """
    Validate the configuration.
    Based on 4.5.2, this class is based on JSON Schema configuration validation, 
    but here simplified to basic checks.
    """
    def __init__(self):
        # _schemas can be used to store the expected structure or type of each configuration part
        self._schemas: Dict[str, Any] = {
            "server": dict,
            "engines": dict,
            "models": dict,
            "request": dict,
            "auth": dict
        } # based on structure in 4.5.3

    def add_schema(self, name: str, schema: Dict):
        """Add a schema for a configuration part (simplified version)"""
        self._schemas[name] = schema

    def validate(self, config: Dict) -> ValidationResult:
        """
        Validate the overall configuration.
        In actual implementation, this will perform more complex schema validation.
        """
        errors = []
        for key, expected_type in self._schemas.items():
            if key not in config:
                errors.append(f"Configuration is missing the key: '{key}'")
            elif not isinstance(config[key], expected_type):
                errors.append(f"Configuration part '{key}' has the wrong type, expected {expected_type}, got {type(config[key])}")
        
        # example: check if 'server' config contains 'port'
        if "server" in config and isinstance(config["server"], dict):
            if "port" not in config["server"]:
                errors.append("Server configuration 'server' is missing 'port'")
        else:
            if "server" not in errors: # avoid duplicate report of missing server
                 errors.append("Configuration is missing 'server' part or its type is incorrect")


        if errors:
            return ValidationResult(False, errors)
        return ValidationResult(True)


class ConfigLoader:
    """
    Configuration loading interface.
    Based on 4.5.2, define the configuration loading interface.
    """
    def load(self) -> Dict:
        raise NotImplementedError

    def supports_reload(self) -> bool:
        return False

    def reload(self) -> Dict:
        raise NotImplementedError("Reloading not supported by this loader.")


class FileConfigLoader(ConfigLoader):
    """
    Load configuration from a configuration file (YAML/JSON).
    Based on 4.5.2, implement the configuration loading from a file.
    """
    def __init__(self, file_path: str, file_format: str = "yaml"):
        self._file_path = file_path
        self._format = file_format.lower()
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"Configuration file not found: {self._file_path}")

    def load(self) -> Dict:
        """Load configuration from a file"""
        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                if self._format == "yaml":
                    return yaml.safe_load(f)
                elif self._format == "json":
                    import json
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {self._format}")
        except Exception as e:
            # In actual project, this should use the logging module to record the error
            logger.error(f"Failed to load configuration file '{self._file_path}': {e}")
            raise

    def supports_reload(self) -> bool:
        """FileConfigLoader can support reloading (by re-reading the file)"""
        return True # based on FileConfigLoader description in 4.5.2

    def reload(self) -> Dict:
        """Reload the configuration file"""
        logger.info(f"Reloading configuration file: {self._file_path}") # better use logging
        return self.load()

    def detect_changes(self) -> bool:
        """
        Detect if the file has changed.
        Simplified implementation: actually should compare file modification time or content hash.
        This feature is mentioned in 4.5.2 FileConfigLoader.
        """
        # Here is a simplified version, in actual application, a more reliable detection mechanism is needed
        logger.warning("Warning: detect_changes is a simplified implementation.")
        return False # default to unchanged, to avoid unnecessary reload


class ConfigManager:
    """
    Central coordinator for the configuration management module.
    Based on 4.5.1, responsible for loading, validating, and accessing system configurations.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, loader: ConfigLoader = None, validator: ConfigValidator = None):
        # Prevent duplicate initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._config_data: Dict = {}
        self._loader = loader
        self._validator = validator or ConfigValidator()
        self._listeners: Dict[str, List[Callable]] = {}
        self._initialized = False # Mark if initialized

        if self._loader:
            self.load_config()
            self._initialized = True

    def load_config(self) -> bool:
        """Load configuration, if a loader is provided."""
        if not self._loader:
            logger.error("Error: No configuration loader (ConfigLoader) provided.") # better use logging
            return False
        try:
            new_config = self._loader.load()
            validation_result = self._validator.validate(new_config)
            if not validation_result:
                logger.error(f"Configuration validation failed: {validation_result.errors}") # better use logging
                # Optionally, raise an exception or keep the old configuration (if supported)
                return False
            self._config_data = new_config
            logger.info("Configuration loaded and validated successfully.") # better use logging
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}") # better use logging
            return False

    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration item by path, e.g. "server.port".
        """
        if not self._config_data:
            # It might be because the loading failed or not loaded
            logger.warning("Warning: Configuration data is empty. Possibly not loaded or loading failed.") # better use logging
            return default
            
        keys = path.split('.')
        value = self._config_data
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else: # if the intermediate path is not a dictionary, continue
                    return default
            return value
        except KeyError:
            return default
        except Exception as e:
            logger.error(f"Error getting configuration '{path}': {e}") # better use logging
            return default

    def set_config(self, path: str, value: Any) -> bool:
        """
        Set a configuration item and notify listeners.
        Note: This method modifies the configuration in memory and attempts to save (if the loader supports it).
        """
        keys = path.split('.')
        data_ref = self._config_data
        try:
            for key in keys[:-1]:
                if key not in data_ref or not isinstance(data_ref[key], dict):
                    data_ref[key] = {} # if the path does not exist, create it
                data_ref = data_ref[key]
            
            old_value = data_ref.get(keys[-1])
            data_ref[keys[-1]] = value
            
            # Validate the modified configuration
            validation_result = self._validator.validate(self._config_data)
            if not validation_result:
                logger.error(f"Configuration validation failed after setting '{path}': {validation_result.errors}") # better use logging
                # Rollback the changes
                if old_value is None: # it did not exist before
                    del data_ref[keys[-1]]
                else:
                    data_ref[keys[-1]] = old_value
                return False

            logger.info(f"Configuration item '{path}' has been updated to: {value}") # better use logging
            self._notify_listeners(path, value)

            # Try to save the configuration (simplified, actual saving logic should be in FileConfigLoader)
            if isinstance(self._loader, FileConfigLoader):
                try:
                    # FileConfigLoader needs a save method
                    # self._loader.save(self._config_data)
                    logger.info(f"Note: Configuration has been updated in memory. To persist, implement FileConfigLoader.save() and call it.")
                except AttributeError:
                    logger.warning(f"Warning: Current FileConfigLoader does not support saving configuration.")
                except Exception as e:
                    logger.error(f"Failed to save configuration: {e}")

            return True
        except Exception as e:
            logger.error(f"Error setting configuration '{path}': {e}") # better use logging
            return False


    def register_listener(self, path: str, callback: Callable):
        """
        Register a function to listen for changes in a specific configuration path.
        """
        if path not in self._listeners:
            self._listeners[path] = []
        self._listeners[path].append(callback)
        logger.info(f"Listener registered for path: {path}")

    def _notify_listeners(self, path: str, value: Any):
        """Notify all listeners matching the path"""
        for listener_path, callbacks in self._listeners.items():
            # Support wildcard or parent path matching (simplified to exact match and parent path)
            if path == listener_path or path.startswith(listener_path + "."):
                for callback in callbacks:
                    try:
                        callback(path, value)
                    except Exception as e:
                        # Use logging to record callback errors
                        logger.error(f"Error executing configuration listener callback (path: {path}): {e}")
    
    def reload_config_from_source(self) -> bool:
        """
        Reload configuration from the source (e.g. file).
        Only effective when the loader supports reloading.
        """
        if not self._loader or not self._loader.supports_reload():
            logger.warning("Warning: Current configuration loader does not support reloading.")
            return False
        
        try:
            logger.info("Attempting to reload configuration...")
            new_config = self._loader.reload()
            validation_result = self._validator.validate(new_config)
            if not validation_result:
                logger.error(f"Reloaded configuration validation failed: {validation_result.errors}")
                return False
            
            self._config_data = new_config
            logger.info("Configuration reloaded and validated successfully.")
            # For simplicity, we might re-notify all listeners or specific ones
            # based on what changed, which can be complex.
            # Here, we assume a general notification might be needed for some root paths.
            self._notify_listeners("", self._config_data) # Notify with root and new data
            return True
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False

# Global configuration instance (simplified access to singleton pattern)
# Initialize when the application starts
# config_manager = ConfigManager()

def get_config_manager(config_file_path: str = None) -> ConfigManager:
    """
    Get the singleton instance of ConfigManager.
    If first call, provide the configuration file path for initialization.
    """
    if ConfigManager._instance is None or not ConfigManager._instance._initialized:
        if config_file_path is None:
            # Try to get from environment variable or use default path
            config_file_path = os.getenv("CINFER_CONFIG_PATH", "config/config.yaml")
            # Ensure path exists, if not, handle it
            if not os.path.exists(config_file_path) :
                 # Create a default, minimal configuration file if it does not exist
                default_config_dir = os.path.dirname(config_file_path)
                if default_config_dir and not os.path.exists(default_config_dir):
                    os.makedirs(default_config_dir, exist_ok=True)
                
                logger.warning(f"Warning: Configuration file '{config_file_path}' not found. Will try to use a minimal default configuration.")
                default_cfg_content = { # Mirrored from 4.5.3 and 8.2
                    "server": {"host": "0.0.0.0", "port": 8000, "workers": 1, "log_level": "info"},
                    "engines": {"default": "onnx", "onnx": {"enabled": True, "execution_providers": ["CPUExecutionProvider"], "threads": 1}},
                    "models": {"storage_path": "data/models", "max_file_size_mb": 100},
                    "request": {"queue_size": 50, "workers_per_model": 1, "timeout_ms": 5000},
                    "auth": {"token_expiry_days": 30, "rate_limit": {"requests_per_minute": 60}, "ip_filter": {"enabled": False}}
                }
                try:
                    with open(config_file_path, 'w', encoding='utf-8') as f_default:
                        yaml.dump(default_cfg_content, f_default, default_flow_style=False)
                    logger.info(f"Default configuration file '{config_file_path}' has been created.")
                except Exception as e_create:
                    logger.error(f"Failed to create default configuration file '{config_file_path}': {e_create}")
                    # If cannot create default config, cannot continue, raise an exception or return an empty ConfigManager
                    raise RuntimeError(f"Failed to load or create configuration file: {config_file_path}") from e_create

        loader = FileConfigLoader(file_path=config_file_path)
        ConfigManager(loader=loader) # Initialize singleton
    
    return ConfigManager._instance

# Example usage (typically called at application entry point):
# if __name__ == "__main__":
#     # Ensure config/config.yaml file exists and content matches 8.2 in doc
#     # For example, first create a sample config/config.yaml
#     sample_config_content = """
# server:
#   host: "0.0.0.0"
#   port: 8000
#   workers: 4
#   log_level: "info"
# engines:
#   default: "onnx"
#   onnx:
#     enabled: true
#     execution_providers: ["CPUExecutionProvider"]
#     threads: 4
#   tensorrt:
#     enabled: false
#     precision: "fp16"
#   pytorch:
#     enabled: true
#     device: "cpu"
# models:
#   storage_path: "data/models"
#   max_file_size_mb: 100
# request:
#   queue_size: 100
#   workers_per_model: 2
#   timeout_ms: 5000
# auth:
#   token_expiry_days: 30
#   rate_limit:
#     requests_per_minute: 60
#     requests_per_month: 10000
#   ip_filter:
#     enabled: true
#     """
#     os.makedirs("config", exist_ok=True)
#     with open("config/config.yaml", "w") as f:
#         f.write(sample_config_content)

#     manager = get_config_manager("config/config.yaml")
    
#     print(f"Server port: {manager.get_config('server.port')}")
#     print(f"Default engine: {manager.get_config('engines.default')}")
#     print(f"Non-existent config: {manager.get_config('nonexistent.key', 'default_value')}")

#     def my_listener(path, value):
#         print(f"Configuration change listener: path='{path}', value='{value}'")

#     manager.register_listener("server.port", my_listener)
#     manager.set_config("server.port", 8081)
#     print(f"Updated server port: {manager.get_config('server.port')}")

#     # Simulate file change and reload (need to manually modify config.yaml file and run)
#     # input("Modify config.yaml server.port and press Enter to test reload...")
#     # manager.reload_config_from_source()
#     # print(f"Reloaded server port: {manager.get_config('server.port')}")