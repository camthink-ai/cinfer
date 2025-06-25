# cinfer/models/model_store.py
import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional

from core.config import ConfigManager

logger = logging.getLogger(f"cinfer.{__name__}")

class ModelStore:
    """
    Manages the physical storage of AI model files.
    Handles saving, retrieving file paths, and deleting model files.
    As per document section 4.2.1, 4.2.2.
    """
    def __init__(self, config_manager: ConfigManager):
        self._config_manager = config_manager
        # Default storage path from config, e.g., "data/models"
        self._storage_path_str = self._config_manager.get_config("models.storage_path", "data/models")
        self._storage_path = Path(self._storage_path_str)
        self._ensure_storage_directory()

    def _ensure_storage_directory(self):
        """Ensures the base storage directory exists."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model storage directory ensured at: {self._storage_path}")

    def _get_model_dir(self, model_id: str) -> Path:
        """Generates the directory path for a model"""
        return self._storage_path / model_id 

    def save_model_file(self, temp_file_path: str, model_id: str, original_filename: str) -> Optional[str]:
        """
        Saves a model file from a temporary location to the persistent model store.
        The actual stored filename might be normalized or kept original.
        Args:
            temp_file_path (str): Path to the temporary uploaded model file.
            model_id (str): The unique ID of the model.
            original_filename (str): The original filename of the uploaded model.
        Returns:
            Optional[str]: The relative path to the saved model file within the storage base,
                           or None if saving failed.
        """
        if not os.path.exists(temp_file_path):
            logger.error(f"Error: Temporary model file not found at {temp_file_path}")
            return None

        model_id_dir = self._get_model_dir(model_id)
        model_id_dir.mkdir(parents=True, exist_ok=True)
        
        # Using original filename within the model directory
        # Alternatively, could use a standard name like 'model.onnx'
        # For simplicity, we use original_filename here. Ensure it's sanitized if needed.
        # Sanitize filename to prevent directory traversal or invalid characters
        safe_filename = Path(original_filename).name # Basic sanitization
        destination_file_path = model_id_dir / safe_filename

        try:
            shutil.move(temp_file_path, destination_file_path)
            logger.info(f"Model file '{original_filename}' saved to {destination_file_path}")
            # Return path relative to the base storage path for storing in DB
            relative_path = destination_file_path.relative_to(self._storage_path).as_posix()
            return relative_path
        except Exception as e:
            logger.error(f"Error saving model file from {temp_file_path} to {destination_file_path}: {e}")
            # Clean up if partial save occurred (though shutil.move is mostly atomic)
            if destination_file_path.exists():
                destination_file_path.unlink(missing_ok=True)
            return None
        
    def save_yaml_file(self, params_yaml: str, model_id: str) -> Optional[str]:
        """
        Saves a yaml file from a temporary location to the persistent model store.
        """
        model_id_dir = self._get_model_dir(model_id)
        model_id_dir.mkdir(parents=True, exist_ok=True)

        # Using original filename within the model directory
        # Alternatively, could use a standard name like 'model.onnx'
        # For simplicity, we use original_filename here. Ensure it's sanitized if needed.
        # Sanitize filename to prevent directory traversal or invalid characters    
        safe_filename = "params.yaml" # Basic sanitization
        destination_file_path = model_id_dir / safe_filename

        try:
            with open(destination_file_path, 'w') as f:
                f.write(params_yaml)
            logger.info(f"Yaml file params.yaml saved to {destination_file_path}")
            # Return path relative to the base storage path for storing in DB
            relative_path = destination_file_path.relative_to(self._storage_path).as_posix()
            return relative_path
        except Exception as e:
            logger.error(f"Error saving yaml file from {params_yaml} to {destination_file_path}: {e}")
            return None
        
    def get_model_file_path(self, relative_model_path: str) -> Optional[str]:
        """
        Gets the absolute path to a model file given its relative path from the DB.
        Args:
            relative_model_path (str): The relative path of the model file as stored in DB.
        Returns:
            Optional[str]: The absolute path to the model file, or None if not found.
        """
        if not relative_model_path:
            return None
        absolute_path = self._storage_path / relative_model_path
        if absolute_path.exists() and absolute_path.is_file():
            return str(absolute_path)
        logger.error(f"Model file not found at resolved path: {absolute_path}")
        return None
    
    def get_yaml_file_path(self, relative_yaml_path: str) -> Optional[str]:
        """
        Gets the absolute path to a yaml file given its relative path from the DB.
        """
        if not relative_yaml_path:
            return None
        absolute_path = self._storage_path / relative_yaml_path
        if absolute_path.exists() and absolute_path.is_file():
            return str(absolute_path)
        logger.error(f"Yaml file not found at resolved path: {absolute_path}")
        return None
    
    def read_yaml_from_file(self, yaml_file_path: str) -> Optional[str]:
        """
        Reads a yaml file and returns its content as a string.
        """
        absolute_path = self.get_yaml_file_path(yaml_file_path)
        if absolute_path:
            with open(absolute_path, 'r') as f:
                return f.read()
        return None
        
    def delete_model_and_yaml_file(self, relative_model_path: str, relative_yaml_path: Optional[str] = None) -> bool:
        """
        Deletes a model file and its yaml file given their relative paths.
        This might also involve deleting the model_id directory if it becomes empty.
        Args:
            relative_model_path (str): The relative path of the model file.
            relative_yaml_path (str): The relative path of the yaml file.
        Returns:
            bool: True if deletion was successful or file didn't exist, False on error.
        """
        absolute_model_path = self.get_model_file_path(relative_model_path)
        absolute_yaml_path = self.get_yaml_file_path(relative_yaml_path)
        if absolute_model_path:
            try:
                model_file_path = Path(absolute_model_path)
                model_file_path.unlink() # Delete the file
                logger.info(f"Model file deleted: {model_file_path}")
                if absolute_yaml_path:
                    yaml_file_path = Path(absolute_yaml_path)
                    yaml_file_path.unlink()
                    logger.info(f"Yaml file deleted: {yaml_file_path}")

                # Try to remove model_id directory if empty
                model_id_dir = model_file_path.parent
                if model_id_dir.exists() and not any(model_id_dir.iterdir()):
                    model_id_dir.rmdir()
                    logger.info(f"Empty model_id directory deleted: {model_id_dir}")
                return True
            except Exception as e:
                logger.error(f"Error deleting model file {absolute_model_path}: {e}")
                return False
        return True # File didn't exist, considered success for deletion intent
    
    def delete_yaml_file(self, relative_yaml_path: str) -> bool:
        """
        Deletes a yaml file given its relative path.
        """
        absolute_path = self.get_yaml_file_path(relative_yaml_path)
        if absolute_path:
            try:
                yaml_file_path = Path(absolute_path)
                yaml_file_path.unlink()
                logger.info(f"Yaml file deleted: {yaml_file_path}")
                return True
            except Exception as e:
                logger.error(f"Error deleting yaml file {absolute_path}: {e}")
                return False
        return True # File didn't exist, considered success for deletion intent