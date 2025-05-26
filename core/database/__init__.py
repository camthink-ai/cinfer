# cinfer/core/database/__init__.py
from typing import Dict, Any
from .base import DatabaseService
from .sqlite import SQLiteDatabase
# Import PostgreSQLDatabase when it's created
# from .postgresql import PostgreSQLDatabase

class DatabaseFactory:
    """
    Database factory to create different database instances based on configuration.
    As described in document section 5.6 .
    """

    @staticmethod
    def create_database(config: Dict[str, Any]) -> DatabaseService:
        """
        Creates a database service instance based on the provided configuration.
        Args:
            config: Database configuration dictionary.
                    Expected keys: "type" (e.g., "sqlite", "postgresql"),
                                   and other type-specific settings.
        Returns:
            DatabaseService: An instance of a database service.
        Raises:
            ValueError: If the specified database type is not supported.
        """
        db_type = config.get("type", "sqlite").lower() # Default to sqlite if not specified

        if db_type == "sqlite":
            # The SQLiteDatabase expects its specific config, e.g., {'path': '...'}
            # The main app config might have: database: {'type': 'sqlite', 'path': 'data/cinfer.db'}
            return SQLiteDatabase(config) # Pass the whole db_config section
        # elif db_type == "postgresql":
        #     return PostgreSQLDatabase(config) # Pass the whole db_config section
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

__all__ = [
    "DatabaseService",
    "SQLiteDatabase",
    "DatabaseFactory",
    # "PostgreSQLDatabase", # Add when implemented
]