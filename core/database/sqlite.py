# cinfer/core/database/sqlite.py
import sqlite3
import os
import json # For handling TEXT fields that store JSON
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

from .base import DatabaseService
# Assuming ConfigManager is accessible for DB path, or pass path directly
# from core.config import get_config_manager

# config = get_config_manager()
# DEFAULT_DB_PATH = config.get_config("database.sqlite.path", "data/cinfer.db")



def dict_factory(cursor, row: Tuple) -> Dict[str, Any]:
    """Converts a database row (tuple) to a dictionary with column names as keys."""
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}

class SQLiteDatabase(DatabaseService):
    """
    SQLite implementation of the DatabaseService.
    Handles database operations for an SQLite database.
    The design document specifies SQLite as the primary database for Cinfer.
    """
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initializes the SQLiteDatabase.
        Args:
            db_config (Dict[str, Any]): Configuration dictionary.
                                         Expected key: 'db_path' (e.g., "data/db/cinfer.db").
        """
        self.db_path = db_config.get("path", "data/db/default_cinfer.db") # Default path if not in config
        self.conn: Optional[sqlite3.Connection] = None
        logger.info(f"SQLiteDatabase initialized with db_path: {self.db_path}")
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensures the directory for the SQLite database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}") 

    def connect(self) -> bool:
        """Establishes a connection to the SQLite database file."""
        if self.conn is not None:
            logger.info("Already connected.") 
            return True
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # check_same_thread for FastAPI
            self.conn.row_factory = dict_factory # Makes fetchone/fetchall return dicts
            # Enable Foreign Key support if not enabled by default (good practice)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            logger.info(f"Successfully connected to SQLite database: {self.db_path}") 
            return True
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database: {e}") 
            self.conn = None
            return False

    def disconnect(self) -> bool:
        """Closes the SQLite database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logger.info("Successfully disconnected from SQLite database.") 
                return True
            except sqlite3.Error as e:
                logger.error(f"Error disconnecting from SQLite database: {e}") 
                return False
        return True # Already disconnected

    def _execute(self, query: str, params: Optional[tuple] = None) -> Optional[sqlite3.Cursor]:
        """Helper method to execute a query and handle connection."""
        if not self.conn:
            # Attempt to reconnect if connection is lost or not established
            if not self.connect():
                logger.error("Failed to execute query: No database connection.") 
                return None
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            return cursor
        except sqlite3.Error as e:
            logger.error(f"SQLite execution error: {e} (Query: {query[:100]}...)" ) 
            # Consider if self.conn.rollback() is needed for some errors
            return None

    def _serialize_if_needed(self, value: Any) -> Any:
        """Serializes complex types (dict, list) to JSON strings for TEXT columns."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return value

    def insert(self, table: str, data: Dict[str, Any]) -> Optional[str]:
        """Inserts a record. Handles JSON serialization for dict/list values."""
        if not data:
            return None
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        # Serialize values that are dicts or lists to JSON strings
        values = tuple(self._serialize_if_needed(v) for v in data.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        cursor = self._execute(query, values)
        if cursor and self.conn:
            try:
                self.conn.commit()
                # Assuming 'id' is typically the primary key generated or provided
                return str(cursor.lastrowid) if cursor.lastrowid else data.get("id")
            except sqlite3.Error as e:
                logger.error(f"Error committing insert: {e}") 
                if self.conn: self.conn.rollback()
                return None
        return None


    def find_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Finds a single record."""
        if not filters: # Finding one without filters is ambiguous, usually needs an ID
            logger.warning("Warning: find_one called without filters. Returning first record.") 
            query = f"SELECT * FROM {table} LIMIT 1"
            params = ()
        else:
            where_clauses = [f"{key} = ?" for key in filters.keys()]
            where_conditions = " AND ".join(where_clauses)
            query = f"SELECT * FROM {table} WHERE {where_conditions} LIMIT 1"
            params = tuple(filters.values())

        cursor = self._execute(query, params)
        if cursor:
            row = cursor.fetchone()
            return row if row else None
        return None

    def find(self, table: str, filters: Dict[str, Any] = None,
             order_by: Optional[str] = None, limit: Optional[int] = None,
             offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Finds multiple records with optional ordering, limit, and offset."""
        base_query = f"SELECT * FROM {table}"
        conditions = []
        params_list = []

        if filters:
            for key, value in filters.items():
                if isinstance(value, list): # Handle 'IN' clauses for lists
                    if not value: continue # Skip empty lists
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"{key} IN ({placeholders})")
                    params_list.extend(value)
                else:
                    conditions.append(f"{key} = ?")
                    params_list.append(value)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        if order_by: # Basic sanitation for order_by to prevent injection with column names
            # A more robust solution would validate 'order_by' against known columns/formats
            safe_order_by = "".join(c for c in order_by if c.isalnum() or c in ('_', ' ', ',', 'ASC', 'DESC'))
            if safe_order_by:
                 base_query += f" ORDER BY {safe_order_by}"

        if limit is not None:
            base_query += " LIMIT ?"
            params_list.append(limit)
        
        if offset is not None:
            if limit is None: # SQLite requires LIMIT with OFFSET
                base_query += " LIMIT -1" # effectively no limit
            base_query += " OFFSET ?"
            params_list.append(offset)

        cursor = self._execute(base_query, tuple(params_list))
        if cursor:
            return cursor.fetchall()
        return []

    def update(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """Updates records. Handles JSON serialization for dict/list values."""
        if not filters or not updates:
            return 0
        
        set_clauses = [f"{key} = ?" for key in updates.keys()]
        set_conditions = ", ".join(set_clauses)
        
        # Serialize update values that are dicts or lists
        update_values = [self._serialize_if_needed(v) for v in updates.values()]

        where_clauses = [f"{key} = ?" for key in filters.keys()]
        where_conditions = " AND ".join(where_clauses)
        
        query = f"UPDATE {table} SET {set_conditions} WHERE {where_conditions}"
        params = tuple(update_values) + tuple(filters.values())
        
        cursor = self._execute(query, params)
        if cursor and self.conn:
            try:
                self.conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Error committing update: {e}") 
                if self.conn: self.conn.rollback()
                return 0
        return 0

    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        """Deletes records."""
        if not filters:
            logger.error("Error: delete operation requires filters.") 
            return 0 # Or raise an error, deleting without WHERE is dangerous

        where_clauses = [f"{key} = ?" for key in filters.keys()]
        where_conditions = " AND ".join(where_clauses)
        query = f"DELETE FROM {table} WHERE {where_conditions}"
        params = tuple(filters.values())

        cursor = self._execute(query, params)
        if cursor and self.conn:
            try:
                self.conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Error committing delete: {e}") 
                if self.conn: self.conn.rollback()
                return 0
        return 0

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[Any]:
        """Executes an arbitrary SQL query."""
        cursor = self._execute(query, params)
        if cursor:
            # For SELECT queries, try to fetch all results
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            # For DML queries (INSERT, UPDATE, DELETE), commit and return rowcount
            elif any(keyword in query.strip().upper() for keyword in ["INSERT", "UPDATE", "DELETE", "REPLACE"]):
                if self.conn:
                    try:
                        self.conn.commit()
                        return cursor.rowcount
                    except sqlite3.Error as e:
                        logger.error(f"Error committing query: {e}") 
                        if self.conn: self.conn.rollback()
                        return None
            # For other DDL or PRAGMA, just return True if execution was successful
            return True # Or cursor itself for more control
        return None

    def execute_script(self, script: str) -> bool:
        """Executes a script containing multiple SQL statements."""
        if not self.conn:
            if not self.connect():
                logger.error("Failed to execute script: No database connection.") 
                return False
        try:
            self.conn.executescript(script)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"SQLite script execution error: {e}") 
            if self.conn: self.conn.rollback()
            return False