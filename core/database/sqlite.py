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
    """
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initializes the SQLiteDatabase.

        Args:
            db_config (Dict[str, Any]): Configuration dictionary.
                                         Expected key: 'path' (e.g., "data/db/cinfer.db").
        """
        self.db_path: str = db_config.get("path", "data/db/default_cinfer.db") # Default path if not in config
        self.conn: Optional[sqlite3.Connection] = None
        logger.info(f"SQLiteDatabase initialized with db_path: {self.db_path}")
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensures the directory for the SQLite database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir): # Ensure db_dir is not empty string if db_path is just a filename
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")
            except OSError as e:
                logger.error(f"Failed to create database directory {db_dir}: {e}", exc_info=True)


    def connect(self) -> bool:
        """Establishes a connection to the SQLite database file."""
        if self.conn is not None:
            logger.debug("Already connected to SQLite database.")
            return True
        try:
            # Using check_same_thread=False is common for web apps like FastAPI,
            # but be aware of SQLite's threading limitations if using it heavily concurrently.
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = dict_factory # Makes fetchone/fetchall return dicts
            self.conn.execute("PRAGMA foreign_keys = ON;") # Enable Foreign Key support
            logger.info(f"Successfully connected to SQLite database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database {self.db_path}: {e}", exc_info=True)
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
                logger.error(f"Error disconnecting from SQLite database: {e}", exc_info=True)
                return False
        logger.debug("Already disconnected or no active connection.")
        return True # Considered success if already disconnected

    def _execute(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[sqlite3.Cursor]:
        """
        Helper method to execute a query and handle connection.
        Ensures connection is active before execution.

        Args:
            query (str): The SQL query string.
            params (Optional[tuple]): Parameters to bind to the query.

        Returns:
            Optional[sqlite3.Cursor]: The cursor object if execution was successful, None otherwise.
        """
        if not self.conn:
            logger.warning("No active database connection. Attempting to connect.")
            if not self.connect():
                logger.error("Failed to execute query: Could not establish database connection.")
                return None
        try:
            cursor = self.conn.cursor() # type: ignore # self.conn is checked
            logger.debug(f"Executing SQL: {query} with params: {params}")
            cursor.execute(query, params or ())
            return cursor
        except sqlite3.Error as e:
            logger.error(f"SQLite execution error: {e} (Query: {query[:200]}..., Params: {params})", exc_info=True)
            # Consider if self.conn.rollback() is needed for some errors if transactions are managed here.
            # For now, assuming transactions are managed by calling methods (insert, update, delete).
            return None

    def _serialize_if_needed(self, value: Any) -> Any:
        """Serializes complex types (dict, list) to JSON strings for TEXT columns."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return value

    def _build_filter_conditions(self, filters: Optional[Dict[str, Any]]) -> Tuple[List[str], List[Any]]:
        """
        Builds WHERE clause conditions and corresponding parameters from a filter dictionary.
        Supports operators via suffixes: __ne, __in, __nin, __lt, __lte, __gt, __gte, __like, __isnull.
        Defaults to '=' if no operator suffix is present. Column names are backticked.

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary of filters.
                Keys can be 'column_name' or 'column_name__operator'.
                Example: {"name__like": "%test%", "status__in": ["active", "pending"], "age__gt": 30}

        Returns:
            Tuple[List[str], List[Any]]: A list of condition strings and a list of parameters.
        """
        conditions: List[str] = []
        params_list: List[Any] = []

        if not filters:
            return conditions, params_list

        for key_with_op, value in filters.items():
            parts = key_with_op.split('__')
            column_name = f"`{parts[0]}`" # Backtick column name for safety
            op_suffix = parts[1].lower() if len(parts) > 1 else None

            if op_suffix == 'ne':
                conditions.append(f"{column_name} != ?") # Or use <>
                params_list.append(self._serialize_if_needed(value))
            elif op_suffix == 'in' or op_suffix == 'nin':
                if not isinstance(value, (list, tuple)) or not value: # Must be a non-empty list/tuple
                    logger.warning(f"Filter '{key_with_op}' for IN/NOT IN expects a non-empty list/tuple, got {type(value)}. Skipping.")
                    continue
                placeholders = ', '.join(['?'] * len(value))
                sql_operator = "IN" if op_suffix == 'in' else "NOT IN"
                conditions.append(f"{column_name} {sql_operator} ({placeholders})")
                params_list.extend(self._serialize_if_needed(v) for v in value)
            elif op_suffix == 'lt':
                conditions.append(f"{column_name} < ?")
                params_list.append(value)
            elif op_suffix == 'lte':
                conditions.append(f"{column_name} <= ?")
                params_list.append(value)
            elif op_suffix == 'gt':
                conditions.append(f"{column_name} > ?")
                params_list.append(value)
            elif op_suffix == 'gte':
                conditions.append(f"{column_name} >= ?")
                params_list.append(value)
            elif op_suffix == 'like':
                conditions.append(f"{column_name} LIKE ?")
                params_list.append(value) # User must provide '%' wildcards in the value
            elif op_suffix == 'isnull':
                if isinstance(value, bool):
                    conditions.append(f"{column_name} IS {'' if value else 'NOT '}NULL")
                    # No parameter needed for IS NULL / IS NOT NULL
                else:
                    logger.warning(f"Filter '{key_with_op}' for ISNULL expects a boolean value, got {type(value)}. Defaulting to equality check.")
                    conditions.append(f"{column_name} = ?") # Fallback or raise error
                    params_list.append(self._serialize_if_needed(value))
            elif op_suffix is None: # No operator suffix, default to equality
                conditions.append(f"{column_name} = ?")
                params_list.append(self._serialize_if_needed(value))
            else: # Unknown operator suffix
                logger.warning(f"Unknown operator suffix in filter key '{key_with_op}'. Defaulting to equality for column '{parts[0]}'.")
                conditions.append(f"{column_name} = ?") # Fallback or raise error
                params_list.append(self._serialize_if_needed(value))
                
        return conditions, params_list

    def insert(self, table: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Inserts a record into the specified table.
        Handles JSON serialization for dictionary/list values.

        Args:
            table (str): Name of the table.
            data (Dict[str, Any]): Dictionary of column names and values to insert.

        Returns:
            Optional[str]: The ID of the inserted row (lastrowid for INTEGER PK, or 'id' from data for TEXT PK like UUID),
                           or None if insertion failed.
        """
        if not data:
            logger.warning(f"Insert called with empty data for table `{table}`.")
            return None
        
        columns = ', '.join([f"`{key}`" for key in data.keys()]) # Backtick column names
        placeholders = ', '.join(['?'] * len(data))
        values = tuple(self._serialize_if_needed(v) for v in data.values())
        query = f"INSERT INTO `{table}` ({columns}) VALUES ({placeholders})"

        cursor = self._execute(query, values)
        if cursor and self.conn:
            try:
                self.conn.commit()
                inserted_id = cursor.lastrowid # For INTEGER PRIMARY KEY AUTOINCREMENT
                if inserted_id:
                    return str(inserted_id)
                # If PK is not auto-incrementing (e.g., UUID provided in data)
                # lastrowid might be 0 or less meaningful if PK is not an alias of rowid.
                # In such cases, the ID should be part of the input `data`.
                if "id" in data: 
                    return str(data["id"])
                logger.warning(f"Insert into `{table}` successful but could not determine a standard way to return ID (lastrowid: {cursor.lastrowid}, 'id' not in data).")
                return "success_no_id_returned" # Or some indicator of success if ID is not retrievable this way
            except sqlite3.Error as e:
                logger.error(f"Error committing insert into `{table}`: {e}", exc_info=True)
                if self.conn: self.conn.rollback()
                return None
        logger.error(f"Failed to execute insert for table `{table}`, cursor not available or connection lost.")
        return None

    def find_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Finds a single record matching the filters.

        Args:
            table (str): Name of the table.
            filters (Dict[str, Any]): Dictionary of filter conditions.

        Returns:
            Optional[Dict[str, Any]]: The record as a dictionary, or None if not found or error.
        """
        if not filters:
            logger.warning(f"find_one called on table `{table}` without filters. This is ambiguous and not recommended. Fetching first arbitrary record.")
            query = f"SELECT * FROM `{table}` LIMIT 1"
            params_tuple: Tuple[Any, ...] = ()
        else:
            conditions_list, params_list_internal = self._build_filter_conditions(filters)
            if not conditions_list:
                logger.warning(f"No valid conditions built from filters for find_one on table `{table}`. Filters: {filters}. Fetching first arbitrary record.")
                query = f"SELECT * FROM `{table}` LIMIT 1"
                params_tuple = ()
            else:
                where_clause = " AND ".join(conditions_list)
                query = f"SELECT * FROM `{table}` WHERE {where_clause} LIMIT 1"
                params_tuple = tuple(params_list_internal)

        cursor = self._execute(query, params_tuple)
        if cursor:
            row = cursor.fetchone() # dict_factory makes this return a dict
            return row if row else None
        return None

    def find(self, table: str, filters: Optional[Dict[str, Any]] = None,
             order_by: Optional[str] = None, limit: Optional[int] = None,
             offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Finds multiple records matching filters with optional ordering, limit, and offset.

        Args:
            table (str): Name of the table.
            filters (Optional[Dict[str, Any]]): Dictionary of filter conditions.
            order_by (Optional[str]): Column(s) to order by (e.g., "created_at DESC").
            limit (Optional[int]): Maximum number of records to return.
            offset (Optional[int]): Number of records to skip.

        Returns:
            List[Dict[str, Any]]: A list of records as dictionaries. Empty list if none found or error.
        """
        base_query = f"SELECT * FROM `{table}`"
        conditions_list, params_list_internal = self._build_filter_conditions(filters or {})

        if conditions_list:
            base_query += " WHERE " + " AND ".join(conditions_list)
        
        if order_by:
            # Basic sanitization for order_by to prevent very simple injection.
            # A robust solution would validate 'order_by' against known columns and formats.
            safe_order_by = "".join(c for c in order_by if c.isalnum() or c in ('_', ' ', ',', '.', '`', 'ASC', 'DESC')) # Allow backticks and dots for table.column
            if safe_order_by:
                base_query += f" ORDER BY {safe_order_by}"
            else:
                logger.warning(f"Invalid or potentially unsafe characters in order_by clause: '{order_by}'. Ignoring.")
        
        if limit is not None:
            base_query += " LIMIT ?"
            params_list_internal.append(limit)
        
        if offset is not None:
            if limit is None: # SQLite requires LIMIT with OFFSET
                base_query += " LIMIT -1" # -1 in SQLite means no upper limit
            base_query += " OFFSET ?"
            params_list_internal.append(offset)

        params_tuple = tuple(params_list_internal)
        cursor = self._execute(base_query, params_tuple)
        if cursor:
            return cursor.fetchall() # dict_factory makes this return list of dicts
        return []

    def update(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Updates records in the table matching the filters with the new update data.

        Args:
            table (str): Name of the table.
            filters (Dict[str, Any]): Filter conditions to select records for update.
            updates (Dict[str, Any]): Dictionary of column names and new values.

        Returns:
            int: The number of rows updated. Returns 0 on failure or if no rows matched.
        """
        if not filters: # Require filters to prevent accidental update of all rows
            logger.error(f"Update operation on table `{table}` requires filters. Aborting.")
            return 0
        if not updates:
            logger.warning(f"Update operation on table `{table}` called with no update data.")
            return 0
        
        set_clauses_list = [f"`{key}` = ?" for key in updates.keys()]
        set_conditions_str = ", ".join(set_clauses_list)
        
        update_values_list = [self._serialize_if_needed(v) for v in updates.values()]

        where_conditions_list, where_params_list_internal = self._build_filter_conditions(filters)
        if not where_conditions_list:
            logger.error(f"Update for table `{table}` failed: no valid WHERE conditions from filters {filters}.")
            return 0
        where_clause_str = " AND ".join(where_conditions_list)
        
        query = f"UPDATE `{table}` SET {set_conditions_str} WHERE {where_clause_str}"
        # Parameters must be in order: first update values, then where clause values
        params_tuple = tuple(update_values_list) + tuple(where_params_list_internal)
        
        cursor = self._execute(query, params_tuple)
        if cursor and self.conn:
            try:
                self.conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Error committing update to `{table}`: {e}", exc_info=True)
                if self.conn: self.conn.rollback()
                return 0
        logger.error(f"Failed to execute update for table `{table}`, cursor not available or connection lost.")
        return 0

    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        """
        Deletes records from the table matching the filters.

        Args:
            table (str): Name of the table.
            filters (Dict[str, Any]): Filter conditions to select records for deletion.
                                      Must not be empty to prevent accidental deletion of all rows.
        Returns:
            int: The number of rows deleted. Returns 0 on failure or if no rows matched.
        """
        if not filters:
            logger.error(f"Delete operation on table `{table}` requires filters. Aborting to prevent deleting all rows.")
            return 0 

        where_conditions_list, params_list_internal = self._build_filter_conditions(filters)
        if not where_conditions_list:
            logger.error(f"Delete for table `{table}` failed: no valid WHERE conditions from filters {filters}.")
            return 0
        
        where_clause_str = " AND ".join(where_conditions_list)
        query = f"DELETE FROM `{table}` WHERE {where_clause_str}"
        params_tuple = tuple(params_list_internal)

        cursor = self._execute(query, params_tuple)
        if cursor and self.conn:
            try:
                self.conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Error committing delete from `{table}`: {e}", exc_info=True)
                if self.conn: self.conn.rollback()
                return 0
        logger.error(f"Failed to execute delete for table `{table}`, cursor not available or connection lost.")
        return 0
    
    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """
        Executes a provided SQL SELECT query and returns results.
        Primarily intended for SELECT queries. For DML, use insert/update/delete methods.
        This method does NOT commit transactions for DML statements.

        Args:
            query (str): The SQL query string.
            params (Optional[tuple]): Parameters to bind to the query.

        Returns:
            List[Dict[str, Any]]: A list of rows as dictionaries for SELECT queries.
                                  Returns empty list for non-SELECT or on error.
        """
        cursor = self._execute(query, params)
        if cursor:
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall() # dict_factory ensures list of dicts
            else:
                logger.warning(f"execute_query called with non-SELECT statement: '{query[:100]}...'. "
                               f"This method does not commit DML. Use insert/update/delete for modifications.")
                return [] 
        return []

    def execute_script(self, script: str) -> bool:
        """
        Executes a script containing multiple SQL statements.
        Each script execution is within a transaction.

        Args:
            script (str): SQL script string.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.conn:
            if not self.connect():
                logger.error("Failed to execute script: No database connection.")
                return False
        try:
            self.conn.executescript(script) # type: ignore # self.conn is checked
            self.conn.commit() # type: ignore
            logger.info("SQL script executed and committed successfully.")
            return True
        except sqlite3.Error as e:
            logger.error(f"SQLite script execution error: {e}", exc_info=True)
            if self.conn: self.conn.rollback()
            return False
        
    def count(self, table: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Counts records in a table matching the given filters.

        Args:
            table (str): Name of the table.
            filters (Optional[Dict[str, Any]]): Dictionary of filter conditions.

        Returns:
            int: The count of matching records. Returns 0 on error or if no records found.
        """
        # Use an alias for COUNT(*) for reliable dictionary key access
        base_query = f"SELECT COUNT(*) as count_result FROM `{table}`"
        
        conditions_list, params_list_internal = self._build_filter_conditions(filters or {})
        
        if conditions_list:
            base_query += " WHERE " + " AND ".join(conditions_list)
        
        params_tuple = tuple(params_list_internal)
        
        result_data = self.execute_query(base_query, params_tuple) # Uses the modified execute_query
        
        if result_data and isinstance(result_data, list) and len(result_data) > 0:
            first_row_dict = result_data[0]
            if isinstance(first_row_dict, dict) and 'count_result' in first_row_dict:
                try:
                    return int(first_row_dict['count_result'])
                except (ValueError, TypeError):
                    logger.error(f"Could not convert count_result '{first_row_dict['count_result']}' to int for table `{table}`.")
                    return 0
            else:
                logger.warning(f"Count query for `{table}` returned unexpected row format (expected dict with 'count_result'): {first_row_dict}")
                return 0
        else:
            logger.warning(f"No data returned for count query on `{table}` with filters {filters}: {result_data}")
            return 0