# cinfer/core/database/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class DatabaseService(ABC):
    """
    Abstract base class for database services.
    Defines a common interface for database operations,
    as outlined in the system's class diagram (document section 4.6.3).
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establishes a connection to the database.
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Closes the database connection.
        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, table: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Inserts a new record into the specified table.
        Args:
            table (str): The name of the table.
            data (Dict[str, Any]): A dictionary where keys are column names
                                   and values are the data to be inserted.
        Returns:
            Optional[str]: The ID of the newly inserted record (if applicable, e.g., lastrowid),
                           or None if insertion failed or ID is not retrievable.
        """
        raise NotImplementedError

    @abstractmethod
    def find_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Finds a single record in the table that matches the filters.
        Args:
            table (str): The name of the table.
            filters (Dict[str, Any]): A dictionary of columns and their expected values
                                     to filter the records.
        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the record,
                                      or None if no record matches.
        """
        raise NotImplementedError

    @abstractmethod
    def find(self, table: str, filters: Optional[Dict[str, Any]] = None,
             order_by: Optional[str] = None, 
             limit: Optional[int] = None,
             offset: Optional[int] = None,
             search_term: Optional[str] = None,
             search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Finds multiple records in the table that match the filters.
        Args:
            table (str): The name of the table.
            filters (Dict[str, Any], optional): Columns and values to filter by. Defaults to None (no filter).
            search_term (Optional[str], optional): Search term for fuzzy search. Defaults to None.
            search_fields (Optional[List[str]], optional): Fields to search in. Defaults to None.
            order_by (Optional[str], optional): Column to order by (e.g., "created_at DESC"). Defaults to None.
            limit (Optional[int], optional): Maximum number of records to return. Defaults to None.
            offset (Optional[int], optional): Number of records to skip. Defaults to None.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a record.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, table: str, filters: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Updates records in the table that match the filters.
        Args:
            table (str): The name of the table.
            filters (Dict[str, Any]): Columns and values to identify records to update.
            updates (Dict[str, Any]): Columns and new values to set for the matched records.
        Returns:
            int: The number of records updated.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, table: str, filters: Dict[str, Any]) -> int:
        """
        Deletes records from the table that match the filters.
        Args:
            table (str): The name of the table.
            filters (Dict[str, Any]): Columns and values to identify records to delete.
        Returns:
            int: The number of records deleted.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[Any]:
        """
        Executes an arbitrary SQL query.
        This method should be used with caution and primarily for operations
        not covered by other specific methods (e.g., complex joins, aggregations).
        Args:
            query (str): The SQL query string (use placeholders like ? for parameters).
            params (Optional[tuple], optional): A tuple of parameters to substitute
                                                into the query. Defaults to None.
        Returns:
            Optional[Any]: Query result (e.g., list of rows for SELECT, row count for DML,
                           or cursor for more complex operations). Behavior depends on query type.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_script(self, script: str) -> bool:
        """
        Executes a script containing multiple SQL statements.
        Args:
            script (str): A string containing SQL statements separated by semicolons.
        Returns:
            bool: True if the script executed successfully, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def count(self, table: str, filters: Optional[Dict[str, Any]] = None,
              search_term: Optional[str] = None,
              search_fields: Optional[List[str]] = None) -> int:
        """
        Counts records in the table that match the filters.
        """
        raise NotImplementedError