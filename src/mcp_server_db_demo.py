"""SQLite Demo for FastMCP Server."""

import argparse
import logging
import sqlite3

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("sqlite-demo")


def init_db() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Initialize the SQLite database and create the 'people' table if it does not exist.

    Returns:
        tuple[sqlite3.Connection, sqlite3.Cursor]: The database connection and cursor.

    """
    conn = sqlite3.connect("demo.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            profession TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn, cursor


@mcp.tool()
def add_data(query: str) -> bool:
    """
    Add new data to the people table using a SQL INSERT query.

    Args:
        query (str): SQL INSERT query following this format:
            INSERT INTO people (name, age, profession)
            VALUES ('John Doe', 30, 'Engineer')

    Schema:
        - name: Text field (required)
        - age: Integer field (required)
        - profession: Text field (required)
        Note: 'id' field is auto-generated

    Returns:
        bool: True if data was added successfully, False otherwise

    Example:
        >>> query = '''
        ... INSERT INTO people (name, age, profession)
        ... VALUES ('Alice Smith', 25, 'Developer')
        ... '''
        >>> add_data(query)
        True

    """
    conn, cursor = init_db()
    try:
        logger.info("\n\nExecuting add_data with query: %s", query)
        cursor.execute(query)
        conn.commit()
    except sqlite3.Error as e:
        logger.info("Error adding data: %s", e)
        return False
    else:
        return True
    finally:
        conn.close()


@mcp.tool()
def read_data(query: str = "SELECT * FROM people") -> list:
    """
    Read data from the people table using a SQL SELECT query.

    Args:
        query (str, optional): SQL SELECT query. Defaults to "SELECT * FROM people".

    Examples:
            - "SELECT * FROM people"
            - "SELECT name, age FROM people WHERE age > 25"
            - "SELECT * FROM people ORDER BY age DESC"

    Returns:
        list: List of tuples containing the query results.
              For default query, tuple format is (id, name, age, profession)

    Example:
        >>> # Read all records
        >>> read_data()
        [(1, 'John Doe', 30, 'Engineer'), (2, 'Alice Smith', 25, 'Developer')]

        >>> # Read with custom query
        >>> read_data("SELECT name, profession FROM people WHERE age < 30")
        [('Alice Smith', 'Developer')]

    """
    conn, cursor = init_db()
    try:
        logger.info("\n\nExecuting read_data with query: %s", query)
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.info("Error reading data: %s", e)
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    # Start the server
    logger.info("🚀Starting server... ")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type",
        type=str,
        default="sse",
        choices=["sse", "stdio"],
    )

    args = parser.parse_args()
    # Only pass server_type to run()
    mcp.run(args.server_type)
