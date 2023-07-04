""" Methods to manage the DB connections """

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.utils.utils import MYSQL_DETAILS


def get_mysql_connection(schema: str = "") -> Engine:
    """Creates a MySQL connection
    Args:
         schema (str): schema to connect"""

    username = MYSQL_DETAILS["username"]
    password = MYSQL_DETAILS["password"]
    host = MYSQL_DETAILS["host"]
    port = MYSQL_DETAILS["port"]

    return create_engine(f"mysql://{username}:{password}@{host}:{port}/{schema}")
