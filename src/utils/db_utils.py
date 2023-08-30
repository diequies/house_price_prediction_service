""" Methods to manage the DB connections """
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, text
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


def execute_mysql_query(query: str, schema: str = "") -> DataFrame:
    """
    Method to execute a mysql query
    :param query: The query to execute
    :param schema: The schema to query
    :return: The DataFrame resulting from the query
    """

    connection = get_mysql_connection(schema=schema)

    return pd.read_sql(text(query), con=connection.connect())
