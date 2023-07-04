""" Methods to download the data """
import logging

import pandas as pd
from pandas import DataFrame
from sqlalchemy import text

from src.utils.db_utils import get_mysql_connection
from src.utils.utils import COLUMNS_TO_LOAD, MYSQL_DETAILS


def load_mysql_house_details() -> DataFrame:
    """Method to get the raw tabular data from the main MySQL table
    :return A pandas DataFrame with the required data"""

    logging.info("Getting raw house details from MySQL")

    connection = get_mysql_connection(schema=MYSQL_DETAILS["schema"])

    query = f"SELECT {', '.join(COLUMNS_TO_LOAD)} FROM houses_details;"

    with connection.connect() as conn:
        raw_data = pd.read_sql(text(query), con=conn)

    return raw_data


if __name__ == "__main__":
    pass
