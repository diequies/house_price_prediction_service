""" Methods to download the data """
import itertools
import logging
from typing import List

import pandas as pd
from pandas import DataFrame
from sqlalchemy import text

from src.utils.db_utils import get_mysql_connection
from src.utils.utils import DATA_TO_LOAD_MAP, MYSQL_DETAILS


def load_mysql_house_details(input_variables: List[str]) -> DataFrame:
    """Method to get the raw tabular data from the main MySQL table
    :return A pandas DataFrame with the required data"""

    logging.info("Getting raw house details from MySQL")

    connection = get_mysql_connection(schema=MYSQL_DETAILS["schema"])

    columns_to_load = [
        DATA_TO_LOAD_MAP.get(key, [])
        for key in DATA_TO_LOAD_MAP.keys()
        if key in input_variables
    ]

    columns_to_load = set(itertools.chain.from_iterable(columns_to_load))

    query = f"SELECT {', '.join(list(columns_to_load))} FROM houses_details;"

    with connection.connect() as conn:
        raw_data = pd.read_sql(text(query), con=conn)

    return raw_data


if __name__ == "__main__":
    pass
