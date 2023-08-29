""" Methods to download the data """
import itertools
import logging
from datetime import datetime
from typing import List

import pandas as pd
from pandas import DataFrame
from sqlalchemy import text

from src.utils.db_utils import get_mysql_connection
from src.utils.exceptions import NotAllInputsAvailableError
from src.utils.utils import CONFIG, DATA_TO_LOAD_MAP, MYSQL_DETAILS, TARGET_FEATURE


def load_mysql_house_details(input_variables: List[str]) -> DataFrame:
    """Method to get the raw tabular data from the main MySQL table
    :return A pandas DataFrame with the required data"""

    logging.info("Getting raw house details from MySQL")

    connection = get_mysql_connection(schema=MYSQL_DETAILS["schema"])

    inputs_required = input_variables + TARGET_FEATURE

    columns_to_load = [
        value for key, value in DATA_TO_LOAD_MAP.items() if key in inputs_required
    ]

    set_columns_to_load = set(itertools.chain.from_iterable(columns_to_load))

    time_now = int(datetime.utcnow().timestamp())
    number_of_days = CONFIG.get("days_of_data", 180)

    query = (
        f"SELECT {', '.join(list(set_columns_to_load))} "
        f"FROM houses_details "
        f"WHERE publish_unix_time >= "
        f"{time_now - number_of_days * 60 * 60 * 24}"
    )

    with connection.connect() as conn:
        raw_data = pd.read_sql(text(query), con=conn)

    if not set_columns_to_load.issubset(set(raw_data.columns)):
        raise NotAllInputsAvailableError("Please review the input data")

    return raw_data


if __name__ == "__main__":
    pass
