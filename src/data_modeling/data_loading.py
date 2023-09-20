""" Methods to download the data """
import itertools
import logging
from datetime import datetime

from pandas import DataFrame

from src.utils.db_utils import execute_mysql_query
from src.utils.exceptions import NotAllInputsAvailableError
from src.utils.utils import (
    COLUMNS_TO_LOAD,
    DATA_TO_LOAD_MAP,
    DAYS_OF_DATA_TO_LOAD,
    MYSQL_DETAILS,
)


def load_mysql_house_details() -> DataFrame:
    """Method to get the raw tabular data from the main MySQL table
    :return A pandas DataFrame with the required data"""

    logging.info("Getting raw house details from MySQL")

    inputs_required = COLUMNS_TO_LOAD

    columns_to_load = [
        value for key, value in DATA_TO_LOAD_MAP.items() if key in inputs_required
    ]

    set_columns_to_load = set(itertools.chain.from_iterable(columns_to_load))

    time_now = int(datetime.utcnow().timestamp())

    query = (
        f"SELECT {', '.join(list(set_columns_to_load))} "
        f"FROM houses_details "
        f"WHERE publish_unix_time >= "
        f"{time_now - DAYS_OF_DATA_TO_LOAD * 60 * 60 * 24}"
    )

    raw_data = execute_mysql_query(query=query, schema=MYSQL_DETAILS["schema"])

    if not set_columns_to_load.issubset(set(raw_data.columns)):
        raise NotAllInputsAvailableError("Please review the input data")

    return raw_data


if __name__ == "__main__":
    pass
