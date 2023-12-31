""" Test for the data loading methods"""
import pandas as pd
import pytest

from src.data_modeling.data_loading import load_mysql_house_details
from src.utils.exceptions import NotAllInputsAvailableError


def test_not_all_columns_available(mocker):
    """
    test that we raise a NotAllInputsAvailableError when not all the required
    inputs are available
    """
    wrong_input_features = ["listing_id"]
    mocker.patch(
        "src.data_modeling.data_loading.execute_mysql_query",
        return_value=pd.DataFrame(columns=wrong_input_features),
    )

    with pytest.raises(NotAllInputsAvailableError):
        load_mysql_house_details()
