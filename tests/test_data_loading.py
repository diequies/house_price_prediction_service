""" Test for the data loading methods"""
import pytest

from src.data_modeling.data_loading import load_mysql_house_details
from src.utils.exceptions import NotAllInputsAvailableError


def test_not_all_columns_available():
    """
    test that we raise a NotAllInputsAvailableError when not all the required
    inputs are available
    """

    input_features = ["listing_id", "floor_m_sqrt"]

    with pytest.raises(NotAllInputsAvailableError):
        load_mysql_house_details(input_variables=input_features)
