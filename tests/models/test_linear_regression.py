""" Tests for linear regression class """
import pandas as pd
import pytest

from src.models.linear_regression import LinearRegressionPredictor
from src.utils.exceptions import MissingColumnError, WrongTypeColumnError


def test_not_all_required_columns():
    """
    Test that we raise a MissingColumnError when not all the required
    inputs are available
    """
    available_columns = ["floor_m_sqrt", "garden", "parking"]
    required_columns = ["floor_m_sqrt", "garden", "parking", "balcony_terrace"]
    required_types = ["int", "bool", "bool", "bool"]
    input_signature = [
        {"name": column_name, "type": column_type}
        for column_name, column_type in zip(
            required_columns, required_types, strict=True
        )
    ]
    model_input = pd.DataFrame(columns=available_columns)

    with pytest.raises(MissingColumnError):
        linear_regression = LinearRegressionPredictor()
        linear_regression.predict(
            model_input=model_input, context={"input_signature": input_signature}
        )


def test_not_all_types_correct():
    """
    Tests that we raise WrongTypeColumnError when any of the model input column types
    is wrong
    """

    available_columns = ["floor_m_sqrt", "garden", "parking", "balcony_terrace"]
    column_values = [True, True, True, True]
    df_dict = dict(zip(available_columns, column_values, strict=True))
    required_columns = ["floor_m_sqrt", "garden", "parking", "balcony_terrace"]
    required_types = ["int", "bool", "bool", "bool"]
    input_signature = [
        {"name": column_name, "type": column_type}
        for column_name, column_type in zip(
            required_columns, required_types, strict=True
        )
    ]
    model_input = pd.DataFrame(df_dict, index=range(0, 1))

    with pytest.raises(WrongTypeColumnError):
        linear_regression = LinearRegressionPredictor()
        linear_regression.predict(
            model_input=model_input, context={"input_signature": input_signature}
        )
