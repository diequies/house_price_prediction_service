""" Linear Regressor model """

from typing import Any, Dict, List, Optional

import pandas as pd
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from src.training.interfaces import PredictorBase
from src.utils.exceptions import (
    MissingColumnError,
    NoModelInfoAvailableError,
    WrongTypeColumnError,
)


class LinearRegressionPredictor(PredictorBase, PythonModel):
    """
    Linear Regression model to predict house prices
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Constructor method for the Linear Regression Model
        :param fit_intercept: Whether to calculate the intercept
        """
        self._model_info = None
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(
        self, x_train: DataFrame, y_train: DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trains the linear regression model
        :param x_train: Processed data to use to train the model
        :param y_train: Target feature
        :param params: Dictionary with the parameters to persist to MLFlow
        """
        params["model"] = self.model.__class__.__name__
        params["fit_intercept"] = self.model.fit_intercept
        params["artifact_path"] = "linear_regression"
        self.model = self.model.fit(X=x_train, y=y_train)

        return params

    def predict(
        self, model_input: DataFrame, context, params: Optional[Dict[str, Any]] = None
    ) -> List:
        """
        Method to predict house prices using the linear regression model
        :param context: A :class:`~PythonModelContext` instance containing artifacts
                        that the model can use to perform inference.
        :param model_input: A pyfunc-compatible input for the model to evaluate.
        :param params: Additional parameters to pass to the model for inference.
        :return: List of predicted values
        """
        if context:
            input_signature = context.get("input_signature", None)
            if input_signature:
                self._validate_inputs(
                    model_input=model_input, input_signature=input_signature
                )

        y_predict = list(self.model.predict(X=model_input))
        return pd.Series(y_predict).tolist()

    @property
    def model_info(self) -> ModelInfo:
        """
        Getter for the model info
        :return: The ModelInfo MLFlow class
        """
        if self._model_info is not None:
            return self._model_info

        raise NoModelInfoAvailableError("Model Info not available yet")

    @model_info.setter
    def model_info(self, model_info: ModelInfo) -> None:
        """
        Setter for the model info
        :param model_info: The model info to be set
        """
        self._model_info = model_info

    @staticmethod
    def _validate_inputs(model_input: DataFrame, input_signature: List[Dict]) -> bool:
        required_columns = {
            columns["name"] for columns in input_signature if "name" in columns.keys()
        }

        all_column_names_in_model_input = (
            required_columns.intersection(set(model_input.columns)) == required_columns
        )

        if not all_column_names_in_model_input:
            raise MissingColumnError("Model input missing required columns")

        for column_data in input_signature:
            column_name = column_data["name"]
            column_type = column_data["type"]

            is_type_correct = column_type == str(model_input[column_name].dtype)

            if not is_type_correct:
                raise WrongTypeColumnError(
                    f"Column {column_name} is "
                    f"{str(model_input[column_name].dtype)} "
                    f"but the expected type was {column_type}"
                )

        return True
