""" Linear Regressor model """

from typing import Any, Dict, List

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from src.training.interfaces import PredictorBase
from src.utils.exceptions import NoModelInfoAvailable


class LinearRegressionPredictor(PredictorBase):
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
        self.model = self.model.fit(X=x_train, y=y_train)

        signature = infer_signature(
            model_input=x_train, model_output=self.model.predict(x_train)
        )

        self.model_info = mlflow.sklearn.log_model(
            sk_model=self.model,
            registered_model_name=params["model"],
            artifact_path="linear_regression",
            signature=signature,
        )

        client = MlflowClient()

        model_version = client.get_latest_versions(params["model"], stages=["None"])

        if len(model_version) > 0:
            version = model_version[0].version
        else:
            version = 1

        model_version = client.transition_model_version_stage(
            name=params["model"],
            version=version,
            stage="staging",
            archive_existing_versions=True,
        )

        print(
            f"Model {model_version.name} registered in {model_version.current_stage} "
            f"at {model_version.last_updated_timestamp}"
        )

        return params

    def predict(self, x_predict: DataFrame) -> List:
        """
        Method to predict house prices using the linear regression model
        :param x_predict: Data to use to predict
        :return: List of predicted values
        """
        y_predict = list(self.model.predict(X=x_predict))
        return pd.Series(y_predict).tolist()

    @property
    def model_info(self) -> ModelInfo:
        """
        Getter for the model info
        :return: The ModelInfo MLFlow class
        """
        if self._model_info is not None:
            return self._model_info
        else:
            raise NoModelInfoAvailable("Model Info not available yet")

    @model_info.setter
    def model_info(self, model_info: ModelInfo) -> None:
        """
        Setter for the model info
        :param model_info: The model info to be set
        """
        self._model_info = model_info
