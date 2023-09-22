""" Methods to support the training processes"""
import ast
from typing import List, Optional, Tuple

import mlflow
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.pyfunc import PyFuncModel
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

from src.training.interfaces import PredictorBase
from src.utils.exceptions import EmptyModelSignatureError


def get_production_model(model_name: str) -> Optional[PyFuncModel]:
    """
    Method to load a specific model in production
    :param model_name: the name of the model to load
    :return: The loaded model for prediction
    """
    client = MlflowClient()

    model_version_prod = client.get_latest_versions(model_name, stages=["Production"])

    if len(model_version_prod) > 0:
        model_uri_prod = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri_prod)

    return None


def compare_models(
    new_model: PredictorBase,
    old_model: PyFuncModel,
    x_test: DataFrame,
    y_test: DataFrame,
) -> Tuple[float, float]:
    """
    Compares the new and old model for selection
    :param new_model: New trained model
    :param old_model: Current production model
    :param x_test: Data to test the models
    :param y_test: Target feature to test the model
    :return: The RMSE for the new and old models against the test data
    """

    old_model_inputs = get_inputs_from_signature(old_model.metadata.signature)

    y_pred_prod = old_model.predict(x_test[old_model_inputs])

    y_pred_staging = new_model.predict(x_test)

    rmse_prod = mean_squared_error(y_true=y_test, y_pred=y_pred_prod, squared=False)
    rmse_staging = mean_squared_error(
        y_true=y_test, y_pred=y_pred_staging, squared=False
    )

    return rmse_staging, rmse_prod


def get_inputs_from_signature(signature: ModelSignature) -> List:
    """
    Method to extract the list of inputs from the MLFlow model signature
    :param signature: MLFlow model signature
    :return: The list of strings with the name of the model inputs
    """
    string_of_inputs = signature.to_dict().get("inputs", "")

    list_of_inputs = ast.literal_eval(string_of_inputs)

    if len(string_of_inputs) == 0 or len(list_of_inputs) == 0:
        raise EmptyModelSignatureError("Model signature does not contain inputs")

    return [item["name"] for item in list_of_inputs]
