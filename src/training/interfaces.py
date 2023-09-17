""" Interfaces to define general contracts """

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pandas import DataFrame


class PredictorBase(ABC):
    """
    Defines the general contract for the prediction models
    """

    @abstractmethod
    def fit(
        self, x_train: DataFrame, y_train: DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Method to train the model
        :param x_train: Processed data to use to train the model
        :param y_train: Target feature
        :param params: Dictionary with the parameters to persist to MLFlow
        """

    @abstractmethod
    def predict(self, x_predict: DataFrame) -> List:
        """
        Method to predict using a trained model
        :param x_predict: Data to use to predict
        :return: Predictions
        """


class TrainingBase(ABC):  # pylint: disable=too-few-public-methods
    """
    Defines the general contract for the training classes
    """

    @abstractmethod
    def __init__(
        self,
        input_variables: List[str],
        model: PredictorBase,
    ) -> None:
        """
        Constructor for the training class
        :param input_variables: list of features to include in the model training
        :param model: Model to use for training
        """

    @abstractmethod
    def _load_data(self) -> DataFrame:
        """
        Loads the data from the necessary data sources
        :return: DataFrame with the raw data
        """

    @abstractmethod
    def _process_data(self) -> None:
        """
        Gets the raw data and process it to be used by the model
        """

    @staticmethod
    @abstractmethod
    def _train_test_split(processed_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Method to separate the train and test data appropriately
        """

    @abstractmethod
    def _fit_predictor(self, x_train: DataFrame, y_train: DataFrame) -> None:
        """
        Method to fit the model
        :param x_train: data to be used for training
        :param y_train: target feature to train the model
        """

    @abstractmethod
    def _log_results(self, x_test: DataFrame, y_test: DataFrame) -> None:
        """
        Method to calculate and log the metrics
        :param x_test: the data to predict the test results
        :param y_test: the true values to compare
        """

    @abstractmethod
    def run_training(self) -> None:
        """
        Method to run the whole training process
        """

    @abstractmethod
    def save_model(self) -> None:
        """
        Method to save or register the trained model
        """
