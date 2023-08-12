""" Interfaces to define general contracts """

from abc import ABC, abstractmethod
from typing import List, Tuple

from pandas import DataFrame, Series


class PredictorBase(ABC):
    """
    Defines the general contract for the prediction models
    """

    @abstractmethod
    def fit(self, x_train: DataFrame, y_train: Series) -> None:
        """
        Method to train the model
        :param x_train: Processed data to use to train the model
        :param y_train: Target feature
        """

    @abstractmethod
    def predict(self, x_predict: DataFrame) -> List:
        """
        Method to predict using a trained model
        :param x_predict: Data to use to predict
        :return: Predictions
        """


class TrainingBase(ABC):
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

    @staticmethod
    @abstractmethod
    def _load_data(data_path: str, filename: str) -> DataFrame:
        """
        Loads the data from the necessary data sources
        :param data_path: path to the data folder
        :param filename: name of the file containing the data
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
    def _fit_predictor(self) -> None:
        """
        Method to fit the model
        """

    @abstractmethod
    def run_training(self) -> None:
        """
        Method to run the whole training process
        """

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Method to store the model in a specific location
        :param path: address to the location
        """
