""" Interfaces to define general contracts """

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

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
        pass

    @abstractmethod
    def predict(self, x: DataFrame) -> Series:
        """
        Method to predict using a trained model
        :param x: Data to use to predict
        :return: Predictions
        """
        pass


class TrainingBase(ABC):
    """
    Defines the general contract for the training classes
    """

    @abstractmethod
    def __init__(
        self,
        input_variables: List[str],
        model: PredictorBase,
        config: Dict[str: Union[str, int, float]],
    ) -> None:
        """
        Constructor for the training class
        :param input_variables: list of features to include in the model training
        :param model: Model to use for training
        :param config: Configuration dictionary, containing setup information
        """
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """
        Loads the data from the necessary data sources
        """
        pass

    @abstractmethod
    def _process_data(self) -> None:
        """
        Gets the raw data and process it to be used by the model
        """
        pass

    @abstractmethod
    def _train_test_split(self, processed_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Method to separate the train and test data appropriately
        :param processed_df: DataFrame containing the processed data
        :return: a tuple of DataFrames, the first for training data and the second for validation
        """
        pass

    @abstractmethod
    def _fit_predictor(self) -> None:
        """
        Method to fit the model
        """
        pass

    @abstractmethod
    def run_training(self) -> None:
        """
        Method to run the whole training process
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Method to store the model in a specific location
        :param path: address to the location
        """
        pass