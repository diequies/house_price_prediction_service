""" Training class to train basic models """
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame

from src.data_modeling.data_loading import load_mysql_house_details
from src.models.linear_regression import LinearRegressionPredictor
from src.training.interfaces import PredictorBase, TrainingBase
from src.utils.utils import INPUT_FEATURES, TARGET_FEATURE, TRAIN_TEST_SPLIT


class TrainingManagerPlain(TrainingBase):
    """
    Plain training manager class to train basic models
    """

    def __init__(
        self,
        input_variables: List[str],
        model: PredictorBase,
    ):
        """
        Constructor for the training manager class
        :param input_variables: list of features to include in the model training
        :param model: Model to use for training
        """
        self.input_variables = input_variables
        self.model = model
        self.raw_data = self._load_data()
        self.processed_data = pd.DataFrame()

    def _load_data(self) -> DataFrame:
        """
        Loads the data from the necessary data sources
        :return: DataFrame with the raw data
        """
        print("Loading data")
        return load_mysql_house_details(self.input_variables)

    def _process_data(self) -> None:
        """
        Method to process the raw data
        """

        self.processed_data = self.raw_data.dropna()

    @staticmethod
    def _train_test_split(processed_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Divide the data into train and testing datasets
        :param processed_data: the processed data to divide
        :return: the tuple of both, train and test datasets
        """

        idx_train = int(TRAIN_TEST_SPLIT * processed_data.shape[0])
        train_data = processed_data.iloc[:idx_train, :]
        test_data = processed_data.iloc[idx_train:, :]
        return train_data, test_data

    def _fit_predictor(self, x_train: DataFrame, y_train: DataFrame) -> None:
        """
        Method to fit the model with the processed data
        :param x_train: data to be used for training
        :param y_train: target feature to train the model
        """

        self.model.fit(x_train=x_train, y_train=y_train)

    def run_training(self) -> None:
        """
        Main method to run the model training
        """

        self._process_data()

        train_data, _ = self._train_test_split(self.processed_data)

        x_train = train_data[self.input_variables]
        y_train = train_data[TARGET_FEATURE]

        self._fit_predictor(x_train=x_train, y_train=y_train)

    def save_model(self, path: str) -> None:
        pass


def run() -> None:
    """
    Main method to run the training manager
    """

    training_manager = TrainingManagerPlain(
        input_variables=INPUT_FEATURES,
        model=LinearRegressionPredictor(fit_intercept=True),
    )

    training_manager.run_training()
    training_manager.save_model(path="path")


if __name__ == "__main__":
    run()
