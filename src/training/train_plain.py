""" Training class to train basic models """
from typing import Dict, List, Tuple, Union

import pandas as pd
from pandas import DataFrame

from src.models.linear_regression import LinearRegressionPredictor
from src.training.interfaces import PredictorBase, TrainingBase
from src.utils.utils import COLUMNS_TO_LOAD


class TrainingManagerPlain(TrainingBase):
    """
    Plain training manager class to train basic models
    """

    def __init__(
        self,
        input_variables: List[str],
        model: PredictorBase,
        config: Dict[str, Union[str, int, float]],
    ):
        """
        Constructor for the training manager class
        :param input_variables: list of features to include in the model training
        :param model: Model to use for training
        :param config: Configuration dictionary, containing setup information
        """
        self.input_variables = input_variables
        self.model = model
        self.raw_data = self._load_data(
            data_path=config["data_path"], filename=config["data_filename"]
        )

    @staticmethod
    def _load_data(data_path: str, filename: str) -> DataFrame:
        """
        Loads the data from the necessary data sources
        :param data_path: path to the data folder
        :param filename: name of the file containing the data
        :return: DataFrame with the raw data
        """
        print(f"Reading {filename} from {data_path}")
        return pd.read_csv(f"{data_path}/{filename}")

    def _process_data(self) -> None:
        """
        Method to process the raw data
        """
        self.processed_data = self.raw_data

    @staticmethod
    def _train_test_split(processed_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        pass

    def _fit_predictor(self) -> None:
        pass

    def run_training(self) -> None:
        pass

    def save_model(self, path: str) -> None:
        pass


def run() -> None:
    """
    Main method to run the training manager
    """
    config = {"data_path": "../../data/raw/", "data_filename": "dummy_data"}

    training_manager = TrainingManagerPlain(
        input_variables=COLUMNS_TO_LOAD,
        model=LinearRegressionPredictor(fit_intercept=True),
        config=config,
    )

    training_manager.run_training()
    training_manager.save_model()


if __name__ == "__main__":
    run()
