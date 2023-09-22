""" Training class to train basic models """
from typing import List, Tuple

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

from src.data_modeling.data_loading import load_mysql_house_details
from src.models.linear_regression import LinearRegressionPredictor
from src.training.interfaces import PredictorBase, TrainingBase
from src.utils.metric_utils import compute_correlation
from src.utils.train_utils import compare_models, get_production_model
from src.utils.utils import (
    FEATURES_TO_FLOAT,
    INPUT_FEATURES,
    MLFLOW_CONFIG,
    TARGET_FEATURE,
    TRAIN_TEST_SPLIT,
    TRAIN_VAL_SPLIT,
)


class TrainingManagerPlain(TrainingBase):  # pylint: disable=too-few-public-methods
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
        return load_mysql_house_details()

    def _process_data(self) -> None:
        """
        Method to process the raw data
        """

        self.processed_data = self.raw_data.dropna()
        self.processed_data = self.processed_data.sort_values(
            ["publish_unix_time"], ascending=True
        )
        self.processed_data["num_rooms"] = (
            self.processed_data["num_living_rooms"]
            + self.processed_data["num_bathrooms"]
            + self.processed_data["num_bedrooms"]
        )
        self.processed_data[FEATURES_TO_FLOAT] = self.processed_data[
            FEATURES_TO_FLOAT
        ].astype(float)

    @staticmethod
    def _train_val_test_split(
        processed_data: DataFrame,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Divide the data into train, validation and testing datasets
        :param processed_data: the processed data to divide
        :return: the tuple of both, train and test datasets
        """

        idx_train = int(TRAIN_TEST_SPLIT * processed_data.shape[0])
        train_val_data = processed_data.iloc[:idx_train, :].copy()
        test_data = processed_data.iloc[idx_train:, :].copy()
        idx_train = int(TRAIN_VAL_SPLIT * train_val_data.shape[0])
        train_data = train_val_data.iloc[:idx_train, :].copy()
        val_data = train_val_data.iloc[idx_train:, :].copy()
        return train_data, val_data, test_data

    def _fit_predictor(self, x_train: DataFrame, y_train: DataFrame) -> None:
        """
        Method to fit the model with the processed data
        :param x_train: data to be used for training
        :param y_train: target feature to train the model
        """
        params = {"variables": ", ".join(self.input_variables)}

        params = self.model.fit(x_train=x_train, y_train=y_train, params=params)
        mlflow.log_params(params)

    def _log_results(self, x_val: DataFrame, y_val: DataFrame) -> None:
        """
        Method to calculate and log the metrics
        :param x_val: the data to predict the validation results
        :param y_val: the true values to compare
        """
        y_pred = self.model.predict(x_val)
        rmse = mean_squared_error(y_true=y_val, y_pred=y_pred, squared=False)
        correlation = compute_correlation(y_val, y_pred)

        print(f"Model rmse: {rmse:.4f} and correlation: {correlation:.4f}")

        mlflow.log_metrics({"rmse": rmse, "correlation": correlation})

    def run_training(self) -> None:
        """
        Main method to run the model training
        """

        self._process_data()

        train_data, val_data, test_data = self._train_val_test_split(
            self.processed_data
        )

        x_train = train_data[self.input_variables]
        y_train = train_data[TARGET_FEATURE]

        x_val = val_data[self.input_variables]
        y_val = val_data[TARGET_FEATURE]

        x_test = test_data[self.input_variables]
        y_test = test_data[TARGET_FEATURE]

        mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
        run_name = MLFLOW_CONFIG["run_name"]
        experiment_id = mlflow.get_experiment_by_name(
            MLFLOW_CONFIG["experiment_name"]
        ).experiment_id
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            self._fit_predictor(x_train=x_train, y_train=y_train)
            self._log_results(x_val=x_val, y_val=y_val)

        self._save_model(x_test=x_test, y_test=y_test)

    def _save_model(self, x_test: DataFrame, y_test: DataFrame) -> None:
        """
        Method to save or register the trained model
        :param x_test: the data to predict the test results
        :param y_test: the true values to compare
        """
        client = MlflowClient()
        model_name = self.model.model.__class__.__name__

        model_prod = get_production_model(model_name=model_name)

        model_version_staging = client.get_latest_versions(
            model_name, stages=["Staging"]
        )

        if model_prod is None:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version_staging[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(
                "New model transitioned to production as no other production model "
                "existed"
            )
            return

        rmse_staging, rmse_prod = compare_models(
            new_model=self.model.model,
            old_model=model_prod,
            x_test=x_test,
            y_test=y_test,
        )

        if rmse_prod > rmse_staging:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version_staging[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(
                f"New model transitioned to production as the performance is better "
                f"than the old model. New model RMSE: {rmse_staging} - Old model "
                f"RMSE: {rmse_prod}"
            )
        else:
            print(
                f"Kept old production model because the new model did not improve "
                f"the performance. New model RMSE: {rmse_staging} - Old model "
                f"RMSE: {rmse_prod}"
            )


def run() -> None:
    """
    Main method to run the training manager
    """

    training_manager = TrainingManagerPlain(
        input_variables=INPUT_FEATURES,
        model=LinearRegressionPredictor(fit_intercept=True),
    )

    training_manager.run_training()


if __name__ == "__main__":
    run()
