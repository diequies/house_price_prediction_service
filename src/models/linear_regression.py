""" Linear Regressor model """

from typing import List

import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from src.training.interfaces import PredictorBase


class LinearRegressionPredictor(PredictorBase):
    """
    Linear Regression model to predict house prices
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Constructor method for the Linear Regression Model
        :param fit_intercept: Whether to calculate the intercept
        """
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, x_train: DataFrame, y_train: Series) -> None:
        """
        Trains the linear regression model
        :param x_train: Processed data to use to train the model
        :param y_train: Target feature
        """
        self.model = self.model.fit(X=x_train, y=y_train)

    def predict(self, x_predict: DataFrame) -> List:
        """
        Method to predict house prices using the linear regression model
        :param x_predict: Data to use to predict
        :return: List of predicted values
        """
        y_predict = self.model.predict(X=x_predict)
        return pd.Series(y_predict).tolist()
