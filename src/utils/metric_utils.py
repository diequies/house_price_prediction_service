""" Methods to compute the training metrics """
from typing import List

import numpy as np
from pandas import DataFrame


def compute_correlation(y_true: DataFrame, y_pred: List) -> float:
    """
    Method to calculate the correlation between the predicted and true target variables
    :param y_true: True target variable as a Dataframe
    :param y_pred: Predicted target variable as a list
    :return: The correlation value
    """
    pred_values = y_true.values.flatten()
    correlation_val = np.corrcoef([i[0] for i in y_pred], pred_values)[0][1]

    return correlation_val
