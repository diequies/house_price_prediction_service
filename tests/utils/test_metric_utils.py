""" Test for the metrics methods"""
import numpy as np
import pandas as pd

from src.utils.metric_utils import compute_correlation


def test_base_compute_correlation():
    """
    Tests that the basic correlation calculation works well
    """
    y_true = pd.DataFrame([100, 200, 300, 400, 500])
    y_pred = [75, 175, 325, 375, 425]
    y_pred = [np.array(i).reshape(-1) for i in y_pred]

    correlation = compute_correlation(y_true=y_true, y_pred=y_pred)

    assert round(correlation, 3) == 0.976
