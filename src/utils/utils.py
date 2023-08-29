""" Main constants and parameters """
import os
from typing import Dict, Union

MYSQL_DETAILS = {
    "username": os.environ["AWS_RDS_USER"],
    "password": os.environ["AWS_RDS_PASS"],
    "host": "house-price-prediction.cmepw22rfdvm.eu-west-1.rds.amazonaws.com",
    "port": "3306",
    "schema": "house-price-prediction",
}

COLUMNS_TO_LOAD = ["listing_id", "price", "floor_m_sqrt"]

DATA_TO_LOAD_MAP = {
    "listing_id": ["listing_id"],
    "price": ["price"],
    "floor_m_sqrt": ["floor_m_sqrt"],
}

INPUT_FEATURES = ["listing_id", "floor_m_sqrt"]

TARGET_FEATURE = ["price"]

CONFIG: Dict[str, Union[int, float]] = {
    "data_path": "../../data/raw",
    "data_filename": "dummy_data.csv",
    "days_of_data": 180,
    "train_test_split": 0.95,
}

MODEL_TRAINING_TIME_WINDOW_IN_SECONDS = 60 * 60 * 24 * 365
