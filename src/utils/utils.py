""" Main constants and parameters """
import os

MYSQL_DETAILS = {
    "username": os.environ["AWS_RDS_USER"],
    "password": os.environ["AWS_RDS_PASS"],
    "host": "house-price-prediction.cmepw22rfdvm.eu-west-1.rds.amazonaws.com",
    "port": "3306",
    "schema": "house-price-prediction",
}

COLUMNS_TO_LOAD = ["listing_id", "price", "floor_m_sqrt"]

INPUT_FEATURES = ["listing_id", "floor_m_sqrt"]

TARGET_FEATURE = ["price"]

CONFIG = {
    "data_path": "../../data/raw",
    "data_filename": "dummy_data.csv",
    "train_test_split": 0.95,
}

MODEL_TRAINING_TIME_WINDOW_IN_SECONDS = 60 * 60 * 24 * 365
