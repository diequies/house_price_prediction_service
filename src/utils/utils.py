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

DATA_TO_LOAD_MAP = {
    "listing_id": ["listing_id"],
    "price": ["price"],
    "floor_m_sqrt": ["floor_m_sqrt"],
}

MLFLOW_CONFIG = {"experiment_name": "experiment_trial", "run_name": "run_trial"}

TRACKING_SERVER_HOST = "http://ec2-34-244-64-152.eu-west-1.compute.amazonaws.com"

INPUT_FEATURES = ["floor_m_sqrt"]

INPUT_SCHEMA = {"floor_m_sqrt": "float"}

OUTPUT_SCHEMA = {"price": "float"}

TARGET_FEATURE = ["price"]

DAYS_OF_DATA_TO_LOAD = 180

TRAIN_TEST_SPLIT = 0.99

TRAIN_VAL_SPLIT = 0.95

MODEL_TRAINING_TIME_WINDOW_IN_SECONDS = 60 * 60 * 24 * 365
