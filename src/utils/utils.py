""" Main constants and parameters """
import os

from mlflow.types import ColSpec, Schema

MYSQL_DETAILS = {
    "username": os.environ["AWS_RDS_USER"],
    "password": os.environ["AWS_RDS_PASS"],
    "host": "house-price-prediction.cmepw22rfdvm.eu-west-1.rds.amazonaws.com",
    "port": "3306",
    "schema": "house-price-prediction",
}

COLUMNS_TO_LOAD = [
    "listing_id",
    "price",
    "floor_m_sqrt",
    "publish_unix_time",
    "garden",
    "parking",
    "balcony_terrace",
    "furnished",
    "num_living_rooms",
    "num_bathrooms",
    "num_bedrooms",
    "is_auction",
    "is_share_ownership",
    "is_retirement",
    "chain_free",
]

DATA_TO_LOAD_MAP = {
    "listing_id": ["listing_id"],
    "price": ["price"],
    "floor_m_sqrt": ["floor_m_sqrt"],
    "publish_unix_time": ["publish_unix_time"],
    "garden": ["garden"],
    "parking": ["parking"],
    "balcony_terrace": ["balcony_terrace"],
    "furnished": ["furnished"],
    "num_living_rooms": ["num_living_rooms"],
    "num_bathrooms": ["num_bathrooms"],
    "num_bedrooms": ["num_bedrooms"],
    "is_auction": ["is_auction"],
    "is_share_ownership": ["is_share_ownership"],
    "is_retirement": ["is_retirement"],
    "chain_free": ["chain_free"],
}

MLFLOW_CONFIG = {"experiment_name": "experiment_trial", "run_name": "run_trial"}

TRACKING_SERVER_HOST = "http://ec2-34-244-64-152.eu-west-1.compute.amazonaws.com"

INPUT_FEATURES = [
    "floor_m_sqrt",
    "garden",
    "parking",
    "balcony_terrace",
    "furnished",
    "num_rooms",
    "is_auction",
    "num_living_rooms",
    "num_bathrooms",
    "num_bedrooms",
    "is_share_ownership",
    "is_retirement",
    "chain_free",
]

FEATURES_TO_FLOAT = [
    "num_rooms",
    "is_auction",
    "num_living_rooms",
    "num_bathrooms",
    "num_bedrooms",
    "is_share_ownership",
    "is_retirement",
    "chain_free",
]

INPUT_SCHEMA = Schema(
    [
        ColSpec("float", "floor_m_sqrt"),
        ColSpec("boolean", "garden"),
        ColSpec("boolean", "parking"),
        ColSpec("boolean", "balcony_terrace"),
        ColSpec("boolean", "furnished"),
        ColSpec("integer", "num_rooms"),
        ColSpec("boolean", "is_auction"),
        ColSpec("integer", "num_living_rooms"),
        ColSpec("integer", "num_bathrooms"),
        ColSpec("integer", "num_bedrooms"),
        ColSpec("boolean", "is_share_ownership"),
        ColSpec("boolean", "is_retirement"),
        ColSpec("boolean", "chain_free"),
    ]
)

OUTPUT_SCHEMA = Schema([ColSpec("float", "price")])

TARGET_FEATURE = ["price"]

DAYS_OF_DATA_TO_LOAD = 365

TRAIN_TEST_SPLIT = 0.99

TRAIN_VAL_SPLIT = 0.95

MODEL_TRAINING_TIME_WINDOW_IN_SECONDS = 60 * 60 * 24 * 365
