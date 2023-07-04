""" Main constants and parameters """


MYSQL_DETAILS = {
    "username": "admin",
    "password": "AWSpricingDB",
    "host": "house-price-prediction.cmepw22rfdvm.eu-west-1.rds.amazonaws.com",
    "port": "3306",
    "schema": "house-price-prediction",
}

COLUMNS_TO_LOAD = ["listing_id", "price"]

MODEL_TRAINING_TIME_WINDOW_IN_SECONDS = 60 * 60 * 24 * 365
