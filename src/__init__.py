"""Source code of your project"""
import sys

import mlflow
import pymysql

from src.utils.utils import TRACKING_SERVER_HOST

mlflow.set_tracking_uri(f"{TRACKING_SERVER_HOST}:5050")
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")
sys.path.append("./src")
pymysql.install_as_MySQLdb()
