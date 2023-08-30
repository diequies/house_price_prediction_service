"""Source code of your project"""
import os
import sys

sys.path.append("./src")
os.environ["AWS_RDS_USER"] = "mocked_user"
os.environ["AWS_RDS_PASS"] = "mocked_pass"
