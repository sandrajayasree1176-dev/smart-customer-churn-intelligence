import os
import sys

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine import train_and_compare_models, predict_churn, load_model_bundle

def train_model(data_path="data/customer_data.csv"):
    """
    Legacy wrapper function for training model pipeline.
    """
    return train_and_compare_models(data_path)

def predict(df):
    """
    Legacy wrapper function for predicting churn on dataframe.
    """
    return predict_churn(df)
