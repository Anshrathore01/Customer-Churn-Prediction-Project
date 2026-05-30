import pandas as pd
import joblib

def load_data(filepath):
    """Loads CSV data from filepath."""
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """Saves model to filepath."""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Loads model from filepath."""
    return joblib.load(filepath)
