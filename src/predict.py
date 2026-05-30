import pandas as pd
from utils import load_model

def predict_churn(model_path, data):
    """
    Makes a prediction given a model and a dataframe row matching the training features.
    """
    model = load_model(model_path)
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]
    
    return prediction, probability

if __name__ == "__main__":
    # Example usage
    print("This script is meant to be imported. Use the function predict_churn()")
