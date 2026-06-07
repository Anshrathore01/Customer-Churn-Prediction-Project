import pandas as pd
from utils import load_model
from preprocess import clean_raw_data

def predict_churn(model_path, data):
    """
    Makes a prediction given a model pipeline and a dataframe row matching the raw features.
    Handles data cleaning before passing it to the model pipeline.
    """
    model = load_model(model_path)
    cleaned_data = clean_raw_data(data)
    prediction = model.predict(cleaned_data)
    probability = model.predict_proba(cleaned_data)[:, 1]
    
    return prediction, probability

if __name__ == "__main__":
    print("This script is meant to be imported. Use the function predict_churn()")
