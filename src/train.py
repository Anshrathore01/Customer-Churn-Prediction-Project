import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from utils import load_data, save_model
from preprocess import clean_raw_data, get_preprocessor

def train_model(data_path, model_save_path):
    print("Loading data...")
    df = load_data(data_path)
    
    print("Performing basic data cleaning...")
    df_cleaned = clean_raw_data(df)
    
    X = df_cleaned.drop("Churn", axis=1)
    y = df_cleaned["Churn"]
    
    # Split BEFORE fitting any transformer/scaler to prevent data leakage
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Building preprocessing and model pipeline...")
    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Fitting model pipeline on training data...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model on test data...")
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model pipeline to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_model(pipeline, model_save_path)
    print("Training complete.")

if __name__ == "__main__":
    # Ensure correct relative path when running from src directory vs root directory
    base_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
    data_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model_save_path = os.path.join(base_dir, "models", "churn_model.pkl")
    train_model(data_path, model_save_path)
