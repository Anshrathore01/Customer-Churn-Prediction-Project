import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from utils import load_data, save_model
from preprocess import preprocess_data

def train_model(data_path, model_save_path):
    print("Loading data...")
    df = load_data(data_path)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = lr.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_model(lr, model_save_path)
    print("Training complete.")

if __name__ == "__main__":
    train_model(
        data_path="../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_save_path="../models/churn_model.pkl"
    )
