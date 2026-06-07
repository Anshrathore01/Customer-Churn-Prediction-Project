import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def clean_raw_data(df):
    """
    Perform basic data cleaning: drop customerID, convert TotalCharges to numeric,
    and convert Churn to binary if present.
    Does not perform scaling, imputation, or dummy encoding (to avoid data leakage).
    """
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop("customerID", axis=1)
        
    if 'TotalCharges' in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
    return df

def get_preprocessor():
    """
    Constructs and returns the scikit-learn ColumnTransformer for preprocessing.
    """
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    bin_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    pass_cols = ["SeniorCitizen"]

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    bin_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    multi_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('bin', bin_transformer, bin_cols),
            ('multi', multi_transformer, multi_cols),
            ('pass', 'passthrough', pass_cols)
        ]
    )
    return preprocessor
