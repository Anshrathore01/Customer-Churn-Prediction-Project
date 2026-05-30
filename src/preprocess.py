import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, is_training=True):
    """
    Preprocess the raw dataframe.
    """
    if 'customerID' in df.columns:
        df = df.drop("customerID", axis=1)
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    
    for col in cat_cols:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
            
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    cat_cols_dummies = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    
    # Filter out columns that don't exist in the df (useful during inference)
    cat_cols_dummies = [c for c in cat_cols_dummies if c in df.columns]
    
    if cat_cols_dummies:
        df = pd.get_dummies(df, columns=cat_cols_dummies, drop_first=True, dtype=int)
        
    return df
