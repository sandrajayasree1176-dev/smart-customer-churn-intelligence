import os
import sys
import pandas as pd
import numpy as np

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import normalize_column_names, detect_target_column

NUMERIC_FEATURES_DEFAULT = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES_DEFAULT = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
TARGET_COLUMN_DEFAULT = "Churn"

def preprocess_dataframe(df):
    """
    Cleans raw DataFrame:
    - Normalizes column names & removes duplicates
    - Parses string numbers (TotalCharges, MonthlyCharges, tenure)
    - Imputes missing numerical and categorical values
    - Guarantees essential feature columns exist with fallbacks
    - Does NOT mutate input dataframe
    """
    df_clean = normalize_column_names(df)
    
    # Handle Customer ID
    if "customerID" not in df_clean.columns:
        df_clean["customerID"] = [f"CUST-{i+1:04d}" for i in range(len(df_clean))]
    else:
        df_clean["customerID"] = df_clean["customerID"].astype(str).str.strip()

    # Parse numeric features
    for num_col in ["MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"]:
        if num_col in df_clean.columns:
            series = df_clean[num_col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if series.dtype == object or str(series.dtype) in ['string', 'category']:
                cleaned_series = (
                    series
                    .astype(str)
                    .str.replace(r'[\$,₹\s]', '', regex=True)
                    .str.replace(',', '', regex=False)
                )
                df_clean[num_col] = pd.to_numeric(cleaned_series, errors="coerce")
            else:
                df_clean[num_col] = pd.to_numeric(series, errors="coerce")

    # Tenure imputation
    if "tenure" in df_clean.columns:
        s_tenure = df_clean["tenure"]
        if isinstance(s_tenure, pd.DataFrame): s_tenure = s_tenure.iloc[:, 0]
        df_clean["tenure"] = s_tenure.fillna(s_tenure.median() if not s_tenure.dropna().empty else 1.0)
    else:
        df_clean["tenure"] = 1.0

    # MonthlyCharges imputation
    if "MonthlyCharges" in df_clean.columns:
        s_mc = df_clean["MonthlyCharges"]
        if isinstance(s_mc, pd.DataFrame): s_mc = s_mc.iloc[:, 0]
        df_clean["MonthlyCharges"] = s_mc.fillna(s_mc.median() if not s_mc.dropna().empty else 50.0)
    else:
        df_clean["MonthlyCharges"] = 50.0

    # TotalCharges imputation
    if "TotalCharges" in df_clean.columns:
        s_tc = df_clean["TotalCharges"]
        if isinstance(s_tc, pd.DataFrame): s_tc = s_tc.iloc[:, 0]
        null_mask = s_tc.isnull()
        df_clean.loc[null_mask, "TotalCharges"] = df_clean.loc[null_mask, "MonthlyCharges"] * df_clean.loc[null_mask, "tenure"]
    else:
        df_clean["TotalCharges"] = df_clean["MonthlyCharges"] * df_clean["tenure"]

    # SeniorCitizen imputation
    if "SeniorCitizen" in df_clean.columns:
        s_sc = df_clean["SeniorCitizen"]
        if isinstance(s_sc, pd.DataFrame): s_sc = s_sc.iloc[:, 0]
        df_clean["SeniorCitizen"] = s_sc.fillna(0).astype(int)
    else:
        df_clean["SeniorCitizen"] = 0

    # Categorical imputation & cleaning
    for cat_col in CATEGORICAL_FEATURES_DEFAULT:
        if cat_col in df_clean.columns:
            s_cat = df_clean[cat_col]
            if isinstance(s_cat, pd.DataFrame): s_cat = s_cat.iloc[:, 0]
            df_clean[cat_col] = s_cat.fillna("Unknown").astype(str).str.strip()
        else:
            df_clean[cat_col] = "Unknown"

    # Detect Target Column if available
    target_col, has_target = detect_target_column(df_clean)
    if has_target:
        s_target = df_clean[target_col]
        if isinstance(s_target, pd.DataFrame): s_target = s_target.iloc[:, 0]
        raw_target = s_target.astype(str).str.strip().str.lower()
        churn_positive = ["yes", "1", "true", "churned", "exit", "exited", "high risk"]
        df_clean["Churn_Binary"] = raw_target.apply(lambda val: 1 if val in churn_positive else 0)
        df_clean["Churn"] = df_clean["Churn_Binary"].apply(lambda v: "Yes" if v == 1 else "No")
    
    return df_clean

def extract_features(df_clean):
    """
    Extracts feature matrix X and target y (if available).
    Returns (X, y_or_none, feature_names_dict).
    """
    numeric_features = [col for col in NUMERIC_FEATURES_DEFAULT if col in df_clean.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES_DEFAULT if col in df_clean.columns]
    
    X = df_clean[numeric_features + categorical_features].copy()
    
    y = None
    if "Churn_Binary" in df_clean.columns:
        y = df_clean["Churn_Binary"].values
        
    return X, y, {
        "numeric": numeric_features,
        "categorical": categorical_features
    }
