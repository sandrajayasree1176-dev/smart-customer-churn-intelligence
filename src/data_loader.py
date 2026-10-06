import os
import re
import pandas as pd
import numpy as np

# Comprehensive target column alias mapping
TARGET_ALIASES = [
    "churn", "churned", "exited", "exit", "customerstatus", 
    "customer_status", "ischurn", "is_churn", "target", "churnflag", "churn_flag"
]

# Comprehensive feature column alias mapping
COLUMN_ALIASES = {
    "customerid": "customerID",
    "customer_id": "customerID",
    "id": "customerID",
    "cust_id": "customerID",
    "tenure": "tenure",
    "months": "tenure",
    "tenure_months": "tenure",
    "contract": "Contract",
    "contracttype": "Contract",
    "contract_type": "Contract",
    "monthlycharges": "MonthlyCharges",
    "monthly_charges": "MonthlyCharges",
    "monthlyrevenue": "MonthlyCharges",
    "monthly_revenue": "MonthlyCharges",
    "revenue": "MonthlyCharges",
    "charge": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "total_charges": "TotalCharges",
    "internetservice": "InternetService",
    "internet_service": "InternetService",
    "techsupport": "TechSupport",
    "tech_support": "TechSupport",
    "churn": "Churn",
    "target": "Churn",
    "exited": "Churn",
    "churn_status": "Churn",
    "seniorcitizen": "SeniorCitizen",
    "senior_citizen": "SeniorCitizen",
    "gender": "gender",
    "partner": "Partner",
    "dependents": "Dependents",
    "phoneservice": "PhoneService",
    "phone_service": "PhoneService",
    "multiplelines": "MultipleLines",
    "multiple_lines": "MultipleLines",
    "onlinesecurity": "OnlineSecurity",
    "online_security": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "online_backup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "device_protection": "DeviceProtection",
    "streamingtv": "StreamingTV",
    "streaming_tv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "streaming_movies": "StreamingMovies",
    "paperlessbilling": "PaperlessBilling",
    "paperless_billing": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "payment_method": "PaymentMethod"
}

def load_dataset(file_or_path, is_sample=False):
    """
    Loads dataset from uploaded file object or file path (CSV or Excel).
    Returns raw DataFrame.
    """
    if df_raw_check(file_or_path):
        return file_or_path
    
    if is_sample or isinstance(file_or_path, str):
        path = file_or_path if isinstance(file_or_path, str) else "data/customer_data.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path '{path}' does not exist.")
        if str(path).lower().endswith('.csv'):
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)

    # Streamlit UploadedFile object
    filename = getattr(file_or_path, "name", "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file_or_path)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    else:
        try:
            return pd.read_csv(file_or_path)
        except Exception:
            return pd.read_excel(file_or_path)

def df_raw_check(obj):
    return isinstance(obj, pd.DataFrame)

def normalize_column_names(df):
    """
    Renames columns to standardized names matching internal model schema.
    Deduplicates columns if multiple map to the same name.
    Does NOT mutate original DataFrame.
    """
    df_clean = df.copy()
    renamed = {}
    for col in df_clean.columns:
        norm_col = re.sub(r'[\s_]+', '', str(col)).lower()
        if norm_col in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[norm_col]
    if renamed:
        df_clean.rename(columns=renamed, inplace=True)

    # Deduplicate column names if any collision
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()].copy()
    return df_clean

def detect_target_column(df):
    """
    Detects target churn column in DataFrame.
    Returns (target_column_name, has_target_boolean).
    """
    for col in df.columns:
        norm_col = re.sub(r'[\s_]+', '', str(col)).lower()
        if norm_col in TARGET_ALIASES:
            return col, True
    return None, False

def inspect_dataset_metadata(df):
    """
    Computes summary metadata dictionary for dataset analysis dashboard.
    """
    df_norm = normalize_column_names(df)
    target_col, has_target = detect_target_column(df_norm)
    
    num_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_norm.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if "customerID" in num_cols:
        num_cols.remove("customerID")
    if "customerID" in cat_cols:
        cat_cols.remove("customerID")
    if has_target:
        if target_col in num_cols: num_cols.remove(target_col)
        if target_col in cat_cols: cat_cols.remove(target_col)

    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    return {
        "num_records": len(df),
        "num_features": df.shape[1],
        "has_target": has_target,
        "target_col": target_col if has_target else "None (Prediction Mode)",
        "numeric_cols_count": len(num_cols),
        "categorical_cols_count": len(cat_cols),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows
    }
