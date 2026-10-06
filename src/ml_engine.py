import os
import sys

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from preprocessing import preprocess_dataframe, NUMERIC_FEATURES_DEFAULT, CATEGORICAL_FEATURES_DEFAULT, TARGET_COLUMN_DEFAULT
from data_loader import load_dataset

MODEL_DIR = os.path.join("outputs", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

def get_preprocessor(numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES_DEFAULT
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES_DEFAULT

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def train_and_compare_models(data_or_path="data/customer_data.csv", selection_metric="f1"):
    """
    Trains multiple models (Random Forest, Logistic Regression, Gradient Boosting, Decision Tree),
    evaluates Accuracy, Precision, Recall, F1, ROC-AUC, selects the best model, and saves model bundle.
    """
    if isinstance(data_or_path, str):
        df_raw = load_dataset(data_or_path)
    elif isinstance(data_or_path, pd.DataFrame):
        df_raw = data_or_path
    else:
        df_raw = load_dataset("data/customer_data.csv")

    df_clean = preprocess_dataframe(df_raw)
    
    if "Churn_Binary" not in df_clean.columns:
        raise ValueError("Target column 'Churn' not found in dataset. Cannot train supervised ML models without target labels.")

    y = df_clean["Churn_Binary"].values
    
    numeric_features = [col for col in NUMERIC_FEATURES_DEFAULT if col in df_clean.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES_DEFAULT if col in df_clean.columns]
    
    X = df_clean[numeric_features + categorical_features]
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    }

    comparison_results = []
    trained_pipelines = {}

    best_score = -1.0
    best_model_name = None
    best_pipeline = None

    for name, clf in models.items():
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5

        res = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        }
        comparison_results.append(res)
        trained_pipelines[name] = pipeline

        # Score selection
        current_score = f1 if selection_metric == "f1" else auc
        if current_score > best_score:
            best_score = current_score
            best_model_name = name
            best_pipeline = pipeline

    comparison_df = pd.DataFrame(comparison_results).sort_values(
        by="F1 Score" if selection_metric == "f1" else "ROC-AUC", ascending=False
    )

    # Feature importances extraction from best model
    feature_importance_df = extract_feature_importance(best_pipeline, numeric_features, categorical_features)

    # Save model bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_metrics = comparison_df[comparison_df["Model"] == best_model_name].iloc[0].to_dict()

    bundle = {
        'pipeline': best_pipeline,
        'best_model_name': best_model_name,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'metrics': best_metrics,
        'comparison_df': comparison_df,
        'feature_importance': feature_importance_df
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"Best model '{best_model_name}' trained and saved to {MODEL_PATH}")
    return bundle

def extract_feature_importance(pipeline, numeric_features, categorical_features):
    """
    Extracts feature importances or coefficients from fitted sklearn pipeline.
    """
    clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
        ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
    except Exception:
        ohe_feature_names = categorical_features
        
    all_feature_names = numeric_features + ohe_feature_names

    importances = None
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])

    if importances is not None and len(importances) == len(all_feature_names):
        df_imp = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return df_imp
    else:
        return pd.DataFrame({
            'feature': all_feature_names,
            'importance': [1.0 / len(all_feature_names)] * len(all_feature_names)
        })

def load_model_bundle():
    """
    Loads saved model bundle or trains default model if missing.
    """
    if not os.path.exists(MODEL_PATH):
        print("Model file missing. Initiating model training on sample dataset...")
        return train_and_compare_models("data/customer_data.csv")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model bundle: {e}. Retraining...")
        return train_and_compare_models("data/customer_data.csv")

def predict_churn(df_raw, risk_threshold_high=0.65, risk_threshold_med=0.35):
    """
    Runs model inference on any user DataFrame.
    Returns DataFrame populated with:
    - Churn_Probability
    - Risk_Level (High Risk, Medium Risk, Low Risk)
    - Predicted_Churn (Yes / No)
    """
    df_clean = preprocess_dataframe(df_raw)
    bundle = load_model_bundle()
    pipeline = bundle['pipeline']
    numeric_features = bundle['numeric_features']
    categorical_features = bundle['categorical_features']

    # Ensure all required columns exist in df_clean
    for col in numeric_features:
        if col not in df_clean.columns: df_clean[col] = 0.0
    for col in categorical_features:
        if col not in df_clean.columns: df_clean[col] = "Unknown"

    X = df_clean[numeric_features + categorical_features]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[:, 1]
    else:
        probabilities = pipeline.predict(X)

    probabilities = np.clip(probabilities, 0.0, 1.0)
    df_clean["Churn_Probability"] = np.round(probabilities, 4)

    def assign_risk(prob):
        if prob >= risk_threshold_high:
            return "High Risk"
        elif prob >= risk_threshold_med:
            return "Medium Risk"
        else:
            return "Low Risk"

    df_clean["Risk_Level"] = df_clean["Churn_Probability"].apply(assign_risk)
    df_clean["Predicted_Churn"] = df_clean["Risk_Level"].apply(lambda r: "Yes" if r == "High Risk" else "No")

    return df_clean
