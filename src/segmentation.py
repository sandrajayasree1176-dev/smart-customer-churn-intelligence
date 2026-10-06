import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def perform_customer_segmentation(df_results, max_clusters=5):
    """
    Performs unsupervised K-Means clustering on customer dataset.
    Uses silhouette score to find optimal cluster count.
    Labels segments with human-readable business names based on CLV & Risk profile.
    """
    df_calc = df_results.copy()
    
    # Select features for clustering
    features = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Probability"]
    for col in features:
        if col not in df_calc.columns:
            if col == "Churn_Probability": df_calc[col] = 0.5
            elif col == "TotalCharges": df_calc[col] = df_calc.get("MonthlyCharges", 50) * df_calc.get("tenure", 12)
            elif col == "MonthlyCharges": df_calc[col] = 50.0
            elif col == "tenure": df_calc[col] = 12.0

    X = df_calc[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = 4
    best_score = -1.0

    # Determine optimal k between 2 and max_clusters
    for k in range(2, min(max_clusters + 1, len(df_calc))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Final KMeans model
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_calc["Cluster_ID"] = kmeans_final.fit_predict(X_scaled)

    # Name clusters dynamically based on average metrics
    cluster_profiles = df_calc.groupby("Cluster_ID").agg({
        "Churn_Probability": "mean",
        "MonthlyCharges": "mean",
        "tenure": "mean"
    }).reset_index()

    segment_names = {}
    for _, row in cluster_profiles.iterrows():
        cid = int(row["Cluster_ID"])
        prob = row["Churn_Probability"]
        charge = row["MonthlyCharges"]
        tenure = row["tenure"]

        if prob >= 0.55 and charge >= 65:
            segment_names[cid] = "Critical High-Value At-Risk"
        elif prob >= 0.55:
            segment_names[cid] = "Budget High-Risk Churners"
        elif tenure >= 36 and charge >= 65:
            segment_names[cid] = "Loyal Champions"
        elif tenure <= 12 and prob < 0.40:
            segment_names[cid] = "New Low-Risk Growth Leads"
        else:
            segment_names[cid] = "Stable Core Customers"

    df_calc["Customer_Segment"] = df_calc["Cluster_ID"].map(segment_names)
    
    return df_calc, best_k, round(best_score, 3)

def render_segmentation_scatter(df_segmented):
    """
    Renders 2D Plotly scatter chart visualizing customer segments across Tenure vs Monthly Charges.
    """
    fig = px.scatter(
        df_segmented,
        x="tenure",
        y="MonthlyCharges",
        color="Customer_Segment",
        size="Churn_Probability",
        hover_data=["customerID", "Contract", "Risk_Level"],
        title="Customer Micro-Segmentation Profile (Tenure vs Monthly Charges)",
        labels={"tenure": "Customer Tenure (Months)", "MonthlyCharges": "Monthly Bill ($)"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=420)
    return fig
