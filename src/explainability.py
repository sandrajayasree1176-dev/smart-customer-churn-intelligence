import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine import load_model_bundle

def get_global_explanations():
    """
    Returns global feature importance breakdown as DataFrame and Plotly Figure.
    """
    bundle = load_model_bundle()
    feat_df = bundle.get("feature_importance", pd.DataFrame())
    
    if feat_df.empty:
        return pd.DataFrame(), None

    top_10 = feat_df.head(10).copy()
    top_10 = top_10.sort_values(by="importance", ascending=True)

    fig = px.bar(
        top_10,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Factors Driving Churn Across All Customers (Global XAI)",
        color="importance",
        color_continuous_scale="Reds",
        labels={"importance": "Global Feature Impact Score", "feature": "Customer Feature"}
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=380)
    
    return top_10, fig

def get_customer_local_explanation(row, top_n=5):
    """
    Calculates customer-level feature contribution breakdown explaining WHY the specific customer is predicted to churn.
    Uses heuristic feature deviation vs population baseline weighted by model feature importances.
    """
    bundle = load_model_bundle()
    feat_df = bundle.get("feature_importance", pd.DataFrame())

    if feat_df.empty:
        return []

    imp_dict = dict(zip(feat_df["feature"], feat_df["importance"]))
    factors = []

    # 1. Contract
    contract = str(row.get("Contract", "")).strip().lower()
    if "month-to-month" in contract:
        factors.append({
            "feature": "Month-to-month contract",
            "impact": "High Risk Factor",
            "direction": "increases_risk",
            "score": imp_dict.get("Contract_Month-to-month", 0.25) * 1.5,
            "description": "Short-term commitment leads to higher cancellation flexibility."
        })
    elif "two year" in contract or "one year" in contract:
        factors.append({
            "feature": "Long-term contract",
            "impact": "Risk Reduction Factor",
            "direction": "decreases_risk",
            "score": -0.15,
            "description": "Longer term commitment provides stability."
        })

    # 2. Tenure
    tenure = float(row.get("tenure", 12))
    if tenure <= 12:
        factors.append({
            "feature": f"Low tenure ({int(tenure)} months)",
            "impact": "High Risk Factor",
            "direction": "increases_risk",
            "score": imp_dict.get("tenure", 0.20) * 1.4,
            "description": "Newer customers (< 1 year) are significantly more prone to switching."
        })
    elif tenure >= 48:
        factors.append({
            "feature": f"High tenure ({int(tenure)} months)",
            "impact": "Risk Reduction Factor",
            "direction": "decreases_risk",
            "score": -0.20,
            "description": "Established long-term customer relationship reduces churn risk."
        })

    # 3. Monthly Charges
    monthly_charges = float(row.get("MonthlyCharges", 50))
    if monthly_charges > 70:
        factors.append({
            "feature": f"High monthly charges (${monthly_charges:.2f})",
            "impact": "High Risk Factor",
            "direction": "increases_risk",
            "score": imp_dict.get("MonthlyCharges", 0.18) * 1.2,
            "description": "Higher bill amounts increase price sensitivity and risk."
        })
    elif monthly_charges < 35:
        factors.append({
            "feature": f"Low monthly charges (${monthly_charges:.2f})",
            "impact": "Risk Reduction Factor",
            "direction": "decreases_risk",
            "score": -0.10,
            "description": "Affordable plan reduces cost-driven churn."
        })

    # 4. Tech Support
    tech_support = str(row.get("TechSupport", "")).strip().lower()
    if "no" in tech_support and "no internet" not in tech_support:
        factors.append({
            "feature": "No tech support service",
            "impact": "Medium Risk Factor",
            "direction": "increases_risk",
            "score": imp_dict.get("TechSupport_No", 0.12),
            "description": "Unresolved technical issues without dedicated support increase frustration."
        })
    elif "yes" in tech_support:
        factors.append({
            "feature": "Has tech support service",
            "impact": "Risk Reduction Factor",
            "direction": "decreases_risk",
            "score": -0.12,
            "description": "Active technical assistance improves customer satisfaction."
        })

    # 5. Internet Service Type
    internet = str(row.get("InternetService", "")).strip().lower()
    if "fiber" in internet:
        factors.append({
            "feature": "Fiber optic internet service",
            "impact": "Medium Risk Factor",
            "direction": "increases_risk",
            "score": imp_dict.get("InternetService_Fiber optic", 0.14),
            "description": "Higher price tier and higher service expectations."
        })

    # 6. Payment Method
    payment = str(row.get("PaymentMethod", "")).strip().lower()
    if "electronic check" in payment:
        factors.append({
            "feature": "Electronic check payment",
            "impact": "Medium Risk Factor",
            "direction": "increases_risk",
            "score": imp_dict.get("PaymentMethod_Electronic check", 0.10),
            "description": "Manual payment methods show higher churn volatility than auto-pay."
        })
    elif "automatic" in payment:
        factors.append({
            "feature": "Automatic payment setup",
            "impact": "Risk Reduction Factor",
            "direction": "decreases_risk",
            "score": -0.12,
            "description": "Automated recurring payment lowers friction."
        })

    factors.sort(key=lambda x: abs(x["score"]), reverse=True)
    return factors[:top_n]

def render_waterfall_chart(customer_id, factors):
    """
    Renders horizontal waterfall/bar chart showing positive & negative risk drivers for a single customer.
    """
    if not factors:
        return None

    df_factors = pd.DataFrame(factors)
    colors = ["#EF4444" if d == "increases_risk" else "#22C55E" for d in df_factors["direction"]]
    
    fig = go.Figure(go.Bar(
        x=df_factors["score"],
        y=df_factors["feature"],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{s:+.2f}" for s in df_factors["score"]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Customer {customer_id}: Specific Churn Impact Drivers",
        xaxis_title="Impact on Churn Probability (+ Increase Risk / - Reduce Risk)",
        yaxis_title="Feature Factor",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    return fig
