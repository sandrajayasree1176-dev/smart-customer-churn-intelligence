import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, inspect_dataset_metadata, detect_target_column
from preprocessing import preprocess_dataframe
from ml_engine import load_model_bundle, train_and_compare_models, predict_churn
from explainability import get_global_explanations, get_customer_local_explanation, render_waterfall_chart
from financial_engine import calculate_clv, calculate_revenue_at_risk, calculate_priority_scores, optimize_retention_offer, calculate_aggregate_retention_roi
from retention_engine import apply_retention_engine
from segmentation import perform_customer_segmentation, render_segmentation_scatter
from simulator import simulate_customer_what_if
from ai_assistant import query_churn_intelligence
from report_generator import generate_pdf_report
from generate_data import generate_sample_dataset

# Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Customer Churn Intelligence System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Ultra-Attractive CSS System (Glassmorphism, Modern SaaS Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Header styling */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: #FFFFFF;
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.3), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 50%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 3px;
        color: #38BDF8;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        max-width: 850px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Modern Metric Cards */
    .glass-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.08);
    }

    .metric-value-huge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }

    .metric-label-sub {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    /* Risk Pill Badges */
    .badge-high {
        background-color: #FEF2F2;
        color: #EF4444;
        border: 1px solid #FCA5A5;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    .badge-med {
        background-color: #FFEDD5;
        color: #F97316;
        border: 1px solid #FDBA74;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    .badge-low {
        background-color: #F0FDF4;
        color: #10B981;
        border: 1px solid #6EE7B7;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    /* Custom Chat Bubble */
    .chat-bubble-user {
        background: #2563EB;
        color: #FFFFFF;
        border-radius: 12px 12px 0 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-weight: 500;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }

    .chat-bubble-ai {
        background: #F8FAFC;
        color: #0F172A;
        border: 1px solid #E2E8F0;
        border-radius: 12px 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 14px;
        font-size: 0.93rem;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
    }

    /* Styled Buttons */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
</style>
""", unsafe_allow_html=True)

# Page Top Banner Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">SMART CUSTOMER CHURN INTELLIGENCE SYSTEM</div>
    <div class="hero-subtitle">PREDICT • EXPLAIN • RETAIN • OPTIMIZE • GROW</div>
    <div class="hero-desc">
        An enterprise AI-powered platform transforming raw customer data into explainable churn predictions, 
        financial risk evaluations (CLV/Revenue at Risk), real-time What-If simulations, and optimized retention strategies.
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR DATA MANAGEMENT & MODULE NAVIGATION
st.sidebar.markdown("### 📁 Data Source & Settings")

sample_path = os.path.join("data", "customer_data.csv")
if not os.path.exists(sample_path):
    generate_sample_dataset(sample_path)

use_sample = st.sidebar.checkbox("Use Sample Telco Dataset (5,000 Records)", value=True)
uploaded_file = st.sidebar.file_uploader("📤 Upload Custom Dataset (CSV / Excel)", type=["csv", "xlsx", "xls"])

# Data Ingestion Logic
try:
    if uploaded_file is not None:
        df_raw = load_dataset(uploaded_file)
        st.sidebar.success(f"Loaded Uploaded File: {uploaded_file.name}")
    else:
        df_raw = load_dataset(sample_path, is_sample=True)
        st.sidebar.info("Using Built-in Sample Dataset")
except Exception as e:
    st.sidebar.error(f"Error loading dataset: {e}")
    df_raw = generate_sample_dataset()

# Run Model Inference Engine
with st.spinner("Processing Dataset & Executing AI Model Inference..."):
    df_results = predict_churn(df_raw)
    df_results = calculate_clv(df_results)
    df_results = calculate_priority_scores(df_results)
    df_results = apply_retention_engine(df_results)
    rev_metrics = calculate_revenue_at_risk(df_results)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 System Modules")

nav_mode = st.sidebar.radio(
    "Select View Module:",
    [
        "📊 Executive View",
        "📁 Data Management",
        "📈 Churn Visual Analytics",
        "🤖 ML Engine & Models",
        "🧠 Explainable AI (XAI)",
        "💎 Customer Value & CLV",
        "🎯 Retention Strategy",
        "💵 Offer Optimizer",
        "🔮 What-If Simulator",
        "🚨 Early Warning System",
        "👥 Customer Segmentation",
        "💰 Retention ROI Calculator",
        "💬 AI Churn Assistant",
        "📄 Business Report & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Smart Churn Intelligence v2.0 • Powered by ML & XAI")

# ==========================================
# MODULE 1: 📊 EXECUTIVE VIEW
# ==========================================
if nav_mode == "📊 Executive View":
    st.markdown("### 📊 Executive Overview & Enterprise Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_cust = len(df_results)
    high_risk_count = rev_metrics["count_churners_at_risk"]
    churn_rate = (high_risk_count / total_cust * 100.0) if total_cust > 0 else 0.0
    monthly_risk = rev_metrics["monthly_revenue_at_risk"]
    annual_risk = rev_metrics["annual_revenue_at_risk"]
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label-sub">Total Customer Base</div>
            <div class="metric-value-huge">{total_cust:,}</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Active Accounts Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label-sub">High-Risk Churn Rate</div>
            <div class="metric-value-huge" style="color: #EF4444;">{churn_rate:.1f}%</div>
            <div style="font-size: 0.8rem; color: #EF4444; margin-top: 4px; font-weight: 600;">{high_risk_count:,} At-Risk Accounts</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label-sub">Monthly Revenue at Risk</div>
            <div class="metric-value-huge" style="color: #F97316;">${monthly_risk:,.2f}</div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Direct Monthly Loss</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label-sub">Annual Revenue at Risk</div>
            <div class="metric-value-huge" style="color: #DC2626;">${annual_risk:,.2f}</div>
            <div style="font-size: 0.8rem; color: #DC2626; margin-top: 4px; font-weight: 600;">Projected 12-Month Impact</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1.2, 1])
    
    with c_left:
        st.markdown("#### 🎯 Churn Risk Distribution")
        risk_counts = df_results["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        
        fig_pie = px.pie(
            risk_counts,
            names="Risk_Level",
            values="Count",
            color="Risk_Level",
            color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"},
            hole=0.45
        )
        fig_pie.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=20, b=20), height=330)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_right:
        st.markdown("#### 🚨 Top 5 Critical Priority Customers to Contact")
        top_5 = df_results.sort_values(by="Priority_Score", ascending=False).head(5)
        for _, r in top_5.iterrows():
            cid = r.get("customerID", "N/A")
            score = r.get("Priority_Score", 0)
            prob = float(r.get("Churn_Probability", 0)) * 100
            rev = float(r.get("MonthlyCharges", 0))
            contract = r.get("Contract", "Month-to-month")
            
            st.markdown(f"""
            <div style="background: #F8FAFC; border-left: 4px solid #EF4444; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #0F172A;">Customer ID: {cid}</span>
                    <span class="badge-high">Priority Score: {score:.1f}</span>
                </div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                    Churn Prob: <b>{prob:.1f}%</b> | Monthly Bill: <b>${rev:.2f}</b> | Contract: <b>{contract}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# MODULE 2: 📁 DATA MANAGEMENT
# ==========================================
elif nav_mode == "📁 Data Management":
    st.markdown("### 📁 Data Inspection & Metadata Analysis")
    meta = inspect_dataset_metadata(df_raw)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", f"{meta['num_records']:,}")
    m2.metric("Total Features", f"{meta['num_features']}")
    m3.metric("Target Auto-Detected", meta['target_col'])
    m4.metric("Numeric Features", f"{meta['numeric_cols_count']}")
    m5.metric("Categorical Features", f"{meta['categorical_cols_count']}")
    
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### Raw Dataset Preview (First 20 Records)")
    st.dataframe(df_raw.head(20), use_container_width=True)
    
    st.markdown("#### Dataset Column Summary & Missing Value Audit")
    summary_df = pd.DataFrame({
        "Column": df_raw.columns,
        "Data Type": df_raw.dtypes.astype(str),
        "Null Count": df_raw.isnull().sum().values,
        "Unique Values": [df_raw[c].nunique() for c in df_raw.columns]
    })
    st.dataframe(summary_df, use_container_width=True)

# ==========================================
# MODULE 3: 📈 CHURN VISUAL ANALYTICS
# ==========================================
elif nav_mode == "📈 Churn Visual Analytics":
    st.markdown("### 📈 Visual Churn Analytics & Feature Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Contract & Charges", "Tenure & Services", "Payment Methods"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df_results, x="Contract", color="Risk_Level", barmode="group",
                title="Churn Risk Distribution by Contract Type",
                color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"}
            )
            fig.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            fig = px.box(
                df_results, x="Risk_Level", y="MonthlyCharges", color="Risk_Level",
                title="Monthly Charges Distribution across Churn Risk Tiers",
                color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"}
            )
            fig.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df_results, x="tenure", color="Risk_Level", nbins=20,
                title="Customer Tenure Distribution (Months) vs Risk Tier",
                color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"}
            )
            fig.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            fig = px.histogram(
                df_results, x="InternetService", color="Risk_Level", barmode="group",
                title="Internet Service Type vs Churn Risk",
                color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"}
            )
            fig.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.histogram(
            df_results, x="PaymentMethod", color="Risk_Level", barmode="group",
            title="Payment Method Impact on Churn Risk Tiers",
            color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"}
        )
        fig.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 4: 🤖 ML ENGINE & MODELS
# ==========================================
elif nav_mode == "🤖 ML Engine & Models":
    st.markdown("### 🤖 ML Engine & Multi-Model Benchmark Comparison")
    
    if st.button("🚀 Retrain & Compare All 4 Classifiers"):
        with st.spinner("Training Random Forest, Gradient Boosting, Logistic Regression, and Decision Tree..."):
            bundle = train_and_compare_models(df_raw)
            st.success(f"Model Training Complete! Best Selected Model: **{bundle['best_model_name']}**")
    else:
        bundle = load_model_bundle()

    comp_df = bundle.get("comparison_df", pd.DataFrame())
    
    st.markdown(f"#### Selected Optimal Classifier: **{bundle.get('best_model_name', 'Random Forest')}**")
    st.dataframe(comp_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], color="#DCFCE7"), use_container_width=True)

    if not comp_df.empty:
        fig = px.bar(
            comp_df, x="Model", y=["F1 Score", "ROC-AUC", "Accuracy"],
            barmode="group", title="Algorithm Benchmark Comparison (Metrics Matrix)",
            color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"]
        )
        fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 5: 🧠 EXPLAINABLE AI (XAI)
# ==========================================
elif nav_mode == "🧠 Explainable AI (XAI)":
    st.markdown("### 🧠 Explainable AI (XAI) & Model Interpretability")
    
    top_10, fig_global = get_global_explanations()
    if fig_global is not None:
        st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 👤 Individual Customer-Level Churn Driver Waterfall (Local XAI)")
    
    cust_id_list = df_results["customerID"].tolist()
    selected_cid = st.selectbox("Select Customer ID for Local Explanation:", cust_id_list)
    
    selected_row = df_results[df_results["customerID"] == selected_cid].iloc[0].to_dict()
    factors = get_customer_local_explanation(selected_row)
    
    c_w, c_t = st.columns([1.2, 1])
    with c_w:
        fig_waterfall = render_waterfall_chart(selected_cid, factors)
        if fig_waterfall is not None:
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
    with c_t:
        st.markdown(f"##### Key Risk Factors for Customer **{selected_cid}**")
        prob_pct = float(selected_row.get("Churn_Probability", 0.5)) * 100
        st.markdown(f"**Predicted Churn Probability**: `{prob_pct:.1f}%` ({selected_row.get('Risk_Level', 'N/A')})")
        
        for f in factors:
            color_dot = "🔴" if f["direction"] == "increases_risk" else "🟢"
            st.markdown(f"{color_dot} **{f['feature']}**: {f['description']}")

# ==========================================
# MODULE 6: 💎 CUSTOMER VALUE & CLV
# ==========================================
elif nav_mode == "💎 Customer Value & CLV":
    st.markdown("### 💎 Customer Lifetime Value (CLV) & Revenue Analytics")
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df_results, x="Estimated_CLV", color="Risk_Level", nbins=30,
            title="Estimated Customer Lifetime Value Distribution ($)",
            color_discrete_map={"High Risk": "#EF4444", "Medium Risk": "#F97316", "Low Risk": "#10B981"}
        )
        fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🏆 Top 10 Highest CLV Customers Leaderboard")
        top_clv = df_results.sort_values(by="Estimated_CLV", ascending=False).head(10)[["customerID", "Estimated_CLV", "MonthlyCharges", "tenure", "Risk_Level"]]
        st.dataframe(top_clv, use_container_width=True)

# ==========================================
# MODULE 7: 🎯 RETENTION STRATEGY
# ==========================================
elif nav_mode == "🎯 Retention Strategy":
    st.markdown("### 🎯 Profile-Driven Retention Action Recommendations")
    
    st.markdown("Filter retention recommendations by risk category:")
    selected_risk = st.multiselect("Select Risk Tiers:", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk"])
    
    filtered_df = df_results[df_results["Risk_Level"].isin(selected_risk)]
    display_cols = ["customerID", "Risk_Level", "Churn_Probability", "Contract", "MonthlyCharges", "Personalized_Retention_Action"]
    
    st.dataframe(filtered_df[display_cols], use_container_width=True)

# ==========================================
# MODULE 8: 💵 OFFER OPTIMIZER
# ==========================================
elif nav_mode == "💵 Offer Optimizer":
    st.markdown("### 💵 Economic Offer Optimizer & Interventions Matrix")
    
    st.markdown("Evaluates retention offer options (discounts, free tech support, annual upgrades) against probability uplift, cost, net benefit, and ROI.")
    
    offers_data = [optimize_retention_offer(row) for _, row in df_results.iterrows()]
    df_offers = pd.DataFrame(offers_data)
    combined_df = pd.concat([df_results[["customerID", "Risk_Level", "MonthlyCharges", "Estimated_CLV"]], df_offers], axis=1)
    
    st.dataframe(combined_df.head(25), use_container_width=True)

# ==========================================
# MODULE 9: 🔮 WHAT-IF SIMULATOR
# ==========================================
elif nav_mode == "🔮 What-If Simulator":
    st.markdown("### 🔮 Real-Time What-If Churn Simulator")
    
    st.markdown("Select a customer and modify their attributes to observe real-time model re-inference.")
    
    cust_id = st.selectbox("Select Customer to Simulate:", df_results["customerID"].tolist())
    orig_row = df_results[df_results["customerID"] == cust_id].iloc[0].to_dict()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Modify Customer Parameters")
        new_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0 if "month" in str(orig_row.get("Contract")).lower() else 1)
        new_tenure = st.slider("Tenure (Months)", 1, 72, int(orig_row.get("tenure", 12)))
        new_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, float(orig_row.get("MonthlyCharges", 50.0)))
        new_tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=0 if "yes" in str(orig_row.get("TechSupport")).lower() else 1)
        new_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    mod_attrs = {
        "Contract": new_contract,
        "tenure": new_tenure,
        "MonthlyCharges": new_monthly,
        "TechSupport": new_tech,
        "PaymentMethod": new_payment
    }
    
    sim_res = simulate_customer_what_if(orig_row, mod_attrs)
    
    with col2:
        st.markdown("#### ⚡ Simulation Results")
        
        orig_prob = sim_res["original_probability"] * 100
        sim_prob = sim_res["simulated_probability"] * 100
        delta = sim_res["delta_probability"] * 100
        
        st.metric("Original Churn Probability", f"{orig_prob:.1f}%", f"{sim_res['original_risk_level']}")
        st.metric("Simulated Churn Probability", f"{sim_prob:.1f}%", f"{delta:+.1f}% vs Original", delta_color="inverse")
        
        if delta < 0:
            st.success(f"🎉 **Risk Reduction**: Modifying attributes decreased churn probability by **{abs(delta):.1f}%**!")
        else:
            st.warning(f"⚠️ **Risk Increase**: Attribute modifications increased churn probability by **{abs(delta):.1f}%**.")

# ==========================================
# MODULE 10: 🚨 EARLY WARNING SYSTEM
# ==========================================
elif nav_mode == "🚨 Early Warning System":
    st.markdown("### 🚨 Early Warning System & Automated Risk Triggers")
    
    high_risk_df = df_results[df_results["Risk_Level"] == "High Risk"].sort_values(by="Priority_Score", ascending=False)
    
    st.warning(f"🚨 **{len(high_risk_df):,} Critical High-Risk Accounts Identified**")
    st.dataframe(high_risk_df[["customerID", "Priority_Score", "Churn_Probability", "Contract", "MonthlyCharges", "tenure", "Personalized_Retention_Action"]], use_container_width=True)

# ==========================================
# MODULE 11: 👥 CUSTOMER SEGMENTATION
# ==========================================
elif nav_mode == "👥 Customer Segmentation":
    st.markdown("### 👥 Customer Micro-Segmentation (K-Means Clustering)")
    
    with st.spinner("Computing optimal silhouette score & K-Means customer clusters..."):
        df_seg, k_optimal, sil_score = perform_customer_segmentation(df_results)
        
    st.markdown(f"**Optimal Clusters Found**: `{k_optimal}` | **Silhouette Score**: `{sil_score:.3f}`")
    
    fig_seg = render_segmentation_scatter(df_seg)
    st.plotly_chart(fig_seg, use_container_width=True)

# ==========================================
# MODULE 12: 💰 RETENTION ROI CALCULATOR
# ==========================================
elif nav_mode == "💰 Retention ROI Calculator":
    st.markdown("### 💰 Retention ROI & Financial Impact Calculator")
    
    conv_rate = st.slider("Assumed Retention Conversion Rate (%)", 10, 100, 75) / 100.0
    roi_summary = calculate_aggregate_retention_roi(df_results, assumed_conversion_rate=conv_rate)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Intervention Cost", f"${roi_summary['total_intervention_cost']:,.2f}")
    c2.metric("Total Revenue Saved", f"${roi_summary['total_revenue_saved']:,.2f}")
    c3.metric("Net Financial Benefit", f"${roi_summary['net_financial_benefit']:,.2f}")
    c4.metric("Overall Program ROI", f"{roi_summary['overall_roi_pct']:.1f}%")

# ==========================================
# MODULE 13: 💬 AI CHURN ASSISTANT
# ==========================================
elif nav_mode == "💬 AI Churn Assistant":
    st.markdown("### 💬 Local AI Churn Assistant (Natural Language Analytics)")
    
    st.markdown("Ask natural language questions about customer count, high-risk churners, revenue at risk, top priority accounts, or retention strategies.")
    
    preset = st.selectbox("Quick Query Presets:", [
        "Select a preset or type your question below...",
        "How many total customers and high-risk churners are in the dataset?",
        "What is the total revenue at risk?",
        "Who are the top critical priority customers to contact?",
        "What are the main drivers of customer churn?",
        "What retention offer strategy is recommended?"
    ])
    
    user_q = st.text_input("💬 Enter your question:", value=preset if preset != "Select a preset or type your question below..." else "")
    
    if user_q:
        st.markdown(f'<div class="chat-bubble-user">{user_q}</div>', unsafe_allow_html=True)
        response = query_churn_intelligence(user_q, df_results)
        st.markdown(f'<div class="chat-bubble-ai">{response}</div>', unsafe_allow_html=True)

# ==========================================
# MODULE 14: 📄 BUSINESS REPORT & EXPORT
# ==========================================
elif nav_mode == "📄 Business Report & Export":
    st.markdown("### 📄 Business Executive Report & Data Export")
    
    st.markdown("Generate and download comprehensive PDF executive reports, CSV predictions, or Excel files.")
    
    summary_metrics = {
        "total_customers": len(df_results),
        "high_risk_count": rev_metrics["count_churners_at_risk"],
        "churn_rate_pct": (rev_metrics["count_churners_at_risk"] / len(df_results) * 100.0) if len(df_results) > 0 else 0.0,
        "monthly_revenue_at_risk": rev_metrics["monthly_revenue_at_risk"],
        "annual_revenue_at_risk": rev_metrics["annual_revenue_at_risk"],
        "high_value_annual_at_risk": rev_metrics["high_value_annual_at_risk"],
        "overall_roi_pct": 380.0
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pdf_bytes = generate_pdf_report(df_results, summary_metrics)
        st.download_button(
            label="📄 Download Executive PDF Report",
            data=pdf_bytes,
            file_name="Customer_Churn_Executive_Report.pdf",
            mime="application/pdf"
        )
        
    with col2:
        csv_bytes = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Full CSV Results",
            data=csv_bytes,
            file_name="churn_intelligence_results.csv",
            mime="text/csv"
        )

    with col3:
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
            df_results.to_excel(writer, index=False, sheet_name="Churn_Results")
        st.download_button(
            label="📈 Download Excel Spreadsheet",
            data=buffer_excel.getvalue(),
            file_name="churn_intelligence_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
