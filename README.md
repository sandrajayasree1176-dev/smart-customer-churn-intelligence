# Smart Customer Churn Intelligence & AI-Powered Retention Optimization System

An enterprise-grade AI intelligence platform that predicts customer churn, explains the underlying reasons behind churn (XAI), estimates Customer Lifetime Value (CLV) & Revenue at Risk, optimizes personalized retention strategies, simulates What-If scenarios in real time, clusters customer micro-segments, and generates executive business reports.

---

## 🌟 Key Features & Capabilities

1. **Flexible Data Management**: Auto-detects target churn columns (`Churn`, `churn`, `Exited`, `Customer_Status`, `is_churn`). Supports CSV, Excel, and built-in synthetic dataset generation (5,000 records).
2. **Multi-Model ML Engine**: Evaluates 4 algorithms (**Random Forest**, **Gradient Boosting**, **Logistic Regression**, **Decision Tree**) with stratified splits, selecting the best model based on F1 / ROC-AUC.
3. **Explainable AI (XAI)**:
   - **Global XAI**: Feature importance ranking across all accounts.
   - **Local Customer XAI**: Individual waterfall contribution chart revealing positive & negative drivers pushing churn risk up or down.
4. **Financial Value & Risk Engine**:
   - **CLV Estimation**: Projected lifetime value across expected lifecycle.
   - **Revenue at Risk**: Monthly & Annual recurring revenue lost to churners.
   - **Priority Scoring (0-100)**: Ranks customers by risk probability and financial value.
5. **Personalized Retention & Offer Optimizer**:
   - Profile-driven action recommendations.
   - Economic ROI calculator evaluating discount offers, free tech support, and annual upgrades against probability reduction and net profit.
6. **Real-Time What-If Churn Simulator**: Interactive sliders allow tweaking customer attributes to observe live before-and-after probability shifts.
7. **Customer Segmentation (K-Means)**: Unsupervised clustering with silhouette score optimization and dynamic segment naming ("Loyal Champions", "Critical High-Value At-Risk").
8. **Local AI Churn Assistant**: Natural language Q&A engine backed by pandas analytics for instant data summaries.
9. **Executive PDF Business Report**: ReportLab PDF compiler producing downloadable reports with KPI tables, priority lists, and strategic action plans.

---

## 📁 System Architecture & Directory Structure

```
smart customer churn/
├── data/
│   └── customer_data.csv        # Synthetic telco customer dataset
├── outputs/
│   └── model/
│       └── churn_model.pkl      # Trained model bundle with preprocessor & metrics
├── src/
│   ├── app.py                   # Main Streamlit Dashboard with custom Glassmorphism UI
│   ├── data_loader.py           # Ingestion, validation & target auto-detection
│   ├── preprocessing.py         # Imputation, string parsing & standard scaling
│   ├── ml_engine.py             # Stratified ML training & model comparison
│   ├── explainability.py        # Global & Local SHAP/Waterfall XAI
│   ├── financial_engine.py      # CLV, Revenue at Risk, Priority Scoring & ROI
│   ├── retention_engine.py      # Rule-based personalized action generator
│   ├── segmentation.py          # K-Means clustering & silhouette optimization
│   ├── simulator.py             # What-If dynamic re-inference
│   ├── ai_assistant.py          # Natural language Q&A engine
│   ├── report_generator.py      # ReportLab PDF compiler
│   ├── generate_data.py         # Synthetic telco data generator
│   └── churn_model.py           # Legacy compatibility module
├── requirements.txt             # Python dependencies
└── README.md                    # System documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train ML Engine
```bash
python src/generate_data.py
python src/ml_engine.py
```

### 3. Launch the Dashboard
```bash
streamlit run src/app.py
```

---

## 📊 Streamlit Modules (14 Navigation Views)
- **Executive View**: Hero metrics, annual revenue at risk, churn risk distribution, top priority accounts.
- **Data Management**: File upload, raw data inspection, target auto-detection metadata.
- **Churn Visual Analytics**: Interactive Plotly scatter, bar, pie, and histogram distribution charts.
- **ML Engine & Models**: Multi-model comparison matrix, confusion matrix, ROC-AUC metrics.
- **Explainable AI (XAI)**: Global feature importance & customer waterfall breakdown.
- **Customer Value & CLV**: Customer Lifetime Value leaderboards & value distribution.
- **Retention Strategy**: Personalized action generator & targeted intervention list.
- **Offer Optimizer**: Offer matrix with net financial benefit & expected ROI %.
- **What-If Simulator**: Live parameter sliders with real-time re-inference.
- **Early Warning System**: Automated high-risk alert trigger matrix.
- **Customer Segmentation**: K-Means clustering scatter plot & segment profiles.
- **Retention ROI Calculator**: Aggregate financial impact & conversion math.
- **AI Churn Assistant**: Natural language query chat interface.
- **Business Report & Export**: Downloadable PDF, CSV, and Excel exports.
