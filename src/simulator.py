import os
import sys
import pandas as pd

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine import predict_churn

def simulate_customer_what_if(original_row, modified_attributes):
    """
    Executes real-time What-If scenario simulation by modifying customer attributes
    and re-running the ML prediction pipeline.
    Returns comparison dict (original vs simulated probability and risk tier).
    """
    df_single = pd.DataFrame([original_row])
    
    # Apply modifications
    for attr, val in modified_attributes.items():
        if attr in df_single.columns:
            df_single[attr] = val

    # Run inference
    df_simulated = predict_churn(df_single)
    sim_row = df_simulated.iloc[0].to_dict()

    orig_prob = float(original_row.get("Churn_Probability", 0.5))
    sim_prob = float(sim_row.get("Churn_Probability", 0.5))

    delta_prob = sim_prob - orig_prob
    delta_pct = (delta_prob / orig_prob * 100.0) if orig_prob > 0 else 0.0

    return {
        "original_probability": round(orig_prob, 4),
        "simulated_probability": round(sim_prob, 4),
        "delta_probability": round(delta_prob, 4),
        "delta_percentage": round(delta_pct, 1),
        "original_risk_level": original_row.get("Risk_Level", "Unknown"),
        "simulated_risk_level": sim_row.get("Risk_Level", "Unknown"),
        "simulated_record": sim_row
    }
