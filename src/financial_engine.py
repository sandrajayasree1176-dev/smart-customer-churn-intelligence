import pandas as pd
import numpy as np

def calculate_clv(df, expected_lifetime_months=24):
    """
    Calculates estimated Customer Lifetime Value (CLV).
    CLV = Monthly Charges * Expected Customer Lifetime (or Total Charges + projected future revenue).
    """
    df_calc = df.copy()
    
    # Check revenue source
    if "MonthlyCharges" in df_calc.columns:
        monthly_rev = df_calc["MonthlyCharges"].astype(float).fillna(50.0)
    elif "TotalCharges" in df_calc.columns and "tenure" in df_calc.columns:
        tenure_s = df_calc["tenure"].astype(float).replace(0, 1)
        monthly_rev = (df_calc["TotalCharges"].astype(float) / tenure_s).fillna(50.0)
    else:
        monthly_rev = pd.Series([50.0] * len(df_calc), index=df_calc.index)

    df_calc["Monthly_Revenue"] = monthly_rev
    df_calc["Estimated_CLV"] = (monthly_rev * expected_lifetime_months).round(2)
    return df_calc

def calculate_revenue_at_risk(df_results):
    """
    Calculates Monthly, Annual, and High-Value Revenue at Risk based on predicted churners.
    """
    if "MonthlyCharges" not in df_results.columns and "Monthly_Revenue" not in df_results.columns:
        df_calc = calculate_clv(df_results)
    else:
        df_calc = df_results.copy()
        if "Monthly_Revenue" not in df_calc.columns:
            df_calc["Monthly_Revenue"] = df_calc["MonthlyCharges"].astype(float)

    if "Predicted_Churn" in df_calc.columns:
        churn_mask = (df_calc["Predicted_Churn"] == "Yes") | (df_calc["Risk_Level"] == "High Risk")
    else:
        churn_mask = df_calc["Risk_Level"] == "High Risk"

    churn_df = df_calc[churn_mask]

    monthly_rev_at_risk = float(churn_df["Monthly_Revenue"].sum())
    annual_rev_at_risk = monthly_rev_at_risk * 12

    clv_threshold = df_calc["Monthly_Revenue"].quantile(0.75) if not df_calc.empty else 0
    high_val_churners = churn_df[churn_df["Monthly_Revenue"] >= clv_threshold]
    high_val_rev_at_risk = float(high_val_churners["Monthly_Revenue"].sum()) * 12

    return {
        "monthly_revenue_at_risk": round(monthly_rev_at_risk, 2),
        "annual_revenue_at_risk": round(annual_rev_at_risk, 2),
        "high_value_annual_at_risk": round(high_val_rev_at_risk, 2),
        "count_churners_at_risk": len(churn_df),
        "count_high_value_at_risk": len(high_val_churners)
    }

def calculate_priority_scores(df_results):
    """
    Computes Customer Retention Priority Score (0-100) based on Churn Probability and CLV.
    Priority Score = (Churn Probability * CLV / Max_CLV) * 100.
    Classifies Priority Tiers: Critical (90-100), High (70-89), Medium (40-69), Low (0-39).
    """
    df_calc = calculate_clv(df_results) if "Estimated_CLV" not in df_results.columns else df_results.copy()
    
    prob = df_calc["Churn_Probability"].astype(float) if "Churn_Probability" in df_calc.columns else pd.Series([0.5]*len(df_calc))
    clv = df_calc["Estimated_CLV"].astype(float)

    max_clv = clv.max() if clv.max() > 0 else 1.0
    normalized_clv = clv / max_clv

    raw_priority = (prob * 0.5 + normalized_clv * 0.5) * 100.0
    priority_score = np.clip(raw_priority, 0.0, 100.0).round(1)

    df_calc["Priority_Score"] = priority_score

    def classify_priority(score):
        if score >= 80.0:
            return "Critical Priority"
        elif score >= 60.0:
            return "High Priority"
        elif score >= 35.0:
            return "Medium Priority"
        else:
            return "Low Priority"

    df_calc["Priority_Tier"] = df_calc["Priority_Score"].apply(classify_priority)
    return df_calc

def optimize_retention_offer(row):
    """
    Determines optimal retention intervention, estimated cost, probability uplift, saved revenue, and ROI.
    """
    prob = float(row.get("Churn_Probability", 0.5))
    clv = float(row.get("Estimated_CLV", 1200.0))
    monthly_rev = float(row.get("MonthlyCharges", row.get("Monthly_Revenue", 50.0)))
    contract = str(row.get("Contract", "")).lower()
    tech_support = str(row.get("TechSupport", "")).lower()

    annual_revenue = monthly_rev * 12

    if prob >= 0.70 and clv >= 1500:
        offer = "Personal Customer Success Call + 15% Annual Discount"
        discount_pct = 0.15
        retention_uplift = 0.45
        cost = annual_revenue * discount_pct + 150.0
    elif prob >= 0.65 and "month-to-month" in contract:
        offer = "Annual Contract Upgrade with 10% Monthly Discount"
        discount_pct = 0.10
        retention_uplift = 0.40
        cost = annual_revenue * discount_pct
    elif prob >= 0.50 and "no" in tech_support:
        offer = "3 Months Free Priority Tech Support & Onboarding"
        retention_uplift = 0.35
        cost = 60.0
    elif prob >= 0.40:
        offer = "5% Monthly Charges Discount + Loyalty Rewards"
        discount_pct = 0.05
        retention_uplift = 0.25
        cost = annual_revenue * discount_pct
    else:
        offer = "Standard Customer Satisfaction Check-in"
        retention_uplift = 0.10
        cost = 10.0

    new_prob = max(0.05, prob * (1.0 - retention_uplift))
    prob_reduction = prob - new_prob

    rev_saved = annual_revenue * prob_reduction
    net_benefit = rev_saved - cost
    roi = (net_benefit / cost * 100.0) if cost > 0 else 0.0

    return {
        "Recommended_Offer": offer,
        "Original_Probability_Pct": round(prob * 100, 1),
        "Simulated_Probability_Pct": round(new_prob * 100, 1),
        "Probability_Reduction_Pct": round(prob_reduction * 100, 1),
        "Estimated_Cost": round(cost, 2),
        "Estimated_Revenue_Saved": round(rev_saved, 2),
        "Net_Benefit": round(net_benefit, 2),
        "Expected_ROI_Pct": round(roi, 1)
    }

def calculate_aggregate_retention_roi(df_results, assumed_conversion_rate=0.75):
    """
    Computes aggregate financial ROI for deploying retention offers across all high/medium risk customers.
    """
    df_calc = calculate_priority_scores(df_results)
    offers_data = [optimize_retention_offer(row) for _, row in df_calc.iterrows()]
    df_offers = pd.DataFrame(offers_data)

    total_cost = df_offers["Estimated_Cost"].sum()
    total_rev_saved = df_offers["Estimated_Revenue_Saved"].sum() * assumed_conversion_rate
    net_benefit = total_rev_saved - total_cost
    overall_roi = (net_benefit / total_cost * 100.0) if total_cost > 0 else 0.0

    return {
        "total_intervention_cost": round(total_cost, 2),
        "total_revenue_saved": round(total_rev_saved, 2),
        "net_financial_benefit": round(net_benefit, 2),
        "overall_roi_pct": round(overall_roi, 1),
        "offers_df": pd.concat([df_calc, df_offers], axis=1)
    }
