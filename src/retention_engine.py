import pandas as pd

def generate_personalized_action(row):
    """
    Generates rule-based personalized retention actions tailored to individual customer profiles.
    """
    risk = str(row.get("Risk_Level", "Low Risk"))
    contract = str(row.get("Contract", "")).lower()
    tech_support = str(row.get("TechSupport", "")).lower()
    tenure = float(row.get("tenure", 12))
    monthly_charges = float(row.get("MonthlyCharges", 50))
    internet = str(row.get("InternetService", "")).lower()

    actions = []

    if risk == "High Risk":
        actions.append("🚨 Executive Outreach: Assign dedicated Customer Success Agent for 1-on-1 consultation.")
        if "month-to-month" in contract:
            actions.append("📜 Contract Incentive: Offer 15% discount on upgrading to 1-Year or 2-Year Contract.")
        if monthly_charges > 70:
            actions.append("💳 Plan Optimization: Recommend tier downgrade or custom bundled discount to lower bill.")
        if "no" in tech_support and "no internet" not in tech_support:
            actions.append("🛠️ Technical Support: Bundle 3 months free VIP Tech Support & priority ticketing.")
        if "fiber" in internet:
            actions.append("🚀 Network Assurance: Conduct internet speed & line quality audit.")
    elif risk == "Medium Risk":
        actions.append("⚠️ Proactive Check-in: Send targeted customer satisfaction survey with loyalty gift.")
        if tenure <= 12:
            actions.append("🎓 Onboarding Refresh: Offer free product feature walkthrough and onboarding assistance.")
        if "month-to-month" in contract:
            actions.append("🎁 Loyalty Offer: Provide 5% monthly bill credit for lock-in commitment.")
    else:
        actions.append("✅ Relationship Nurture: Include in standard quarterly newsletter and product update list.")
        if tenure >= 36:
            actions.append("⭐ VIP Recognition: Enroll in Loyalty Advocate program with exclusive perks.")

    return " | ".join(actions)

def apply_retention_engine(df_results):
    """
    Applies personalized retention action generator to all rows in DataFrame.
    """
    df_calc = df_results.copy()
    df_calc["Personalized_Retention_Action"] = df_calc.apply(generate_personalized_action, axis=1)
    return df_calc
