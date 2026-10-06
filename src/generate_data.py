import os
import pandas as pd
import numpy as np

def generate_sample_dataset(output_path="data/customer_data.csv", num_samples=5000, random_seed=42):
    """
    Generates a realistic synthetic Telco Customer Churn dataset (default 5,000 records).
    Includes rich customer attributes: tenure, charges, contract type, services, and realistic churn logic.
    """
    np.random.seed(random_seed)
    
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_samples)]
    genders = np.random.choice(["Male", "Female"], size=num_samples, p=[0.5, 0.5])
    senior_citizens = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=num_samples, p=[0.30, 0.70])
    
    # Tenure in months (1 to 72)
    tenure = np.random.randint(1, 73, size=num_samples)
    
    phone_service = np.random.choice(["Yes", "No"], size=num_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=num_samples, p=[0.44, 0.44, 0.12])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for net in internet_service:
        if net == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.36, 0.64]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.44, 0.56]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    payment_methods = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=num_samples, p=[0.34, 0.22, 0.22, 0.22])
    
    # Monthly Charges calculation based on services
    base_charges = []
    for net, ts, stv in zip(internet_service, tech_support, streaming_tv):
        c = 20.0
        if net == "DSL": c += 30.0
        elif net == "Fiber optic": c += 55.0
        if ts == "Yes": c += 15.0
        if stv == "Yes": c += 15.0
        c += np.random.normal(0, 5)
        base_charges.append(max(18.5, round(c, 2)))
        
    monthly_charges = np.array(base_charges)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(-10, 10, num_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Realistic Churn Probability Model
    churn_prob = np.zeros(num_samples)
    
    # Contract effect
    churn_prob += np.where(contracts == "Month-to-month", 0.35, 0.05)
    
    # Internet type effect
    churn_prob += np.where(internet_service == "Fiber optic", 0.18, 0.05)
    
    # Tech Support effect
    churn_prob += np.where(np.array(tech_support) == "No", 0.15, -0.05)
    
    # Tenure effect (Newer customers churn more)
    churn_prob += np.where(tenure <= 12, 0.20, -0.15)
    
    # Payment Method effect
    churn_prob += np.where(payment_methods == "Electronic check", 0.12, -0.05)
    
    # High Charges effect
    churn_prob += np.where(monthly_charges > 75, 0.10, -0.05)
    
    # Clip probabilities between 0.03 and 0.92
    churn_prob = np.clip(churn_prob, 0.03, 0.92)
    
    churn_labels = np.random.binomial(1, churn_prob)
    churn_str = np.where(churn_labels == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_str
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset containing {num_samples} records successfully generated at: {output_path}")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
