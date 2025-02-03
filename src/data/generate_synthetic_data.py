import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_samples=10000):
    """
    Generates synthetic data for bank customers.
    """
    np.random.seed(42)
    
    # Demographics
    customer_ids = np.arange(1, num_samples + 1)
    ages = np.random.randint(18, 90, size=num_samples)
    genders = np.random.choice(['Male', 'Female'], size=num_samples)
    geographies = np.random.choice(['France', 'Germany', 'Spain'], size=num_samples)
    
    # Bank details
    credit_scores = np.random.randint(300, 850, size=num_samples)
    tenures = np.random.randint(0, 11, size=num_samples)
    balances = np.random.uniform(0, 250000, size=num_samples)
    num_of_products = np.random.randint(1, 5, size=num_samples)
    has_credit_card = np.random.randint(0, 2, size=num_samples)
    is_active_member = np.random.randint(0, 2, size=num_samples)
    estimated_salaries = np.random.uniform(20000, 150000, size=num_samples)
    
    # Churn (Exited)
    # Let's assume customers with low credit score and high balance are more likely to churn
    churn_prob = (
        (credit_scores < 400).astype(int) +
        (balances > 150000).astype(int) +
        (is_active_member == 0).astype(int)
    )
    churn = np.where(churn_prob > 1, 1, 0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'CreditScore': credit_scores,
        'Geography': geographies,
        'Gender': genders,
        'Age': ages,
        'Tenure': tenures,
        'Balance': balances,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salaries,
        'Exited': churn
    })
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    data.to_csv('data/raw/synthetic_data.csv', index=False)
    
    return data
