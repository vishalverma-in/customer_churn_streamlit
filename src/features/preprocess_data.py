import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocesses the data for modeling.
    """
    df = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Geography'] = le.fit_transform(df['Geography'])
    
    # Feature scaling
    scaler = StandardScaler()
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Features and target
    columns_to_drop = ['CustomerID', 'Exited']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        X = df.drop(existing_columns_to_drop, axis=1)
    else:
        X = df.copy()
    # X = df.drop(['CustomerID', 'Exited'], axis=1)
    # y = df['Exited']
    
    return X, df['Exited'] if 'Exited' in df.columns else None
