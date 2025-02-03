import streamlit as st
import pandas as pd
import os
from data.generate_synthetic_data import generate_synthetic_data
from features.preprocess_data import preprocess_data
from models.train_models import train_models
from models.predict import load_best_model, make_prediction
from visualization.visualize import plot_correlation_matrix

def main():
    st.title("Customer Churn Prediction")

    # Generate Synthetic Data
    if st.checkbox("Generate Synthetic Data"):
        data = generate_synthetic_data()
        st.write("Synthetic data generated successfully!")
        st.dataframe(data.head())
    else:
        if os.path.exists('data/raw/synthetic_data.csv'):
            data = pd.read_csv('data/raw/synthetic_data.csv')
        else:
            st.warning("Please generate synthetic data first.")
            return

    # Data Visualization
    st.header("Data Insights")
    if st.checkbox("Show Correlation Matrix"):
        plot_correlation_matrix(data)
        st.image('data/processed/correlation_matrix.png')

    # Data Preprocessing
    X, y = preprocess_data(data)

    # Model Training
    if st.button("Train Models"):
        best_model_name, best_accuracy = train_models(X, y)
        st.success(f"Best Model: {best_model_name} with accuracy {best_accuracy:.2f}")
    else:
        if os.path.exists('models/best_model.pkl') or os.path.exists('models/best_model.h5') or os.path.exists('models/best_model.pth'):
            st.info("Model already trained.")
        else:
            st.warning("Please train the models.")

    # Load Best Model
    if st.checkbox("Load Best Model"):
        if 'best_model_name' not in locals():
            # Assume RandomForest as default
            best_model_name = 'RandomForest'
        model = load_best_model(best_model_name)
        st.write(f"Loaded {best_model_name} model.")

        # User Input for Prediction
        st.header("Predict Churn for a New Customer")
        input_data = {}
        for feature in X.columns:
            if feature in ['Gender', 'Geography']:
                input_data[feature] = st.selectbox(feature, sorted(data[feature].unique()))
            else:
                if pd.api.types.is_integer_dtype(data[feature]):
                    input_data[feature] = st.number_input(feature, int(data[feature].min()), int(data[feature].max()), int(data[feature].mean()))
                else:
                    input_data[feature] = st.number_input(feature, float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
        # for feature in X.columns:
        #     if feature in ['Gender', 'Geography']:
        #         input_data[feature] = st.selectbox(feature, sorted(data[feature].unique()))
        #     else:
        #         input_data[feature] = st.number_input(feature, float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
        input_df = pd.DataFrame([input_data])

        # Preprocess Input Data
        input_df_processed, _ = preprocess_data(input_df)
        
        if st.button("Predict Churn"):
            prediction = make_prediction(model, best_model_name, input_df_processed)
            if prediction[0] == 1 or prediction[0] == 1.0:
                st.warning("This customer is likely to churn.")
            else:
                st.success("This customer is likely to stay.")

if __name__ == '__main__':
    main()
