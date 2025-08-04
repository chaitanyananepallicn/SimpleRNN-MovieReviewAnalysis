import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import streamlit as st
import os

# --- Build Correct File Paths for Deployment ---
# Get the absolute path to the directory where this script is located
# This makes the app work reliably when deployed.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join the script directory with the filenames to create absolute paths
model_path = os.path.join(script_dir, 'model.keras')
gender_encoder_path = os.path.join(script_dir, 'gender_labelencoder.pkl')
geo_encoder_path = os.path.join(script_dir, 'geo_onehotencoder.pkl')
scaler_path = os.path.join(script_dir, 'standardscaler.pkl')

# --- Load Models and Preprocessors ---
# Use a try-except block for robust error handling
try:
    model = load_model(model_path)
    with open(gender_encoder_path, 'rb') as file:
        gender_labelencoder = pickle.load(file)
    with open(geo_encoder_path, 'rb') as file:
        geo_onehotencoder = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        standardscaler = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Error loading a required file: {e}. Please ensure all model and preprocessor files are in the same directory as app.py.")
    st.stop() # Stop the app if files can't be loaded
except Exception as e:
    st.error(f"An unexpected error occurred while loading files: {e}")
    st.stop()


# --- Streamlit App UI ---
st.title('Customer Estimated Salary Prediction')

# User input section
st.header("Enter Customer Details")
geography = st.selectbox('Geography', geo_onehotencoder.categories_[0])
gender = st.selectbox('Gender', gender_labelencoder.classes_)
age = st.slider('Age', 18, 92, 35)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600)
tenure = st.slider('Tenure (years)', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.selectbox('Is Active Member?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
# 'Exited' is now an input feature for predicting salary
exited = st.selectbox('Has the Customer Churned (Exited)?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')


# --- Prediction Logic ---
if st.button('Predict Estimated Salary'):
    # 1. Prepare the input data into a DataFrame
    # Note: 'EstimatedSalary' is removed as it is now the target
    input_data_dict = {
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    }
    input_df = pd.DataFrame(input_data_dict)

    # 2. Encode categorical features
    input_df['Gender'] = gender_labelencoder.transform(input_df['Gender'])
    geo_encoded = geo_onehotencoder.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=geo_onehotencoder.get_feature_names_out(['Geography']))

    # 3. Combine all features into a single DataFrame
    final_df = pd.concat([input_df.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

    # 4. Ensure the column order matches the training data for the new regression model
    # This is a critical step. The order must be exactly the same as when the new model was trained.
    # 'EstimatedSalary' is removed and 'Exited' is added.
    training_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'Exited', 'Geography_France',
        'Geography_Germany', 'Geography_Spain'
    ]
    
    # Reorder the dataframe to match the training column order
    # It's important to handle potential missing columns if the scaler expects them
    for col in training_columns:
        if col not in final_df.columns:
            final_df[col] = 0 # Add missing columns if any, and set to 0
    final_df = final_df[training_columns]


    # 5. Scale the final data
    # The scaler must be the one trained with the regression data.
    input_scaled = standardscaler.transform(final_df)

    # 6. Make prediction
    predicted_salary = model.predict(input_scaled)[0][0]

    # 7. Display the result
    st.header("Prediction Result")
    st.success(f'The Predicted Estimated Salary is: ${predicted_salary:,.2f}')

