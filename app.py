import streamlit as st
import numpy as np
import joblib

# Load the logistic regression model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prediction function
def predict_heart_failure_stage(features):
    # Scale the features
    features_scaled = scaler.transform([features])
    # Make a prediction
    prediction = model.predict(features_scaled)
    # Map prediction to stage names
    stage_map = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
    return stage_map[prediction[0]]

# Streamlit app title
st.title('Heart Failure Patient Classification App')

# Instructions
st.write('Use the sliders below to input patient data and predict the heart failure stage.')

# Sliders for patient input data based on ranges (in the correct order)
bnp = st.slider('BNP (pg/mL)', min_value=80, max_value=300, value=150, step=1)
nt_probnp = st.slider('NT-proBNP (pg/mL)', min_value=300, max_value=5000, value=600, step=100)
heart_rate = st.slider('Heart Rate (bpm)', min_value=60, max_value=120, value=80, step=1)
sbp = st.slider('Systolic Blood Pressure (mmHg)', min_value=80, max_value=180, value=120, step=1)
dbp = st.slider('Diastolic Blood Pressure (mmHg)', min_value=50, max_value=100, value=80, step=1)
spo2 = st.slider('SpO2 (%)', min_value=85, max_value=100, value=95, step=1)

# When the 'Predict' button is clicked, make a prediction
if st.button('Predict'):
    # Prepare the feature list in the correct order
    features = [bnp, nt_probnp, heart_rate, sbp, dbp, spo2]
    # Predict the heart failure stage
    result = predict_heart_failure_stage(features)
    # Display the result
    st.success(f'The predicted heart failure stage is: {result}')
