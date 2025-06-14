import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"  # Change if deployed

st.title("Aircraft Damage Prediction")

columns = ['feature1', 'feature2', 'feature3']  # same order as training

input_data = {}
for col in columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    response = requests.post(API_URL, json={"data": input_data})

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {'Damage' if result['prediction'] else 'No Damage'}")
        st.write(f"Probability: {result['probability']:.2%}")

        st.subheader("SHAP Feature Impact")
        st.bar_chart(result['shap_values'])

        st.subheader("LIME Explanation")
        for feature, weight in result['lime_explanation']:
            st.write(f"{feature}: {weight:.4f}")
    else:
        st.error("Error contacting backend")
