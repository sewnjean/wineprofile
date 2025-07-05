import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🍷 Red Wine Quality Classifier")
st.write("Enter the wine's chemical attributes to determine its quality.")

# Define features
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']

# Input fields
input_values = []
for feature in features:
    val = st.number_input(f"{feature.title()}", step=0.01, format="%.2f")
    input_values.append(val)

# Predict button
if st.button("Predict Quality"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]
    
    if prediction == 1:
        st.success(f"✅ This is a GOOD quality wine! (Confidence: {confidence:.2f})")
    else:
        st.error(f"❌ This is NOT good quality wine. (Confidence: {confidence:.2f})")