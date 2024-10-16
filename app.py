import streamlit as st
import joblib
import pandas as pd

# Load the trained model
rf_model = joblib.load('random_forest_model.pkl')  
svm_model = joblib.load('svm_model.pkl')            

# Title of the app
st.title("Spam Email Detection")

# Input text area for user to enter email content
email_content = st.text_area("Enter the email content:")

if st.button("Predict"):
    
    prediction = rf_model.predict([email_content])
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    st.write(f"Prediction: {result}")
