# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:15:20 2025
@author: RAM KUMAR
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler
loaded_model = pickle.load(open('C:/Users/RAM KUMAR/Desktop/WORK/ML-DL/ML PROJECTS/1-Dia/trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('C:/Users/RAM KUMAR/Desktop/WORK/ML-DL/ML PROJECTS/1-Dia/scaler.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = loaded_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_data)
    return '🟢 The person is **not diabetic**' if prediction[0] == 0 else '🔴 The person is **diabetic**'

# Main app
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="centered")
    
    # Header Section
    st.markdown("<h1 style='text-align: center; color: teal;'>🧬 Diabetes Prediction Web App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter your medical details below and click the button to check your diabetes risk.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Input form layout using columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input('🤰 Number of Pregnancies')
        BloodPressure = st.text_input('🩺 Blood Pressure (mm Hg)')
        Insulin = st.text_input('💉 Insulin Level')
        DiabetesPedigreeFunction = st.text_input('👨‍👩‍👧 Diabetes Pedigree Function')

    with col2:
        Glucose = st.text_input('🍬 Glucose Level')
        SkinThickness = st.text_input('🧪 Skin Thickness (mm)')
        BMI = st.text_input('📏 BMI (Body Mass Index)')
        Age = st.text_input('🎂 Age')

    # Prediction result
    diagnosis = ""

    if st.button('🚀 Get Diabetes Test Result'):
        try:
            input_list = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            input_list = list(map(float, input_list))
            diagnosis = diabetes_prediction(input_list)
        except ValueError:
            diagnosis = "⚠️ Please enter valid numeric values in all fields."

    if diagnosis:
        st.markdown("---")
        st.success(diagnosis)

    # Footer
    st.markdown("""
    <br><hr>
    <p style='text-align: center; font-size: 12px;'>Made with ❤️ by RAM KUMAR | Machine Learning Project</p>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
