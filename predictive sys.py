# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Load the saved model and scaler
loaded_model = pickle.load(open('C:/Users/RAM KUMAR/Desktop/WORK/ML-DL/ML PROJECTS/1-Dia/trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('C:/Users/RAM KUMAR/Desktop/WORK/ML-DL/ML PROJECTS/1-Dia/scaler.sav', 'rb'))

# Example input (same as used in training feature order)
input_data = (1, 103, 30, 38, 83, 43.3, 0.183, 33)

# Convert input data to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data, dtype=float)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = loaded_scaler.transform(input_data_reshaped)

# Predict
prediction = loaded_model.predict(std_data)
print(prediction)

# Output result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
