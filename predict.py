# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:45:40 2025

@author: aruna
"""

import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text to classify
sample_text = ["Congratulations! You've won a free ticket to Bahamas!"]

# Vectorize and predict
sample_vec = vectorizer.transform(sample_text)
prediction = model.predict(sample_vec)

print(f"Prediction: {prediction[0]}")
