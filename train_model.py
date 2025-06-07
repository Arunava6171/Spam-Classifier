# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:52:19 2025

@author: aruna
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load CSV file
df = pd.read_csv("spam (2).csv", encoding='latin-1')

# Step 2: Clean and prepare data
df = df[['v1', 'v2']]  # Keep only relevant columns
df = df.rename(columns={'v1': 'label', 'v2': 'text'})  # Rename for clarity

# Step 3: Split data into features and labels
X = df['text']
y = df['label']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Optional: Evaluate the model
X_test_vec = vectorizer.transform(X_test)
accuracy = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved as model.pkl and vectorizer.pkl")
