import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PorterStemmer
ps = PorterStemmer()

# Preprocessing function (renamed to preprocess_text to match the call)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Function to predict spam
def predict_spam(text, tfidf, model):
    try:
        # Preprocess the text
        preprocessed_text = preprocess_text(text)  # Use the defined function
        # Vectorize the preprocessed text
        vectorized_text = tfidf.transform([preprocessed_text])
        # Predict
        prediction = model.predict(vectorized_text)[0]
        return 1 if prediction == "spam" else 0
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Load the model and vectorizer using joblib
try:
    tfidf = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Error: vectorizer.pkl or model.pkl not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading vectorizer or model: {e}")
    st.stop()

# Streamlit app interface
st.title("Email/SMS Spam Classifier")
st.write("Enter an SMS message below to classify it as Spam or Not Spam.")

# Input field for the SMS message
input_sms = st.text_input("Enter the message", placeholder="Type your message here...")

# Predict button
if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Make prediction
        prediction = predict_spam(input_sms, tfidf, model)

        # Display result
        if prediction is not None:
            st.subheader("Prediction:")
            if prediction == 1:
                st.error("Spam")
            else:
                st.success("Not Spam")

# Optional: Display the input message
if input_sms.strip():
    st.subheader("Input Message:")
    st.write(input_sms)