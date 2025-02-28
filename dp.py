import pandas as pd

# Load Disease-Symptom Dataset
df = pd.read_csv("fag.csv")

# Display first 5 rows
print(df.head())

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define the features (symptoms) and target (disease)
X = df.drop(columns=['diseases'])
y = df['diseases']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Get all feature names used in training
feature_names = model.feature_names_in_

# Display all feature names
print("Total Features:", len(feature_names))
print(feature_names)

import joblib

# Save the model to a file
joblib.dump(model, 'disease_prediction_model.joblib')

print("Model saved successfully.")

joblib.dump(feature_names, 'feature_names.joblib')

import pandas as pd
import numpy as np

# Create an empty DataFrame with all required columns
df_test = pd.DataFrame(columns=feature_names)

# Fill missing values with 0 (or use NaN if unsure)
df_test.loc[0] = np.zeros(len(feature_names))

# Display first few columns to verify
print(df_test.head())

# Manually set symptoms that are present
# Example: Set 'anxiety and nervousness' and 'depression' to 1 (present)
df_test.at[0, 'anxiety and nervousness'] = 1
df_test.at[0, 'depression'] = 1

# Display the updated DataFrame
print(df_test.head())
df_test = df_test[feature_names]  # Reorder columns
predictions = model.predict(df_test)
print(predictions)
import joblib
import pandas as pd
import numpy as np
#lets create a function to predict the disease using saved model
def predict_disease(symptoms):
    # Load the model
    model = joblib.load('disease_prediction_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    
    # Create an empty DataFrame with all required columns
    df_test = pd.DataFrame(columns=feature_names)
    df_test.loc[0] = np.zeros(len(feature_names))
    
    # Set the symptoms to 1 (present)
    for symptom in symptoms:
        if symptom in feature_names:
            df_test.at[0, symptom] = 1
    
    # Make a prediction
    #use the saved model disease_prediction_model.joblib to make a prediction
    #model = joblib.load('disease_prediction_model.joblib')
    predictions = model.predict(df_test)
    return predictions[0]
#take symptoms from user
symp = input("Enter the symptoms separated by comma: ")
symptoms = symp.split(", ")
# Predict the disease
disease = predict_disease(symptoms)
print("Predicted Disease:", disease)



import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Load the model and feature names once
@st.cache_resource
def load_model():
    model = joblib.load('disease_prediction_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    
    return model, feature_names

model, feature_names = load_model()

# Function to get precautions from Llama 3 model via Groq API
def get_precautions(symptoms, disease):
    if not client:
        return "❌ API key is missing. Please check your .env file."

    # Define the prompt
    prompt = f"""
    A patient has reported the following symptoms: {', '.join(symptoms)}.
    The predicted disease is: {disease}.
    Based on this information, provide some precautionary measures before visiting a doctor.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a medical assistant providing health precautions."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
            top_p=1,
            stream=True
        )

        # Stream response
        precautions_text = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            precautions_text += content
            yield content  # Stream response in real-time

    except Exception as e:
        yield f"❌ Error fetching precautions: {str(e)}"

# Streamlit UI
st.title("Disease Prediction & Precautions")
st.write("Enter the symptoms separated by commas (e.g., fever, cough, headache)")

# Get user input
symptom_input = st.text_input("Symptoms")

if symptom_input:
    symptoms = [s.strip().lower() for s in symptom_input.split(",") if s.strip()]
    
    # Create a DataFrame with all feature names, initializing with zeros
    df_test = pd.DataFrame(columns=feature_names)
    df_test.loc[0] = np.zeros(len(feature_names))

    # Set symptom values to 1 if present in feature names
    for symptom in symptoms:
        if symptom in feature_names:
            df_test.at[0, symptom] = 1

    # Make a prediction
    try:
        prediction = model.predict(df_test)[0]
        st.success(f"**Predicted Disease:** {prediction}")

        # Display Precautions with Streaming
        st.subheader("Precautionary Measures Before Visiting a Doctor:")
        precautions_container = st.empty()  # Placeholder for streaming response

        precautions_text = ""
        for chunk in get_precautions(symptoms, prediction):
            precautions_text += chunk
            precautions_container.write(precautions_text)  # Update in real-time

        # Disclaimer
        st.write("⚠️ **Disclaimer:** This is AI-generated information. Consult a doctor for medical advice.")

    except Exception as e:
        st.error(f"❌ An error occurred while predicting: {e}")
else:
    st.warning("⚠️ Please enter at least one symptom.")
