import os

import re

import pickle

import pandas as pd

import streamlit as st



# -------------------------

# UI Configuration

# -------------------------



def load_css(file_name="style.css"):

    """Loads custom CSS to style the Streamlit app seamlessly."""

    if os.path.exists(file_name):

        with open(file_name) as f:

            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    else:

        st.warning(f"⚠️ '{file_name}' not found. Standard Streamlit styling applied.")





# -------------------------

# Load Model Safely

# -------------------------



@st.cache_resource

def load_model():

    """

    Loads the trained model and vectorizer.

    Using @st.cache_resource prevents reloading the file on every user interaction.

    """

    if not os.path.exists("model.pkl"):

        return None, None

        

    with open("model.pkl", "rb") as f:

        model_data = pickle.load(f)

        

    return model_data["vectorizer"], model_data["model"]





# -------------------------

# Clean Text

# -------------------------

def clean_text(text):
    # 1. Force the input to be a string to prevent Pandas NaN errors
    text = str(text)
    
    # 2. Replace links with a standardized URL token
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
    
    # 3. Remove special characters, keeping letters, numbers, spaces, and key symbols
    text = re.sub(r"[^a-zA-Z0-9\s!$%]", "", text)
    
    # 4. Convert to lowercase and strip leading/trailing whitespace
    return text.lower().strip()




# -------------------------

# Single Prediction

# -------------------------


def predict_message(message, vectorizer, model):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max() * 100
    
    
    if int(prediction) == 1: 
        return "Spam", confidence
    else:
        return "Ham", confidence





# -------------------------

# Batch Prediction

# -------------------------

def batch_predict(df, text_column, vectorizer, model):
    # 1. Prepare and vectorize text
    texts = df[text_column].fillna("").astype(str)
    X_vectorized = vectorizer.transform(texts)
    
    # 2. Generate predictions
    predictions = model.predict(X_vectorized)
    
    # 3. Safely map predictions to 'Spam' or 'Ham'
    formatted_preds = []
    for p in predictions:
        # If model outputs strings directly ('spam' / 'ham')
        if str(p).lower() in ["spam", "ham", "1", "0"]:
            if str(p).lower() in ["spam", "1"]:
                formatted_preds.append("Spam")
            else:
                formatted_preds.append("Ham")
        else:
            formatted_preds.append(str(p).title())
            
    df["Prediction"] = formatted_preds
    return df