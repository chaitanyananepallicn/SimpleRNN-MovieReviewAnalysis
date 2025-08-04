import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences

# --- Caching Function to Load Model and Word Index ---
# @st.cache_resource tells Streamlit to run this function only once.
# The returned model and word_index are stored in a cache.
# On subsequent reruns, Streamlit will skip this function and use the cached objects.
@st.cache_resource
def load_model_and_dependencies():
    """
    Loads the pre-trained Keras model and the IMDB word index.
    This function is cached to prevent reloading on every script rerun.
    """
    # --- File Path Setup ---
    try:
        # This works when running as a script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model.keras')
    except NameError:
        # This is a fallback for environments where __file__ is not defined.
        model_path = 'model.keras'

    # --- Load Model and Word Index ---
    try:
        model = load_model(model_path)
        word_index = imdb.get_word_index()
        return model, word_index
    except Exception as e:
        st.error(f"Error loading model or preprocessor files. Please ensure 'model.keras' is in the same directory.")
        st.error(f"Details: {e}")
        st.stop()

# --- Load the model and word_index from the cached function ---
model, word_index = load_model_and_dependencies()


# --- Preprocessing Function (Corrected) ---
# This function converts raw text into a format the model can understand.
def preprocessing(text):
    """
    Converts a text review into a padded sequence of integers.
    """
    words = text.lower().split()
    encoded_rev = []
    oov_index = 2 # Keras IMDB dataset reserves index 2 for OOV words.
    for word in words:
        # The +3 offset is for reserved indices: <PAD>, <START>, and <OOV>.
        index = word_index.get(word)
        if index is not None:
            encoded_rev.append(index + 3)
        else:
            encoded_rev.append(oov_index)
    
    padded_rev = pad_sequences([encoded_rev], maxlen=500)
    return padded_rev


# --- Prediction and Analysis Functions ---
def prediction(text):
    """
    Uses the model to predict the sentiment probability of the preprocessed text.
    """
    preprocessed_text = preprocessing(text)
    pred = model.predict(preprocessed_text)
    return pred[0][0]

def analyze(text):
    """
    Analyzes the text, displays the sentiment, and shows the prediction score.
    """
    prob = prediction(text)
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    
    if sentiment == "Positive":
        st.success(f"**Sentiment: {sentiment} Review**")
    else:
        st.error(f"**Sentiment: {sentiment} Review**")
        
    st.write(f"**Prediction Score:** `{prob:.4f}`")
    st.info("The prediction score represents the model's confidence. Scores closer to 1.0 indicate a strong positive sentiment, while scores closer to 0.0 indicate a strong negative sentiment.", icon="ℹ️")


# --- Streamlit User Interface ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below, and the model will classify it as either positive or negative.")

user_input = st.text_area("Movie Review", height=150, placeholder="e.g., 'This movie was fantastic! The acting was superb and the plot was thrilling.'")

if st.button('Classify Review'):
    if user_input.strip():
        with st.spinner('Analyzing...'):
            analyze(user_input)
    else:
        st.warning("Please enter a review before classifying.", icon="⚠️")

