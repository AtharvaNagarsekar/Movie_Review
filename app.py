import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🍿", layout="centered")
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("imdb_sentiment_model.keras")
model = load_model()
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)
tokenizer = load_tokenizer()
def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return " ".join(words)
def predict_sentiment(user_input):
    processed_text = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding="post")
    prediction = model.predict(padded_sequence)
    sentiment = "🔥 Positive Vibes Only!" if prediction >= 0.5 else "💀 Major Red Flag!"
    return sentiment, prediction[0][0]
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;700&display=swap');
    
    html, body {
        background-color: #0d1117;
        color: #fff;
        font-family: 'Lexend', sans-serif;
    }
    .main-container {
        background: linear-gradient(135deg, #ff8a00, #e52e71);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(255, 138, 0, 0.5);
        text-align: center;
    }
    .title {
        font-size: 40px;
        font-weight: 700;
        background: -webkit-linear-gradient(#ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        background: #161b22;
        color: #fff;
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        box-shadow: 0px 4px 10px rgba(255, 138, 0, 0.4);
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<p class='title'>🎬 Movie Review Analyzer</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Drop your review and let’s see if it’s 🔥 or 💀!</p>", unsafe_allow_html=True)
user_input = st.text_area("🎥 Type your movie review here:", "", height=120)
if st.button("🎬 Analyze Now!"):
    if user_input.strip():
        with st.spinner("Rolling the tape... 🎞️"):
            time.sleep(1.5)  # Simulating processing time
            sentiment, confidence = predict_sentiment(user_input)

        st.markdown(f"## {sentiment}")
        st.write(f"Confidence Score: **{confidence:.4f}**")
    else:
        st.warning("⚠️ No review? No rating! Type something first.")
st.markdown("</div>", unsafe_allow_html=True)
