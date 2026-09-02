import streamlit as st
import pickle
import re
import string
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


# ============================================================
# NLTK RESOURCES
# ============================================================

@st.cache_resource
def download_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


download_nltk_resources()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gru_model.keras")


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_label_mapping():
    with open("label_mapping.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_config():
    with open("config.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
tokenizer = load_tokenizer()
label_mapping = load_label_mapping()
config = load_config()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LENGTH = config["max_length"]
MAX_FEATURES = config["max_features"]


# ============================================================
# STOPWORDS
# ============================================================

STOP_WORDS = set(stopwords.words("english"))


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def remove_special(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


def remove_stopwords(text):
    return [
        word
        for word in text
        if word not in STOP_WORDS
    ]


def remove_hyperlink(word):
    return re.sub(r"http\S+", "", word)


def preprocess_text(text):

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = remove_special(text)

    # 3. Tokenize
    text = word_tokenize(text)

    # 4. Remove English stopwords
    text = remove_stopwords(text)

    # 5. Convert tokens back to text
    text = " ".join(text)

    # 6. Remove hyperlinks
    text = remove_hyperlink(text)

    return text


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_email(email):

    # Same preprocessing as training
    cleaned_text = preprocess_text(email)

    # Convert words to integer sequences
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Pad to exactly 500 tokens
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    # GRU prediction
    spam_probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # Same threshold used in notebook
    prediction = int(spam_probability > 0.5)

    return prediction, spam_probability, cleaned_text


# ============================================================
# HEADER
# ============================================================

st.title("📧 Email Spam Detector")

st.write(
    "Enter an email message below and the trained "
    "**GRU Neural Network** will classify it as "
    "**Spam** or **Ham**."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Model Information")

    st.write("**Model:** GRU Neural Network")
    st.write("**Task:** Binary Text Classification")
    st.write("**Vocabulary Size:** 5,000")
    st.write("**Maximum Sequence Length:** 500")
    st.write("**Embedding Dimension:** 32")
    st.write("**GRU Units:** 64")
    st.write("**Output:** Sigmoid")

    st.divider()

    st.subheader("🧠 NLP Pipeline")

    st.write(
        """
        1. Lowercase conversion
        2. Punctuation removal
        3. Word tokenization
        4. Stopword removal
        5. Hyperlink removal
        6. Sequence tokenization
        7. Padding
        8. GRU classification
        """
    )

    st.divider()

    st.subheader("📚 Topics Covered")

    st.write(
        """
        • Natural Language Processing  
        • Text Preprocessing  
        • Tokenization  
        • Stopword Removal  
        • Word Embeddings  
        • Sequence Padding  
        • GRU Neural Networks  
        • Binary Classification  
        • Model Evaluation  
        • Streamlit
        """
    )


# ============================================================
# EMAIL INPUT
# ============================================================

st.subheader("✉️ Enter Email")

email_text = st.text_area(
    "Paste the email content below:",
    height=220,
    placeholder=(
        "Example: Congratulations! You have won "
        "a prize. Click here to claim your reward..."
    )
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Check Email",
    use_container_width=True
):

    if not email_text.strip():

        st.warning(
            "⚠️ Please enter an email message first."
        )

    else:

        prediction, spam_probability, cleaned_text = (
            predict_email(email_text)
        )

        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            st.error("🚨 SPAM EMAIL")

            confidence = spam_probability * 100

        else:

            st.success("✅ HAM / SAFE EMAIL")

            confidence = (
                1 - spam_probability
            ) * 100


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )


        # ====================================================
        # PROBABILITY DETAILS
        # ====================================================

        with st.expander("📊 Prediction Details"):

            st.write(
                f"**Predicted Class:** "
                f"{label_mapping[prediction]}"
            )

            st.write(
                f"**Spam Probability:** "
                f"{spam_probability * 100:.2f}%"
            )

            st.write(
                f"**Ham Probability:** "
                f"{(1 - spam_probability) * 100:.2f}%"
            )


        # ====================================================
        # PREPROCESSED TEXT
        # ====================================================

        with st.expander("🔎 View Preprocessed Text"):

            st.write(cleaned_text)