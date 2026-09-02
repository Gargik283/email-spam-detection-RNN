# 📧 Email Spam Detection using GRU

A deep learning-based **Email Spam Detection** project that classifies email messages as **Spam** or **Ham (Safe)** using Natural Language Processing (NLP) and a **GRU (Gated Recurrent Unit)** neural network.

## 🚀 Project Overview

This project uses NLP techniques to preprocess email text and a GRU-based deep learning model to identify potentially unwanted or spam emails.

The trained model is deployed through a **Streamlit web application**, allowing users to enter an email message and receive a real-time prediction.

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* GRU (Gated Recurrent Unit)
* Natural Language Processing (NLP)
* NLTK
* NumPy
* Pandas
* Scikit-learn
* Streamlit

## 🔄 Workflow

```text
Email Text
    ↓
Lowercase Conversion
    ↓
Punctuation Removal
    ↓
Word Tokenization
    ↓
Stopword Removal
    ↓
Keras Tokenization
    ↓
Sequence Padding
    ↓
Word Embedding
    ↓
GRU Neural Network
    ↓
Sigmoid Classification
    ↓
Spam / Ham
```

## 🤖 Model Architecture

The final model uses:

* **Embedding Layer:** 5,000 vocabulary size, 32-dimensional embeddings
* **GRU Layer:** 64 units
* **Output Layer:** 1 neuron with sigmoid activation
* **Maximum Sequence Length:** 500
* **Classification:** Binary (Ham / Spam)

The GRU model was selected after comparing **Simple RNN, LSTM, and GRU** models using accuracy and Spam F1-score.

## 📂 Project Structure

```text
email-spam-detection-RNN/
│
├── app.py
├── combined_data.csv
├── gru_model.keras
├── tokenizer.pkl
├── label_mapping.pkl
├── config.pkl
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Gargik283/email-spam-detection-RNN.git
cd email-spam-detection-RNN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Demo

**Streamlit App:** https://email-spam-detection-rnn-7zvqcr32qdwhk5un9xej85.streamlit.app/

## 📌 Key Concepts

* Text preprocessing
* NLP tokenization
* Stopword removal
* Sequence padding
* Word embeddings
* Recurrent Neural Networks
* GRU architecture
* Binary classification
* Model evaluation
* Streamlit deployment

## 👩‍💻 Author

**Gargi Kundu**

B.Tech – Electronics & Communication Engineering (VLSI Design), 2025

GitHub: https://github.com/Gargik283

```

⭐ If you find this project useful, consider giving the repository a star!
