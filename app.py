import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Path ke folder model
model_path = 'saved_model'  # Sesuaikan dengan path model Anda

# Memuat model DistilBERT untuk klasifikasi (dengan output hidden states)
model = DistilBertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Fungsi untuk prediksi dan mendapatkan embedding
def predict_sentiment_and_embedding(text):
    # Tokenisasi input teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Prediksi menggunakan model dengan hidden states
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mengambil hidden states
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]  # Mengambil last hidden state
    embeddings = last_hidden_state.mean(dim=1).squeeze().tolist()  # Rata-rata embedding
    predictions = torch.argmax(outputs.logits, dim=-1)  # Prediksi sentimen (0: negatif, 1: positif)
    
    return embeddings, predictions.item()

# Judul dan penjelasan aplikasi
st.title('Sentiment Analysis - Movie Reviews with DistilBERT')
st.write("Enter a movie review, and the model will predict whether the sentiment is positive or negative.")

# Input teks dari pengguna
user_input = st.text_area("Enter your movie review here:")

# Tombol Submit
if st.button("Submit"):
    if user_input:
        # Mendapatkan embedding dan prediksi sentimen
        embeddings, sentiment = predict_sentiment_and_embedding(user_input)
        
        # Menampilkan hasil prediksi
        if sentiment == 1:
            st.success("The sentiment is **Positive**!")
        else:
            st.error("The sentiment is **Negative**!")
        
        # Menampilkan embedding
        st.write("Embedding (vector representation of the text):")
        st.write(embeddings)  # Menampilkan embedding sebagai vektor
    else:
        st.warning("Please enter a movie review to analyze.")
