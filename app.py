import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Path ke folder model
model_path = 'saved_model'  # Sesuaikan dengan path model Anda

# Memuat model dan tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Fungsi untuk prediksi
def predict_sentiment(text):
    # Tokenisasi input teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Prediksi menggunakan model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Menghitung prediksi (0: negatif, 1: positif)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Judul dan penjelasan aplikasi
st.title('Sentiment Analysis - Movie Reviews with DistilBERT')
st.write("Enter a movie review, and the model will predict whether the sentiment is positive or negative.")

# Input teks dari pengguna
user_input = st.text_area("Enter your movie review here:")

# Tombol Submit
if st.button("Submit"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        
        # Menampilkan hasil prediksi
        if sentiment == 1:
            st.success("The sentiment is **Positive**!")
        else:
            st.error("The sentiment is **Negative**!")
    else:
        st.warning("Please enter a movie review to analyze.")
