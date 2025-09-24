import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Fungsi untuk memuat model dan tokenizer dari Hugging Face
@st.cache_resource
def load_model():
    # Ganti dengan nama model di Hugging Face
    model_name = 'reyhannandita/analisissentimen'  
    # Memuat tokenizer dan model dari Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return tokenizer, model

import torch

import torch

# Fungsi untuk memindahkan model dan input ke perangkat yang sesuai
def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Tentukan perangkat CPU secara eksplisit
    device = torch.device('cpu')  # Menggunakan CPU untuk memastikan kompatibilitas

    # Gunakan `to_empty()` untuk memindahkan model jika tensor meta
    model = model.to_empty(device)  # Menggunakan to_empty() untuk model yang belum sepenuhnya dimuat
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Melakukan prediksi
    with torch.no_grad():
        logits = model(**inputs).logits
        
        # Pastikan tensor berada di CPU sebelum menggunakan .item()
        prediction = torch.argmax(logits, dim=-1).cpu()
        return prediction.item()

# Memuat model
tokenizer, model = load_model()

# Streamlit UI
st.title("Sentiment Analysis with Hugging Face Model")
st.write("Masukkan teks untuk analisis sentimen")

# Input teks dari pengguna
user_input = st.text_area("Teks:", "")

# Tombol untuk prediksi
if st.button('Prediksi'):
    if user_input:
        prediction = predict(user_input, tokenizer, model)
        if prediction == 1:
            st.write("Prediksi: Positif")
        else:
            st.write("Prediksi: Negatif")
    else:
        st.write("Harap masukkan teks untuk dianalisis.")
