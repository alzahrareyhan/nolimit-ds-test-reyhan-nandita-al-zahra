import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@st.cache_resource
def load_model():
    model_path = 'saved_model/model'  # Gunakan path tanpa ekstensi safetensors jika bisa menggunakan model standar
    
    # Tokenizer dan model standar Hugging Face
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    return tokenizer, model

# Fungsi untuk prediksi
def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1)
    return prediction.item()

# Memuat model
tokenizer, model = load_model()

# Mengatur UI Streamlit
st.title("Sentiment Analysis with Model.safetensors")
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

