import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fungsi untuk memindahkan model dan input ke perangkat yang sesuai
def predict(text, tokenizer, model):
    # Tokenisasi input teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Tentukan perangkat yang sesuai (CPU atau GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pindahkan input tensor ke perangkat yang sesuai (CPU atau GPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Lakukan prediksi tanpa memindahkan model
    with torch.no_grad():
        logits = model(**inputs).logits  # Prediksi langsung
        prediction = torch.argmax(logits, dim=-1).cpu().item()  # Ambil hasil prediksi dan pindahkan ke CPU
    
    return prediction

# Muat tokenizer dan model dari Hugging Face
model_name = 'reyhannandita/analisissentimen'  # Ganti dengan nama model di Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# UI Streamlit
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
