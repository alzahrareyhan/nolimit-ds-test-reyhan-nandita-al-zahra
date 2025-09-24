import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from safetensors.torch import load_file

# Path ke folder model
model_path = 'saved_model'

# Memuat state_dict dari file safetensors
state_dict = load_file(f"{model_path}/model.safetensors")

# Tentukan perangkat yang digunakan (menggunakan CPU)
device = torch.device("cpu")  # Gunakan CPU untuk deployment

# Inisialisasi model DistilBERT dengan konfigurasi yang sesuai
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Memuat state_dict ke model
model.load_state_dict(state_dict)

# Memindahkan model ke perangkat yang sesuai (CPU)
model.to(device)

# Memuat tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Fungsi untuk prediksi sentimen menggunakan model DistilBERT
def predict_sentiment(text):
    # Tokenisasi input teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Prediksi menggunakan model dengan hidden states
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Cek apakah tensor valid sebelum menggunakan .item()
    logits = outputs.logits
    if logits.is_meta:
        raise RuntimeError("Received meta tensor, cannot perform .item()")

    predictions = torch.argmax(logits, dim=-1)  # Mengambil prediksi dengan nilai tertinggi
    
    return predictions.item()

# Judul dan penjelasan aplikasi
st.title('Sentiment Analysis - Movie Reviews with DistilBERT')
st.write("Enter a movie review, and the model will predict whether the sentiment is positive or negative.")

# Input teks dari pengguna
user_input = st.text_area("Enter your movie review here:")

# Tombol Submit
if st.button("Submit"):
    if user_input:
        try:
            # Prediksi sentimen dengan model DistilBERT
            sentiment = predict_sentiment(user_input)
            
            # Menampilkan hasil prediksi
            if sentiment == 1:
                st.success("The sentiment is **Positive**!")
            else:
                st.error("The sentiment is **Negative**!")
        except RuntimeError as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a movie review to analyze.")
