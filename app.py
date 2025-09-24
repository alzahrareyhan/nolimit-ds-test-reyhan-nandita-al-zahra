import streamlit as st
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Path ke folder model
model_path = 'saved_model'  # Sesuaikan dengan path model Anda

# Memuat model DistilBERT untuk ekstraksi embedding
distilbert_model = DistilBertModel.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Fungsi untuk mendapatkan embedding
def get_embedding(text):
    # Tokenisasi input teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Mengambil embedding dari DistilBERT
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    
    # Mengambil last hidden state dan menghitung rata-rata untuk embedding
    last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze()

    # Pastikan tensor ada di CPU sebelum memanggil .numpy() dan pastikan tensor bukan meta tensor
    if last_hidden_state.device.type == 'meta':
        raise ValueError("The model did not return valid data (meta tensor).")
    
    return last_hidden_state.cpu().numpy()


# Data latih dan label
train_texts = ["I love this movie", "This movie is bad", "Amazing film", "Not worth watching", "Great movie"]
train_labels = [1, 0, 1, 0, 1]  # 1 = Positif, 0 = Negatif

# Mendapatkan embedding untuk teks
train_embeddings = np.array([get_embedding(text) for text in train_texts])

# Membagi data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(train_embeddings, train_labels, test_size=0.2, random_state=42)

# Latih model menggunakan Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Prediksi dengan model yang telah dilatih
y_pred = clf.predict(X_test)

# Evaluasi akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan akurasi
st.write(f'Accuracy: {accuracy * 100:.2f}%')

# Fungsi untuk prediksi sentimen menggunakan model sklearn
def predict_sentiment(text):
    embedding = get_embedding(text)  # Dapatkan embedding
    sentiment = clf.predict([embedding])  # Prediksi sentimen menggunakan model sklearn
    return sentiment[0]  # Mengembalikan hasil prediksi

# Fungsi untuk mencari ulasan terdekat menggunakan cosine similarity
def find_closest_reviews(user_embedding, dataset_embeddings, dataset_texts):
    # Menghitung cosine similarity antara embedding input dan dataset embeddings
    similarities = cosine_similarity([user_embedding], dataset_embeddings)
    
    # Menemukan indeks ulasan dengan similarity tertinggi
    closest_indices = similarities.argsort()[0][-3:][::-1]  # Ambil 3 ulasan terdekat
    closest_reviews = [(dataset_texts[i], similarities[0][i]) for i in closest_indices]
    
    return closest_reviews

# Judul dan penjelasan aplikasi
st.title('IMDB Sentiment Analysis with DistilBERT and Scikit-Learn')
st.write("Enter a movie review, and the model will predict whether the sentiment is positive or negative.")

# Input teks dari pengguna
user_input = st.text_area("Enter your movie review here:")

# Tombol Submit
if st.button("Submit"):
    if user_input:
        # Prediksi sentimen dengan model sklearn
        sentiment = predict_sentiment(user_input)
        
        # Menampilkan hasil prediksi
        if sentiment == 1:
            st.success("The sentiment is **Positive**!")
        else:
            st.error("The sentiment is **Negative**!")
        
        # Mendapatkan embedding dari input pengguna
        user_embedding = get_embedding(user_input)
        
        # Menemukan ulasan terdekat
        closest_reviews = find_closest_reviews(user_embedding, train_embeddings, train_texts)
        
        # Menampilkan ulasan terdekat dan jaraknya
        st.write("Closest Reviews:")
        for review, similarity in closest_reviews:
            st.write(f"Review: {review}")
            st.write(f"Similarity: {similarity:.2f}")
    else:
        st.warning("Please enter a movie review to analyze.")
