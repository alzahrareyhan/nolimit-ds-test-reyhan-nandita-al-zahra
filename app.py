import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Memuat model DistilBERT untuk klasifikasi sentimen dan tokenizer dari Hugging Face
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Memastikan model menggunakan perangkat yang tepat (GPU atau CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fungsi untuk mendapatkan embedding dan prediksi sentimen
def predict_sentiment_and_embedding(text):
    # Tokenisasi input teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Prediksi menggunakan model dengan hidden states
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mengambil logits dan menghitung prediksi sentimen (0 = Negatif, 1 = Positif)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # Mengambil prediksi dengan nilai tertinggi
    
    # Mengambil hidden states untuk embedding
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]  # Mengambil last hidden state
    embeddings = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Rata-rata embedding
    
    return embeddings, predictions.item()

# Dataset Dummy untuk Latihan (ganti dengan dataset asli Anda)
train_texts = ["I love this movie", "This movie is bad", "Amazing film", "Not worth watching", "Great movie"]
train_labels = [1, 0, 1, 0, 1]  # 1 = Positif, 0 = Negatif

# Membuat TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

# Mendapatkan fitur TF-IDF untuk teks
train_features = vectorizer.fit_transform(train_texts)

# Membagi data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

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
    # Mengubah input teks ke dalam bentuk TF-IDF
    text_features = vectorizer.transform([text])
    sentiment = clf.predict(text_features)  # Prediksi sentimen menggunakan model sklearn
    return sentiment[0]  # Mengembalikan hasil prediksi

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
        
        # Menampilkan embedding (hanya sebagian agar tidak terlalu panjang)
        st.write("Embedding (vector representation of the text):")
        st.write(embeddings[:10])  # Menampilkan hanya sebagian dari embedding (misalnya 10 nilai pertama)
        
        # Menampilkan embedding panjang secara lebih terstruktur (opsional)
        if len(embeddings) > 10:
            st.write("... and more values in the embedding vector.")
    else:
        st.warning("Please enter a movie review to analyze.")
