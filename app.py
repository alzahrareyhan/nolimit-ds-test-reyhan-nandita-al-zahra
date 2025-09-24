import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset Dummy untuk Latihan (ganti dengan dataset asli Anda)
train_texts = ["I love this movie", "This movie is bad", "Amazing film", "Not worth watching", "Great movie"]
train_labels = [1, 0, 1, 0, 1]  # 1 = Positif, 0 = Negatif

# Membuat TF-IDF Vectorizer
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
st.title('Sentiment Analysis - Movie Reviews')
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
    else:
        st.warning("Please enter a movie review to analyze.")
