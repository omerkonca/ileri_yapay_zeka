# Gerekli kütüphaneleri yükleyin
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re

# Veri setini Hugging Face'den yükleyin
dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")

# Veri setinin ilk birkaç satırını inceleyin
print(dataset["train"].to_pandas().head())

# Veriyi pandas DataFrame'e dönüştürme
train_data = dataset["train"].to_pandas()

# Etiketlerin dağılımını inceleyin
label_counts = train_data['label'].value_counts()
print("Etiket Dağılımı:\n", label_counts)

# Etiket dağılımını görselleştirme
label_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Etiket Dağılımı')
plt.xlabel('Etiketler (Negative, Notr, Positive)')
plt.ylabel('Frekans')
plt.show()

# Veri temizleme fonksiyonu
def clean_text(text):
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r'http\S+', '', text)  # URL'leri kaldır
    text = re.sub(r'[^a-zçğıöşü0-9\s]', '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları temizle
    return text

# Metinleri temizleme
train_data['text'] = train_data['text'].apply(clean_text)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    train_data['text'], train_data['label'], test_size=0.2, random_state=42
)

# TF-IDF ile metinleri sayısallaştırma
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# İlk birkaç örneği kontrol edin
print("Eğitim Seti İlk 5 Örnek:\n", X_train.head())
print("TF-IDF ile Sayısallaştırılmış İlk Örnek:", X_train_tfidf[0])

# Logistic Regression Modeli Eğitimi
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Modelin Test Setindeki Performansı
y_pred = model.predict(X_test_tfidf)
print("Model Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
