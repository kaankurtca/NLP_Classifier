import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pickle


df = pd.read_csv("dataset.csv",encoding='cp1254',delimiter=';'); df = df.iloc[:1490]
X=df.comment
y=df.Label
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
model = pickle.load(open('finalized_model.sav', 'rb'))

print("\nChatbot Açılıyor...")
time.sleep(2)
print("Sohbete başlayabilirsiniz...\n")
while True:
    X = input("Kullanıcı >> ")
    if "çıkış" in X:
        print("\nChatbot kapatılıyor...")
        break
    newRequest = pd.Series([X])
    newRequest = vectorizer.transform(newRequest).toarray()
    prediction = model.predict(newRequest)
    print(f"Chatbot >> Girdiğiniz yazı {prediction[0]} sınıfına aittir.")


