import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("dataset.csv",encoding='cp1254',delimiter=';'); df = df.iloc[:1490]

X=df.comment
y=df.Label
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X = X.toarray()
# Kelimelerin verisetinde ne sıklıkla bulundukları belirlendi. (kelimelerin ağırlıkları)
# Sıklıkları bulunan kelimeler, verisetinin feature'larını oluşturdu ve veri eğitime hazır hale getirildi.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=42)

models = ["BernoulliNB()","MultinomialNB()","SVC(kernel='linear')","MLPClassifier(alpha=0.01, batch_size=len(X_train), epsilon=1e-08, hidden_layer_sizes=(64), learning_rate='adaptive', max_iter=500)"]
names = ["Bernoulli Naive Bayes","Multinomial Naive Bayes","Support Vector Machines","Multilayer Perceptron",]
accuracy=[]
for i,model in enumerate(models):
    model = eval(model)
    model.fit(X_train,y_train)
    y_predict= model.predict(X_test)
    print(f"\n\nAccuracy for {names[i]}: {accuracy_score(y_test,y_predict)}")
    accuracy.append(accuracy_score(y_test,y_predict))
    #print(classification_report(y_test,y_predict))
    print(f"\nConfusion matrix for {names[i]}:\n{pd.crosstab(y_test,y_predict)}")

# döngü ile modeller sırasıyla eğitildi ve test sonuçları yazdırıldı.

accuracy=np.array(accuracy)
maxIndex = np.argmax(accuracy)
bestModel = eval(models[maxIndex])
print(f"\nEn iyi model ({names[maxIndex]}) kaydediliyor. Bekleyiniz...")
bestModel.fit(X,y)  # en iyi model, tüm verisetiyle tekrar eğitiliyor...
filename = 'finalized_model.sav'
pickle.dump(bestModel, open(filename, 'wb')) # En iyi doğruluk oranına sahip olan model kaydedildi.
print("Model kaydedildi. Chatbot'a geçilebilir...")

# Not: SVM ve MLP'nin eğitimi diğerlerine göre biraz daha uzun sürmektedir...


