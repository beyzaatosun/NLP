import pandas as pd

df = pd.read_csv("spam.csv",encoding="latin-1")

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1, inplace=True)

df.columns = ["label", "text"]

print(df.isnull().sum())

import nltk
nltk.download("stopwords") #cok kullanılan ve anlam tasimayanları cikartmak icin
nltk.download("wordnet") #lema bulmak icin
nltk.download("omw-1.4") #wordnete ait farkli dilleri kelime anlamlarini iceren bir veri seti 

import re
from nltk.corpus import stopwords #stopwordsleri kaldirmak icin
from nltk.stem import WordNetLemmatizer #lemmatization

text = list(df.text)
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub("[^a-zA-Z]"," ",text[i]) #harf olmayan karakterler silinir
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words("english")]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = " ".join(r)
    corpus.append(r)
    
df["text2"] = corpus

X = df["text2"]
y = df["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

#feature extraction : bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

#classifier tranining 

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train)

X_test_cv = cv.transform(X_test)

prediction = dt.predict(X_test_cv)

from sklearn.metrics import confusion_matrix, accuracy_score

c_matrix = confusion_matrix(y_test, prediction)
print("Confusion Matrix:", c_matrix)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)









