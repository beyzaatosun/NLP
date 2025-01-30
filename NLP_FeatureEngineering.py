#%% BAG OF WORDS(BOW) LOGIC
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "kedi bahcede",
    "kedi evde"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
#kelime kumesi olusturma
feature_names = vectorizer.get_feature_names_out()
print(f"kelime kumesi: {feature_names}")
#vektor temsili
vector_temsili = X.toarray()

print(f"vektor temsili: {vector_temsili}")
#%% BAG OF WORDS ON IMDB DATASETS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("IMDB Dataset.csv")
documents = df["review"]
labels = df["sentiment"]
nltk.download("stopwords")
stop_words_eng = set(stopwords.words("english"))

def clean_text(text):
    #buyuk-kucuk harf cevrimi
    text = text.lower()
    #rakamlari temizleme
    text = re.sub(r"\d+","",text)
    #ozel karakterleri kaldırma
    text = re.sub(r"[^\w\s]","",text)
    #kisa kelimeleri temizleme
    text = " ".join([word for word in text.split() if len(word)>2])
 
    return text

cleaned_doc = [clean_text(row) for row in documents]

vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(cleaned_doc)

feature_names = vectorizer.get_feature_names_out()

vektor_temsili = X.toarray()
print(f"Vektor temsili: {vektor_temsili}")

df_bow = pd.DataFrame(vektor_temsili, columns = feature_names)

word_counts = X.sum(axis = 0).A1
word_freq = dict(zip(feature_names, word_counts))


top_5_words = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

# Sonuçları yazdırıyoruz
print("En çok kullanılan 5 kelime:")
for word in top_5_words:
    print(f"{word}: {word_freq[word]}")

#%% TF- IDF 
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "köpek çok tatlı bir hayvandır",
    "köpek ve kuşlar çok tatlı hayvanlardır",
    "inekler süt üretirler."]

tfidf_vectorizer = TfidfVectorizer()

X = tfidf_vectorizer.fit_transform(documents)

feature_names = tfidf_vectorizer.get_feature_names_out()

vektor_temsili = X.toarray()
print(f"Vektor temsili: {vektor_temsili}")


df_tfidf = pd.DataFrame(vektor_temsili, columns=feature_names)

tf_idf = df_tfidf.mean(axis=0)

#%% TFIDF SMS SPAM 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re 

df = pd.read_csv("spam.csv",encoding='latin-1')

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r"\d+","",text)
    
    text = re.sub(r"[^\w\s]","",text)
    #kisa kelimeleri temizleme
    text = " ".join([word for word in text.split() if len(word)>2]) 
    return text 

cleaned_doc = [clean_text(row) for row in df.v2]

vectorizer = TfidfVectorizer(stop_words='english')
X=vectorizer.fit_transform(cleaned_doc)

feature_names = vectorizer.get_feature_names_out()

tfidf_score = X.mean(axis=0).A1

df_tfidf = pd.DataFrame({"word":feature_names,
                         "tfidf_score":tfidf_score})
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score",ascending=False)
print(df_tfidf_sorted.head(10))

#%% NGRAM
from sklearn.feature_extraction.text import CountVectorizer

documents = ["bu çalişma NGRAM çalışmasıdır.",
             "bu çalişma doğal dil işleme çalışmasıdır."]

vectorizer_unigram = CountVectorizer(ngram_range = (1,1))
vectorizer_bigram = CountVectorizer(ngram_range = (2,2))
vectorizer_trigram = CountVectorizer(ngram_range = (3,3))

X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_feature = vectorizer_unigram.get_feature_names_out()

X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_feature = vectorizer_bigram.get_feature_names_out()

X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_feature = vectorizer_trigram.get_feature_names_out()


print(f"unigram_feature: {unigram_feature}\n")

print(f"bigram_feature: {bigram_feature}\n")

print(f"trigram_feature: {trigram_feature}")
#%% WORD EMBEDDING

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

sentences =[
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi severler." ,
    "Köpekler sadık ve dost canlısı hayvanlardır",
    "hayvanlar insanlar için iyi arkadaşlardır"]

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

word2_vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

fasttext_model = FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

def plot_word_embedding(model, title):
    word_vectors = model.wv
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]
    
    pca= PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection="3d")
    
    ax.scatter(reduced_vectors[:,0], reduced_vectors[:,1],reduced_vectors[:,2])
    
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i,0], reduced_vectors[i,1],reduced_vectors[i,2], word, fontsize=12)
        
    ax.set_title(title)
    ax.set_xlabel("Componenet 1")
    ax.set_ylabel("Componenet 2")
    ax.set_zlabel("Componenet 3")
    plt.show()
        
plot_word_embedding(word2_vec_model, "Word2Vec")
plot_word_embedding(fasttext_model, "FastText")







