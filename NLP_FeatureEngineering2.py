# %%WORD EMBEDDING - IMDB DATASET
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models  import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("IMDB Dataset.csv")

documents = df["review"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+","",text) #sayiları temizle
    text = re.sub(r"[^\w\s]","",text)#ozel karakterleri temizle
    text = " ".join([word for word in text.split() if len(word)>2])
    
    return text

cleaned_documents = [clean_text(row) for row in documents]

tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents]

model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5,min_count=1,sg=0)
word_vector = model.wv

words = list(word_vector.index_to_key)[:500]
vectors = [word_vector[word] for word in words]

kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters = kmeans.labels_

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c=clusters , cmap="viridis")

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c="red" ,marker="x", s=150, label="Center")


for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0], reduced_vectors[i,1], word, fontsize=7)
plt.title("Word2Vec")


#%%Text Representation Transformers Based
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch

model_name="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Transformers can be used for natural language processing."
#metni tokenlara cevirmek
inputs = tokenizer(text, return_tensors ="pt") #cıktı pytorch tensoru olarak return edilir

#modeli kullanarak metin temsili olustur
with torch.no_grad(): #gradyanların hesaplanmasi durdurulur boylece bellegi daha verimli kullanılırız
    output = model(**inputs)

#modelin cikisindan son gizli durumu alalim
last_hidden_state = output.last_hidden_state #tum token ciktilarini almak icin

first_token_embedding = last_hidden_state[0,0,:].numpy()

print(f"metin temsili:{first_token_embedding}")
























