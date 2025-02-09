#Sentiment Analysis in NLP (with RNN)
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = {
    "text": [
        "Yemekler çok güzeldi, gerçekten harika bir deneyimdi.",
        "Servis çok hızlıydı ve garsonlar oldukça nazikti.",
        "Restoranın atmosferi çok rahatlatıcıydı, çok beğendim.",
        "Mükemmel bir akşam yemeğiydi, kesinlikle tekrar gelirim.",
        "Yemekler taze ve lezzetliydi, ortam harikaydı.",
        "Restoran çok gürültülüydü, bir daha gelmem.",
        "Yemekler soğuktu ve garsonlar ilgisizdi, çok kötü bir deneyim yaşadım.",
        "Çalışanlar çok ilgiliydi, yemeğim mükemmeldi.",
        "Yemeklerin tadı harikaydı, kesinlikle tekrar geleceğim.",
        "Hizmet çok yavaştı, yemekleri beklerken çok sıkıldım.",
        "Yemeklerin sunumu mükemmeldi ve porsiyonlar çok büyüktü.",
        "Çalışanlar çok güler yüzlüydü, yemekler harikaydı.",
        "Restoran çok temizdi ve yemekler çok lezzetliydi.",
        "Gerçekten çok keyifli bir akşam yemeğiydi, kesinlikle tavsiye ederim.",
        "Burası kesinlikle favori mekanım oldu. Her şey harikaydı.",
        "Restoranın ortamı çok şık, yemekler de mükemmeldi.",
        "Yemekler oldukça taze ve çok lezzetliydi, çok memnun kaldık.",
        "Restoran çok lüks, servis de harikaydı.",
        "Bu kadar lezzetli yemekler yemek için uzun süre beklemeye değdi.",
        "Yemeklerin sunumu gerçekten harikaydı, çok beğendim.",
        "Yemekler oldukça sıradandı, bir daha gelmeyi düşünmüyorum.",
        "Hizmet çok yavaştı, garsonlar ilgisizdi.",
        "Restoran çok kirliydi, yemekler de beklediğim gibi değildi.",
        "Yemeklerin lezzeti oldukça vasattı, kesinlikle tavsiye etmem.",
        "Beklediğimizin aksine, hizmet çok kötüydü.",
        "Restoranın atmosferi harikaydı, yemekler çok lezzetliydi.",
        "Garsonlar oldukça ilgisizdi, yemekler geç geldi.",
        "Servis çok hızlıydı ve garsonlar oldukça profesyoneldi.",
        "Restoran çok kalabalıktı, hizmet biraz yavaşladı.",
        "Yemekler çok lezzetliydi ancak restoranın dekorasyonu oldukça sıradandı.",
        "Yemekler soğuk geldi, tekrar gelmeyi düşünmüyorum.",
        "Çalışanlar çok güler yüzlüydü, yemekler çok lezzetliydi.",
        "Yemeklerin sunumu harikaydı, restoranın havası çok hoştu.",
        "Restoran oldukça temizdi, hizmet harikaydı.",
        "Yemekler beklediğimizin çok altında kaldı, bir daha gelmem.",
        "Hizmet mükemmeldi, yemekler ise harikaydı.",
        "Burası oldukça lüks, yemekler çok kaliteli.",
        "Restoran çok sesliydi, rahat bir akşam yemeği deneyimi yaşamadım.",
        "Yemeklerin tadı vasattı, biraz daha özen gösterilebilir.",
        "Atmosfer çok güzeldi ama yemekler pek hoşuma gitmedi.",
        "Yemekler çok güzeldi ama garsonlar biraz daha dikkatli olabilirdi.",
        "Çalışanlar çok nazikti ama yemekler o kadar lezzetli değildi.",
        "Mekan oldukça şık, yemekler de lezzetliydi.",
        "Yemekler beklediğimin çok üzerinde, mükemmel bir deneyimdi.",
        "Hizmet çok hızlıydı ama yemekler beklediğimin çok gerisindeydi.",
        "Çalışanlar ilgiliydi, ama restoran çok karışıktı.",
        "Yemekler çok lezzetliydi, fakat fiyatlar biraz yüksekti.",
        "Restoranın iç dekorasyonu çok güzeldi, yemekler ise çok lezzetliydi.",
        "Yemekler güzel ama garsonlar biraz daha dikkatli olmalı.",
        "Atmosfer çok sıcak ve davetkar, yemekler ise mükemmeldi."
        
    ],
    "label": [
        "positive", "positive", "positive", "positive", "positive", "negative", "negative", "positive", "positive", "negative",
        "positive", "positive", "positive", "positive", "positive", "negative", "negative", "positive", "positive", "negative",
        "positive", "negative", "negative", "negative", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "positive", "positive", "positive", "negative",
        "positive", "positive", "negative", "negative", "negative", "positive", "negative", "positive", "positive", "negative",
        "positive", "negative", "positive", "positive", "negative"
    ]
}

df = pd.DataFrame(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

#padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)

#label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

#train test split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Word Embedding - Word2Vec

sentences = [text.split() for text in df["text"]]
word2vec_model= Word2Vec(sentences, vector_size =50, window = 5, min_count=1)


embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

model = Sequential()

model.add(Embedding(input_dim = len(word_index) + 1, output_dim =embedding_dim, weights= [embedding_matrix], input_length=maxlen ,trainable = False))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(1,activation="sigmoid"))


model.compile(optimizer ="adam", loss="binary_crossentropy",metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test,y_test))

test_loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {accuracy}")
#%%
def classify_sentences(sentence):
    seq = tokenizer.texts_to_sequences(sentence)
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    prediction = model.predict(padded_seq)
    predicted_class = (prediction>0.5).astype(int)
    label="positive" if predicted_class[0][0] ==1 else "negative"
    return prediction 



sentence="Yemekler çok güzeldi memnun kaldık"

result = classify_sentences(sentence)
print(f"Result: {result}")