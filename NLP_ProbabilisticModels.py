#%% NGRAM Models
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

corpus = ["I hata apple",
          "I hate NLP",
          "You hate orange",
          "They hate apple",
          "He hates water",
          "I hate you",
          "I hate him"]

"""
dil modeli yapmak
amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek : metin uretmek/olusturmak
"""

tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

#bigram 2 li kelime gruplari
bigrams= []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list,2)))
    
bigrams_freq = Counter(bigrams)

#trigram

trigram =[]
for token_list in tokens:
    trigram.extend(list(ngrams(token_list, 3)))
    
trigram_freq = Counter(trigram)


bigram =("i","hate")

prob_you = trigram_freq[("i","hate","you")]/bigrams_freq[bigram]
print(f"you kelimesinin olma olasiligi {prob_you}")

prob_apple = trigram_freq[("i","hate","apple")]/bigrams_freq[bigram]
print(f"apple kelimesinin olma olasiligi {prob_apple}")



#%% Hidden Markov Modeli - part of speech

import nltk
from nltk.tag import hmm

train_data = [[("I","PRP"),("am","VBP"),("a","DT"),("engineer","NN")],
             [("You","PRP"),("are","VBP"),("a","DT"),("police","NN")]]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

test_sentence = "I am a police".split()

tags = hmm_tagger.tag(test_sentence)
print(f"New Sentences: {tags}")


#%% Hidden Markov Models - part of speech2

import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

nltk.download("conll2000")

train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


test_sentence = "I hate run".split()
tags= hmm_tagger.tag(test_sentence)

print(f"New Sentences: {tags}")

#%% Maximum Entropy Models 

from nltk.classify import MaxentClassifier

train_data=[
    ({"love":True, "amazing":True, "happy":True, "terrible":False},"positive"),
    ({"hate":True,"terrible":True},"negative"),
    ({"joy":True, "happy":True,"hate":False},"positive"),
    ({"sad":True, "depressed":True, "love":False},"negative")
    ]

#train max entropy classifier
classifier = MaxentClassifier.train(train_data, max_iter=10)

test_sentences = "I love this movie and it was terrible"
features = {word:(word in test_sentences.lower().split()) for word in ["love", "amazing","happy", "joy","sad","terrible","hate","depressed"]}

label = classifier.classify(features)
print(f"result: {label}")






