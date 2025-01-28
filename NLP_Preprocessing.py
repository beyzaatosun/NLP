#%% VERI TEMIZLEME
#bosluklari split etme
text="Hello,        World!    2035"
text.split()
cleaned_text1=" ".join(text.split())
print(f"text: {text} \ncleaned text: {cleaned_text1}")

#%% buyuk - kucuk harf cevrimi
text="Hello, World! 2035"
cleaned_text2=text.lower()
print(f"text: {text} \ncleaned text: {cleaned_text2}")
#%% noktalama isaretleri kaldirma
import string

text="Hello, World! 2035"
cleaned_text3 = text.translate(str.maketrans("","",string.punctuation))
print(f"text: {text} \ncleaned text: {cleaned_text3}")
#%% ozel karakterleri kaldir , %,#,@,/,*,#

import re

text="Hello, World! 2035%"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]","",text)
print(f"text: {text} \ncleaned text: {cleaned_text4}")

#%% yazim hatalarini duzelt
from textblob import TextBlob

text="Hellio, Wirld! 2035%"

cleaned_text5 = TextBlob(text).correct()
print(f"text: {text} \ncleaned text: {cleaned_text5}")

#%% html ya da url etiketleri kaldir
from bs4 import BeautifulSoup

html_text= "<div>Hello, World 2035<div>"
cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(f"text: {html_text} \ncleaned text: {cleaned_text6}")

#%% TOKENIZATION
import nltk
nltk.download("punkt") #kelime ve cumle bazinda tokenlara ayirmak icin gerekli
text = "Hello, World! How are you? Hello, hi ..."

#klime tokenization - word tokenization : metni kelimelere ayirir, 
#noktalama isaretleri ve bosluklar ayri birer token olarak elde edilecektir
word_tokens = nltk.word_tokenize(text)

#cumle tokenization - sent_tokenization : metni cumlelere ayirir, her cumle birer token olur
sentence_token = nltk.sent_tokenize(text)

#%% STEMMING AND LEMMATIZATION
import nltk
nltk.download("wordnet") #lemmatization
from nltk.stem import PorterStemmer #stemming

#porter stemmer nesnesi olusturma
stemmer = PorterStemmer()
words = ["running","runner","ran","runs","better","go","went"]

stems = [stemmer.stem(w) for w in words]
print(f"Stems: {stems}")

#%% LEMMATIZATION

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = ["running","runner","ran","runs","better","go","went"]

lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words] #pos="v" verb olarak almasını saglanir
print(f"lemma: {lemmas}")

#%% STOP WORDS
import nltk
from nltk.corpus  import stopwords

nltk.download("stopwords")
stop_words_eng = set(stopwords.words("english"))

text = "There are some examples of handling stop words from some texts."
text_list = text.split()
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered word: {filtered_words}")
















