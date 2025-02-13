#%% NLTK Lesk
import nltk 
from nltk.wsd import lesk

nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")

s1 = "They won the tennis match yesterday."
w1 = "match"

sense1 = lesk(nltk.word_tokenize(s1),w1)
print(f"Cumle: {s1}")
print(f"Word: {w1}")
print(f"Sense: {sense1.definition()}")
print()

s2 = "This shirt is a perfect match for those pants."
w2 = "match"

sense2 = lesk(nltk.word_tokenize(s2),w2)
print(f"Cumle: {s2}")
print(f"Word: {w2}")
print(f"Sense: {sense2.definition()}")

"""
Sense: strike with, or as if with a baseball bat

Sense: have a turn at bat
"""

#%% ADAPTED Lesk
import nltk
nltk.download('averaged_perceptron_tagger')
from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk

sentences = [
    "They won the tennis match yesterday.",
    "This shirt is a perfect match for those pants."
    ]


word = "match"

for sentence in sentences:
    print(f"Sentence:{sentence}")
    
    sense_simple_lesk = simple_lesk(sentence, word)
    print(f"Simple Sentence:{sense_simple_lesk.definition()}")
    
    sense_adapted_lesk = adapted_lesk(sentence, word)
    print(f"Adapted Sentence:{sense_adapted_lesk.definition()}")

    sense_cosine_lesk = cosine_lesk(sentence, word)
    print(f"Cosine Sentence:{sense_cosine_lesk.definition()}")


