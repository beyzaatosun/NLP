import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "She enjoys reading books in the evening while drinking tea."
doc = nlp(sentence)

for token in doc:
    print(token.text, token.pos_)
    
