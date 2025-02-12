import spacy

nlp = spacy.load("en_core_web_sm")

word = "The cat sat on the mat, watching the birds outside the window."

doc = nlp(word)

for token in doc:
    print(f"Text: {token.text}")         #kelimenin kendisi 
    print(f"Lemma: {token.lemma_}")      #kelimenin kok hali
    print(f"POS: {token.pos_}")          #kelimenin dilbigisel ozelligi
    print(f"Tag: {token.tag_}")          #kelimenin detaylı dilbigisel ozelligi
    print(f"Dependency: {token.dep_}")   #kelimenin rolu
    print(f"Shape: {token.shape_}")      #kelimenin yapisi
    print(f"Is alpha: {token.is_alpha}") #kelimenin yalnizca alfabetik karakterlerden olusup olusmadigi
    print(f"Is stop: {token.is_stop}")   #kelimenin stop words olup olmadigi
    print(f"Morfoloji: {token.morph}")   #kelimenin morfolojij ozelligi
    print(f"Is plural: {'Number=Plur' in token.morph}")   #kelimenin cogul olup olmadigi

    print() 
    