#NAME ENTITY RECOGNITION

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

content = "Alice works at Amazon and lives in London. She visited the British Museum last weekend."

doc = nlp(content)

for ent in doc.ents:
    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print(ent.text, ent.label_)

entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]

df = pd.DataFrame(entities, columns = ["text","type","lemma"])

