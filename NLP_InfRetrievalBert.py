from transformers import BertTokenizer, BertModel
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

documents = [
    "Machine learning algorithms are capable of analyzing large datasets to uncover patterns and make predictions without being explicitly programmed.",
    "Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and human language.",
    "Deep learning, a subset of machine learning, uses neural networks with many layers to model complex patterns in data.",
    "AI is revolutionizing industries such as healthcare, finance, and transportation by providing innovative solutions to complex problems.",
    "NLP techniques, like sentiment analysis, enable machines to understand and interpret the emotions conveyed in text data.",
    "Supervised learning requires labeled data, where the machine is trained on input-output pairs to predict the correct output for new inputs.",
    "The use of reinforcement learning in AI enables systems to learn optimal actions through trial and error to maximize a specific goal.",
    "Machine learning models can be used for various tasks, such as classification, regression, and clustering, depending on the nature of the data.",
    "AI-powered chatbots are becoming more sophisticated, using NLP to understand user queries and provide relevant responses.",
    "With advancements in NLP, AI systems are becoming more capable of translating languages, summarizing texts, and even generating human-like content.",
    "I go to shop"
]

query ="What is deep learning?"

def get_embedding(text):
    #tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    #son gizli katmani alalim
    last_hidden_state = outputs.last_hidden_state
    #metin temsili olustur
    embedding = last_hidden_state.mean(dim=1)
    
    return embedding.detach().numpy()

#belgeler ve sorgu icin embedding vektorleri al 
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)

#cosinus benzerligi ile belgeler arasi benzerlik hesaplama

similarities = cosine_similarity(query_embedding, doc_embeddings)

for i, score in enumerate(similarities[0]):
    print(f"Document{i+1} : {score}")

most_similar_index = similarities.argmax()
print(f"Most similar documents: {documents[most_similar_index]}")