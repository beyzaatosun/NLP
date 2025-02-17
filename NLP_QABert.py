from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

tokenizer = BertTokenizer.from_pretrained(model_name)

#ince ayar yapilmis bert modeli
model = BertForQuestionAnswering.from_pretrained(model_name)

def predict_answer(context, question):
    #metni ve soruyu tokenlara ayirma
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)#pt=pytorch
    #giris tensorleri
    input_ids = encoding["input_ids"] #token id
    attention_mask = encoding["attention_mask"] #hangi tokenlar dikkate alinacagi
    #model calisir skor hesaplama
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask = attention_mask, return_dict=False)
    #en yuksek olasiliga sahip start ve end indeksleri
    start_index = torch.argmax(start_scores, dim=1).item() #baslangic index
    end_index = torch.argmax(end_scores, dim=1).item() #bitis index
    
    #token id kullnarak cevap metni 
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])
    
    #tokenlari birlestir okunabilir hale getir
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer

question = "What is the capital of France"
context = "France, offiicially the French Republic, is a country whose capital is Paris"
answer = predict_answer(context, question)

print(f"Answer: {answer}")