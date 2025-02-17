from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import warnings
warnings.filterwarnings("ignore")

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


question = "What is the capital of France"
context = "France, offiicially the French Republic, is a country whose capital is Paris"

def generate_answer(context, question):
    input_text = f"Question: {question}, Context: {context}. Please answer the question"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    #model calistirma
    with torch.no_grad():
        outputs = model.generate(inputs, max_length = 500)
        
    #uretilen yaniti decode etme
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #yanitlari ayiklama
    answer = answer.split("Answer:")[-1].strip()
    
    return answer


answer = generate_answer(context, question)

print(f"Answer: {answer}")
        