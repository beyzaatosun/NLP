#TEXT GENERATION GPT-2

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)

text = "Afternoon "

#tokenization
inputs = tokenizer.encode(text, return_tensors="pt") #cıktı pytorch tensoru

outputs = model.generate(inputs, max_length = 55)

#modelin urettigi tokenları okunabilir hale getirme
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) #ozel tokenları(orn:cumle baslangic bitis) metinden cikart

#uretilen metni print etme
print(generated_text)

"""
Afternoon  (5:30pm)
I'm going to be back in the studio for a few hours. I'm going to be doing a lot of work on the new album. 
I'm going to be doing a lot of work on the new album.
"""

#%%TEXT GENERATION LLama

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "huggyLLama/Llama-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

text = "Afternoon "

#tokenization
inputs = tokenizer.encode(text, return_tensors="pt") #cıktı pytorch tensoru

outputs = model.generate(inputs.input_ids, max_length = 55)

#modelin urettigi tokenları okunabilir hale getirme
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) #ozel tokenları(orn:cumle baslangic bitis) metinden cikart

#uretilen metni print etme
print(generated_text)

"""

"""