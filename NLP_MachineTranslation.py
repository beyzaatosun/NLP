from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Machine learning algorithms are capable of analyzing large datasets to uncover patterns and make predictions without being explicitly programmed.",

translated_text = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

translated_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)
print(f"Translated_text: {translated_text}")

