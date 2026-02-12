from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Sample network traffic
samples = ["Suspicious TCP connection", "Normal HTTP request"]
inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")

# Predictions
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)  # 0 = normal, 1 = suspicious