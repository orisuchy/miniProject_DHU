from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "./stanza-he/models/pos"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

results = classifier(["mama just killed a man", "put a gun against his head", "pull my trigger now he is dead", "happy at home"])
# results = classifier(["איזה יום שמח לי היום"])

for res in results:
    print(res)

tokens = tokenizer.tokenize("mama just killed a man")
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("mama just killed a man")

print(f'\nTokens: {tokens}')
print(f'Token IDs: {tokens_ids}')
print(f'Input IDs: {input_ids}')


X_train = ["mama just killed a man", "put a gun against his head", "happy at home"]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors='pt')
print(f'\nbatch:\n{batch}')

with torch.no_grad():
    outputs = model(**batch) # the ** unpack the dictionary
    print(f'\noutputs:\n{outputs}')
    predictions = F.softmax(outputs.logits, dim=1)
    print(f'predictions:\n{predictions}')
    labels = torch.argmax(predictions, dim=1)
    print(f'labels IDs:\n{labels}')
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(f'labels:\n{labels}')
save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

