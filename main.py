from transformers import pipeline
from transformers import BertModel, BertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

# model_name = "vitvit/xlm-roberta-base-finetuned-heb_HebrewSentiment"
alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert = BertModel.from_pretrained('onlplab/alephbert-base')

classifier = pipeline('sentiment-analysis', model=alephbert, tokenizer=alephbert_tokenizer)
results = classifier(["אני כל כך עצוב לי וגשם על העיר", "איזה יום שמח לי היום"])

# if not finetuning - disable dropout
alephbert.eval()

for res in results:
    print(res)

tokens = alephbert_tokenizer.tokenize("איזה יום שמח לי היום")
tokens_ids = alephbert_tokenizer.convert_tokens_to_ids(tokens)
input_ids = alephbert_tokenizer("איזה יום שמח לי היום")

print(f'\nTokens: {tokens}')
print(f'Token IDs: {tokens_ids}')
print(f'Input IDs: {input_ids}')

