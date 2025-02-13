from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model_dir = "./out_base"

tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = AutoModelForTokenClassification.from_pretrained(model_dir)

id2label = model.config.id2label

text = "George Washington was born on February 22, 1732 at Popes Creek in Westmoreland County, Virginia. He was the first of six children of Augustine and Mary Ball Washington."

inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

predictions = torch.argmax(logits, dim=-1)

predicted_labels = [id2label[idx.item()] for idx in predictions[0]]

words = tokenizer.convert_ids_to_tokens(input_ids[0])
word_predictions = []
current_word = ""
current_label = "O"

for token, label in zip(words, predicted_labels):
    if token.startswith("##"):
        current_word += token[2:]  # Merge subwords
    else:
        if current_word:  # Store the previous word and its label
            word_predictions.append((current_word, current_label))
        current_word = token
        current_label = label

# Append the last word
if current_word:
    word_predictions.append((current_word, current_label))

# Print final cleaned predictions
for word, label in word_predictions:
    print(f"{word}: {label}")
