from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import torch.onnx

model_dir = "./out_base"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
id2label = model.config.id2label

text = "George Washington was born on February 22, 1732 at Popes Creek in Westmoreland County, Virginia. He was the first of six children of Augustine and Mary Ball Washington."

# preprocessing tokenize input
inputs = tokenizer(
    text, return_tensors="pt", truncation=True, is_split_into_words=False
)
input_ids = inputs["input_ids"]

# pass inputs through model
with torch.no_grad():
    outputs = model(**inputs)

# export model to onnx format
torch.onnx.export(
    model,  # model being run
    (
        inputs.input_ids,
        inputs.attention_mask,
    ),  # model input (or a tuple for multiple inputs)
    "NER.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=14,  # the ONNX version to export the model to
    input_names=["input_ids", "attention_mask"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "se_length"},  # variable length axes
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "output": {0: "batch_size", 1: "seq_length"},
    },
)

# get class predictions
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# map label indices to NER entity labels
predicted_labels = [id2label[idx.item()] for idx in predictions[0]]

# postprocessing convert tokens back into original words
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
