import onnx
import onnxruntime
import numpy as np
from transformers import AutoTokenizer
import json

model_dir = "./out_base"


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class NER_Processing:
    def __init__(self):
        self.model = onnx.load("NER.onnx")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.id2label = {
            0: "LABEL_0",
            1: "O",
            2: "B-MISC",
            3: "I-MISC",
            4: "B-PER",
            5: "I-PER",
            6: "B-ORG",
            7: "I-ORG",
            8: "B-LOC",
            9: "I-LOC",
            10: "[CLS]",
            11: "[SEP]",
        }

        self.session = onnxruntime.InferenceSession(
            "NER.onnx", providers=["CPUExecutionProvider"]
        )

    def preprocess(self, input):
        inputs = self.tokenizer(
            input, return_tensors="pt", truncation=True, is_split_into_words=False
        )
        inputs = {name: to_numpy(input) for name, input in inputs.items()}
        input_ids = inputs["input_ids"]
        return inputs, input_ids

    def postprocess(self, output, input_ids):
        # get class predictions
        predictions = np.argmax(output[0][0], axis=-1)

        # map label indices to NER entity labels
        predicted_labels = [self.id2label[idx.item()] for idx in predictions]

        # postprocessing convert tokens back into original words
        words = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        word_predictions = {}
        current_word = ""
        current_label = "O"

        for token, label in zip(words, predicted_labels):
            if token.startswith("##"):
                current_word += token[2:]  # Merge subwords
            else:
                if current_word and current_label not in [
                    "O",
                    "[SEP]",
                    "[CLS]",
                ]:  # Store the previous word and its label
                    word_predictions[current_word] = current_label
                current_word = token
                current_label = label

        # Append the last word
        if current_word and current_label not in ["O", "[SEP]", "[CLS]"]:
            word_predictions[current_word] = current_label

        return json.dumps(word_predictions)

    def predict(self, input):
        inputs, input_ids = self.preprocess(input)
        # compute ONNX Runtime output prediction
        output = self.session.run(
            ["output"],
            {"input_ids": input_ids, "attention_mask": inputs["attention_mask"]},
        )

        results = self.postprocess(output, input_ids)

        return results
