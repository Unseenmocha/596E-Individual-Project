from NER_onnx import NER_Processing
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, required=True)
args = parser.parse_args()


NER = NER_Processing()
outputs = NER.predict(args.input_text)
pprint(outputs)
