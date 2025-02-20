import onnxruntime as ort
import numpy as np
from NER_onnx import NER_Processing
import argparse
from pathlib import Path
import json
from pprint import pprint

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, required=True)
# parser.add_argument("--output_di", type=str, required=True)
args = parser.parse_args()
# input_dir = Path(args.input_dir)
# output_dir = Path(args.output_dir)
# output_dir.mkdir(parents=True, exist_ok=True)


NER = NER_Processing()
outputs = NER.predict(args.input_text)
pprint(outputs)
# with open(output_dir / "output.json", "w") as f:
#     json.dump(outputs, f)