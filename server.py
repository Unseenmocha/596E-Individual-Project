import argparse
from typing import TypedDict
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    ResponseBody,
    TextInput,
    TextResponse,
    TaskSchema,
    InputSchema,
    InputType,
)

from NER_onnx import NER_Processing

import torch


# Configure UI Elements in RescueBox Desktop
def create_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="text_input",
        label="Input Text",
        input_type=InputType.TEXT,
        placeholder="Enter input text for NER",
        multiline=True,
    )

    return TaskSchema(inputs=[input_schema], parameters=[])


class NERInput(TypedDict):
    text_input: TextInput


class NERParameters(TypedDict):
    pass


# Create a server instance
server = MLServer(__name__)

server.add_app_metadata(
    name="Named Entity Recognition",
    author="Andrew Lin",
    version="0.1.0",
    info=load_file_as_string("info.md"),
)

NER = NER_Processing()


@server.route("/predict", task_schema_func=create_task_schema)
def give_prediction(inputs: NERInput, parameters: NERParameters) -> ResponseBody:
    print(inputs)
    output = NER.predict(inputs["text_input"].text)
    return ResponseBody(root=TextResponse(value=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()
    print(
        "CUDA is available." if torch.cuda.is_available() else "CUDA is not available."
    )
    server.run(port=args.port)
