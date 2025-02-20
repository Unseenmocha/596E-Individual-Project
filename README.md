# Named Entity Recognition (NER) Project

## Setup Instructions

Follow these steps to set up the environment and run the project.

### 1. Clone the Repository
Clone the repo using:
```bash
git clone git@github.com:Unseenmocha/596E-Individual-Project.git
cd 596E-Individual-Project
```

### 2. Install Dependencies

#### a) Create and Activate Conda Environment
```bash
conda create --name ner_env python=3.12.9 -y
conda activate ner_env
```

#### b) Install Conda Dependencies
Install the required dependencies
```bash
conda install -c conda-forge transformers==4.46.1 pytorch onnxruntime onnx black
```

#### c) Install FlaskML 
Install FlaskML
```bash
pip install flask-ml 
```

### 3. Download Pretrained Model Files
Download the original model files from [BERT-NER GitHub](https://github.com/kamalkraj/BERT-NER?tab=readme-ov-file) and place the file `pytorch_model.bin` out_base directory. The structure should look like this:
```
/project-directory
    ├── NER.py
    ├── NER_cli.py
    ├── NER_onnx.py
    ├── server.py
    ├── requirements.txt
    ├── out_base/
        ├── added_tokens.json
        ├── config.json
        ├── eval_results.txt
        ├── model_config.json
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── vocab.txt
        ├── pytorch_model.bin <-- added file

```

### 4. Generate ONNX Model
Run the following command to create the ONNX model:
```bash
python NER.py
```
Make sure `NER.onnx` is present in the project directory. The model is loaded in using Huggingface transformers AutoModelForTokenClassification.from_pretrained, and is exported to onnx using torch.onnx.export the code for exporting is in `NER.onnx`

### 5. Run Command Line Interface
To use the command-line interface for NER, run:
```bash
python NER_cli.py --input_text "input text"
```

### 6. Start the Server
Launch the server using:
```bash
python server.py
```

The server should now be running and ready to process requests.

### 7. Connect via RescueBox
- Open rescuebox desktop
- Register a Model with the IP and port the server is running on (defaults 127.0.0.1:5000)
- Click run on the NER job
- enter the input text to perform NER on
- click Run Model
- view the results from the Results tab on the left
---