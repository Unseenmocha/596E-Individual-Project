# Named Entity Recognition (NER) Project

## Setup Instructions

Follow these steps to set up the environment and run the project.

### 1. Clone the Repository (if applicable)
If the project is hosted on a repository, clone it using:
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Install Dependencies

#### a) Create and Activate Conda Environment
```bash
conda create --name ner_env python=3.12.9 -y
conda activate ner_env
```

#### b) Install Conda Dependencies
Install the required dependencies from `requirements.txt`:
```bash
conda install --file requirements.txt
```

### 3. Download Pretrained Model Files
Download the original model files from [BERT-NER GitHub](https://github.com/kamalkraj/BERT-NER?tab=readme-ov-file) and place the extracted folder in the project directory. The structure should look like this:
```
/project-directory
    ├── NER.py
    ├── NER_cli.py
    ├── NER_onnx.py
    ├── server.py
    ├── requirements.txt
    ├── <downloaded model folder>/
```

### 4. Generate ONNX Model
Run the following command to create the ONNX model:
```bash
python NER.py
```
Make sure `NER_onnx.py` is present in the project directory.

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

---