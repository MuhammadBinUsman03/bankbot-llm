# 🏦 BankBot-LLM

**BankBot-LLM** is an end-to-end pipeline for building a banking-focused language model. The project encompasses dataset preparation, fine-tuning, and inference stages, aiming to streamline domain-specific conversational AI in the finance sector.

---

## 📁 Project Structure

```

bankbot-llm/
├── Dataset/
│   └── prepare\_dataset.ipynb       # Prepares dataset from JSON & Excel files
├── Fine-Tune/
│   └── Fine-Tuning.ipynb           # Fine-tunes the model using the prepared dataset
├── Inference/
│   └── Inference.ipynb             # Loads the fine-tuned model and performs inference
├── arch\_diagram.png                # Architecture diagram of the pipeline
└── README.md

````

---

## 📊 1. Dataset Preparation

- **Notebook:** `Dataset/prepare_dataset.ipynb`
- **Output:** Cleaned and structured dataset ready for training

---

## 🧠 2. Fine-Tuning

- **Notebook:** `Fine-Tune/Fine-Tuning.ipynb`
- **Model:** A pre-trained LLM (e.g., LLaMA, Falcon, or similar)
- **Process:**
  - Load the preprocessed dataset
  - Tokenize and batch the input
  - Fine-tune the model on domain-specific Q&A data

Compatible with Google Colab or any GPU-enabled environment.

---

## 🔍 3. Inference

- **Notebook:** `Inference/Inference.ipynb`
- **Description:**
  - Loads the fine-tuned model from Google Drive
  - Performs inference on user-defined queries
  - Returns model-generated answers

Supports both single-query testing and in batch
Compatible with Google Colab or any GPU-enabled environment.
---

## 📌 Architecture Overview
This Diagram contains the complete architecture of the project (including the future work)
<p align="center">
  <img src="bankbot-llm\Images\llm _arch_2.png" alt="Architecture Diagram" width="700"/>
</p>

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/MuhammadBinUsman03/bankbot-llm.git
cd bankbot-llm
````

## ✅ Requirements

* Python ≥ 3.12
* Jupyter or Google Colab (with GPU for fine-tuning or inference)
* `transformers`
* `datasets`
* `pandas`, `numpy`, `openpyxl`
* `torch` or `tensorflow` (based on model backend)

## SETUP
Pull the Local running Qdrant image
```bash
docker pull qdrant/qdrant
```
Start Qdrant container at Port `6333` and `6334`
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```
Start the FastAPI server at port `8080` for AI inference
```bash
cd ai_core
pip install -e .
python run_server.py --host 0.0.0.0 --port 8080 --qdrant-url http://localhost:6333 --reload
```
Start the Streamlit Frontend at port `8502`

```bash
cd ../bankbot-llm/Front-end
pip install -r requirements.txt
streamlit run app.py
```

