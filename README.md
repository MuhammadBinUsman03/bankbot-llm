# 🏦 BankBot-LLM

**BankBot-LLM** is an end-to-end pipeline for building a banking-focused language model. The project encompasses dataset preparation, fine-tuning, and inference stages, aiming to streamline domain-specific conversational AI in the finance sector.

---

## 📁 Project Structure

```

bankbot-llm/
├── ai_core/                        # Core backend logic (separate from bankbot-llm)
│   ├── api/                        # FastAPI app setup
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── routes.py
│   ├── database/                   # Vector database interface
│   │   ├── __init__.py
│   │   └── qdrant_client.py
│   ├── llm/                        # LLM interaction layer
│   │   ├── __init__.py
│   │   └── client.py
│   ├── models/                     # Embedding models
│   │   ├── __init__.py
│   │   └── embedding.py
│   ├── processors/                 # Data preprocessing utilities
│   │   ├── __init__.py
│   │   └── data_processor.py
│   └── rag/                        # Retrieval-Augmented Generation chain logic
│       ├── __init__.py
│       ├── main.py                 # Builds and executes the RAG chain
│       ├── run_server.py           # Starts the FastAPI server
│       ├── setup.py                # Configuration/setup script
│       └── .env.example            # Sample environment file
├── bankbot-llm/                    # Project root folder
│   ├── Dataset/
│   │   └── Instruction Tuning Data/
│   │       ├── BankProducts_FineTuning.json
│   │       ├── Finetuning_dataset_preparation.ipynb
│   │       └── finetune_data.json
│   ├── Fine-Tune/
│   │   └── Fine-Tuning.ipynb
│   ├── Inference/
│   │   └── Inference.ipynb
│   ├── frontend/                   # Frontend interface
│   │   ├── README.md
│   │   ├── app.py
│   │   ├── components.py
│   │   ├── requirements.txt
│   │   └── services.py
│   ├── Images/                     # Architecture diagram(s)
│   │   └── llm_arch_2.png
│   └── README.md                   # Project overview documentation

````

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

