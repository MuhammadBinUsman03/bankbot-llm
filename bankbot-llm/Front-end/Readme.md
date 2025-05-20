# 🧠 LLM Banking Assistant – Frontend (Streamlit)

This is the **frontend interface** of a smart customer-service assistant for a banking system. The assistant leverages a **LoRA-fine-tuned Llama 3.2-3B-Instruct model** for answering FAQs and processing document uploads using Retrieval-Augmented Generation (RAG).

The frontend is built with **Streamlit**, providing a lightweight interface for end users to:

- 💬 Chat with the assistant  
- 📄 Upload policy documents for updates  
- 🧹 Clear conversations  

> ⚠️ Currently, this app uses **dummy API endpoints** for both chat and document upload. You can plug in your real endpoints later.

---

## 🚀 Features

- **Chat Interface**: Input queries and get simulated LLM-based responses
- **Document Upload**: Upload updated policies (PDFs) to simulate RAG integration
- **Clear Conversation**: Resets the chat session
- **Error Handling**: Graceful handling of API errors and validation issues

---

## 🧩 Tech Stack

- **Frontend Framework**: Streamlit
- **API Calls**: `requests`
- **Env Config**: `python-dotenv`

---

## 🛠️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/Muhammad-AB/bankbot-llm
   cd bankbot-llm/Front-end
2. **Create a Virtual Environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py

---

## 📁 Folder Structure
llm-banking-frontend/

│

├── app.py              # Main Streamlit entry point

├── components.py       # UI components: chat UI, upload UI, layout

├── utils.py            # Helper functions: API calls, state management

├── requirements.txt    # Python dependencies

└── README.md           # You’re here!
