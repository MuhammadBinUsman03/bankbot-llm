import requests

CHAT_API_URL = "http://localhost:8000/chat"
UPLOAD_API_URL = "http://localhost:8000/upload"

def get_llm_response(prompt):
    try:
        response = requests.post(CHAT_API_URL, json={"prompt": prompt})
        response.raise_for_status()
        return response.json().get("response", "❗No response received.")
    except requests.exceptions.RequestException as e:
        return f"❌ Chat API error: {str(e)}"

def upload_document(file):
    try:
        files = {"file": (file.name, file.getvalue())}
        response = requests.post(UPLOAD_API_URL, files=files)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"❌ Upload API error: {str(e)}"
