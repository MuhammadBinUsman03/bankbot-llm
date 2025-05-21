import requests

CHAT_API_URL = "http://localhost:8080/api/v1/rag/answer"
UPLOAD_API_URL = "http://localhost:8080/api/v1/vectordb/load"

def get_llm_response(prompt, collection_name):
    try:
        query = {
            "text": prompt
        }

        params = {
            "collection_name": collection_name,
            "top_k": 3
        }

        response = requests.post(CHAT_API_URL, json=query, params=params)
        result = response.json()

        return result['answer']
    except requests.exceptions.RequestException as e:
        return f"❌ Chat API error: {str(e)}"

def upload_document(file, collection_name):
    try:
        files = {"file": file}
        data = {"collection_name": collection_name}

        response = requests.post(UPLOAD_API_URL, files=files, data=data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"❌ Upload API error: {str(e)}"
