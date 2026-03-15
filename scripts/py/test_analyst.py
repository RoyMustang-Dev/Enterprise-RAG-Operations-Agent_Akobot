import requests
import json
import uuid

URL = "http://localhost:8000/api/v1/business_analyst/chat"
FILE_PATH = r"D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\data\uploads\marketing_ecommerce_benchmark.csv"

# First, construct the multipart form data payload
session_id = str(uuid.uuid4())
query = "What is the total revenue over this time period, and what is the trend of the conversion rate?"

with open(FILE_PATH, "rb") as f:
    files = {"files": ("marketing_ecommerce_benchmark.csv", f, "text/csv")}
    data = {"query": query, "session_id": session_id}
    headers = {"x-tenant-id": "default"}
    
    print(f"Testing the Business Analyst Endpoint on Session: {session_id}")
    try:
        response = requests.post(URL, data=data, files=files, headers=headers)
        response.raise_for_status()
        
        print("\n--- ANALYTICS DASHBOARD RESPONSE (200 OK) ---")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n--- API FAILURE ---")
        print(f"Error: {e}")
        if response is not None:
            print("Response Body:", response.text)
