import json
import uuid
import requests

URL = "http://localhost:8000/api/v1/business_analyst/chat"
FILE_PATH = r"D:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\data\uploads\marketing_ecommerce_benchmark.csv"
QUERY = "identify the future trends of all the products based on the current data for every singlle region individually"

session_id = str(uuid.uuid4())

with open(FILE_PATH, "rb") as f:
    files = {"files": ("marketing_ecommerce_benchmark.csv", f, "text/csv")}
    data = {
        "query": QUERY,
        "session_id": session_id,
        "model_provider": "auto",
    }
    headers = {"x-tenant-id": "default"}

    print(f"Testing BA trend query. Session: {session_id}")
    response = requests.post(URL, data=data, files=files, headers=headers, timeout=180)
    response.raise_for_status()

    payload = response.json()
    print(json.dumps(payload, indent=2)[:4000])

    # Minimal sanity checks
    assert payload.get("agent") == "BUSINESS_ANALYST"
    data = payload.get("data", {})
    assert "summary_paragraph" in data
    assert "per_region_summaries" in data
    assert "forecast_table" in data
    print("Trend query test passed.")
