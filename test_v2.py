import requests
import json

try:
    res = requests.post("http://localhost:8000/api/v2/chat", json={"query": "Hello!"})
    print(res.status_code)
    print(res.text)
except Exception as e:
    print(e)
