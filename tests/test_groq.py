import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GROQ_API_KEY')

headers = {
    'Authorization': f'Bearer {api_key}', 
    'Content-Type': 'application/json'
}

sys_prompt = '''SYSTEM: You are the enterprise-grade reasoning brain. Your responsibility is to synthesize final answers using ONLY the provided CONTEXT. Follow these rules EXACTLY. 
1) INPUTS available: 
   - USER_PROMPT: "<user text>" 
   - CONTEXT_CHUNKS: [ {text, source_meta...}, ... ] 

2) OUTPUT format (Output exactly one JSON object): 
   - "answer": <concise human-readable result, <= 400 words> 
   - "provenance": [ {"source_id":"", "quote":"<=40 words"} ... ] 
   - "confidence": 0.00-1.00 

3) RULES: 
   - Use only facts present in CONTEXT_CHUNKS. If you must hypothesize, clearly label as "hypothesis". 
   - For each factual claim, attach a provenance entry. 
   - If you cannot answer, respond: "I don't know based on the provided documents." 
   - Do NOT include chain-of-thought in the final visible output.'''

payload = {
    'model': 'llama-3.3-70b-versatile', 
    'messages': [
        {'role': 'system', 'content': sys_prompt}, 
        {'role': 'user', 'content': 'USER_PROMPT: Hello\n\nCONTEXT_CHUNKS:\n'}
    ], 
    'temperature': 0.1, 
    'max_tokens': 1024, 
    'response_format': {'type': 'json_object'}
}

res = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload)
print(res.status_code)
print(res.text)
