import requests
import json
import sys
import os
from typing import Dict, Any, Generator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    Defaults to http://localhost:11434.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model
        self.api_generate = f"{base_url}/api/generate"

    def check_connection(self) -> bool:
        """Checks if the Ollama server is reachable."""
        try:
            response = requests.get(self.base_url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def generate(self, prompt: str, system_prompt: str = None, stream: bool = False) -> str:
        """
        Generates a completion for the given prompt using local Ollama.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False # Enforce no streaming for simplicity in v1
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(self.api_generate, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return f"Error: Failed to generate response from {self.model}."


class SarvamClient:
    """
    Client for interacting with Sarvam AI API using the OpenAI SDK proxy.
    Supported model(s): `sarvam-m`.
    """
    def __init__(self, api_key: str = None, model: str = "sarvam-m"):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.model = model
        self.base_url = "https://api.sarvam.ai/v1/chat/completions" 
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generates a completion using Sarvam AI.
        Requires an active API key.
        """
        if not self.api_key or self.api_key.strip() == "":
             return "Error: Sarvam API Key is missing. Please enter your API key in the sidebar."
             
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2, 
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            if response.status_code == 401:
                return "Error: Invalid Sarvam API Key. Please check your credentials."
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("message", {}).get("content", "")
            return "Error: Unexpected response format from Sarvam API."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Sarvam API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print("Response text:", e.response.text)
                return f"Error: Sarvam API Failed ({e.response.status_code}): {e.response.text}"
            return f"Error: Failed to generate response from Sarvam ({self.model}). Check backend logs."


if __name__ == "__main__":
    # Quick test Ollama
    client = OllamaClient()
    if client.check_connection():
        print(f"Connected to Ollama. Model: {client.model}")
        # print("Response:", client.generate("Why is the sky blue?", system_prompt="Answer briefly."))
    else:
        print("Could not connect to Ollama. Is it running?")
        
    # Quick test Sarvam Setup (needs mock key)
    # s_client = SarvamClient(api_key="mock_key")
    # print(s_client.generate("Hello"))
