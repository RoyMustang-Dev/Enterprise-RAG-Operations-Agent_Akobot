"""
Internal LLM Provider Wrapper (Sarvam AI API)

This module handles the physical HTTP POST logic required to stream tokens 
from external GenAI Providers (primarily Sarvam-M or OpenAI paradigms).
Abstracting raw `requests` protects upstream agent code from handling JSON Decode exceptions.
"""
import requests
import json
import sys
import os
from typing import Dict, Any, Generator
from dotenv import load_dotenv

# Ensure environment parsing loads instantly
load_dotenv()

# Boilerplate safety to ensure relative imports do not crash when testing natively
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class SarvamClient:
    """
    Client wrapper for interacting with Sarvam AI API using generic completion logic.
    Supports robust Server-Sent Events (SSE) streaming necessary for the Web UI.
    Supported core model: `sarvam-m`.
    """
    def __init__(self, api_key: str = None, model: str = "sarvam-m"):
        """
        Initializes the low-level provider.
        
        Args:
            api_key (str, optional): Key override, otherwise fallback to OS Env.
            model (str): Engine slug. Defaults to standard Sarvam-M.
        """
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.model = model
        self.base_url = "https://api.sarvam.ai/v1/chat/completions" 
        
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = None, reasoning_effort: str = None, stream: bool = False) -> Any:
        """
        Executes a completion generation request to the API.
        
        Args:
            prompt (str): The User text.
            system_prompt (str, optional): The Agent Persona constraints.
            temperature (float, optional): Algorithmic creativity metric (0.0 to 1.0).
            reasoning_effort (str, optional): High/Low compute flags specifically for logic tests.
            stream (bool): If True, yields string chunks asynchronously. If False, blocks until full string.
            
        Returns:
            Generator[str] OR str: The parsed text response.
        """
        # 1. Credential Gatekeeping
        if not self.api_key or self.api_key.strip() == "":
             return "Error: Sarvam API Key is missing. Please enter your API key in the sidebar."
             
        # 2. Compile Subscription Headers
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # 3. Message Serialization Array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 4. Standard Payload Construction
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048, # Increased to prevent Enterprise Document sequence truncation
            "stream": stream
        }
        
        if temperature is not None:
             payload["temperature"] = temperature
        else:
             payload["temperature"] = 0.2
             
        # 5. Vendor Bug Mitigation
        # Current Sarvam SaaS layer possesses a bug where assigning `reasoning_effort` while 
        # `stream=True` causes the pipe to yield empty None payloads. We deliberately intercept this here.
        if reasoning_effort is not None and not stream:
             payload["reasoning_effort"] = reasoning_effort
             
        # 6. Execution Call & Stream Decoding
        try:
            # Dispatch the HTTPS POST socket
            response = requests.post(self.base_url, headers=headers, json=payload, stream=stream)
            
            # Catch Explicit Unauthorized Errors
            if response.status_code == 401:
                return "Error: Invalid Sarvam API Key. Please check your credentials."
            response.raise_for_status()
            
            # --- Code Block for Yielding Asynchronous Streams (Typewriter Effect) ---
            if stream:
                def generate_stream():
                    for line in response.iter_lines():
                        if line:
                            line_dec = line.decode('utf-8').strip()
                            if line_dec.startswith('data: '):
                                data_str = line_dec[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                        delta = chunk_data["choices"][0].get("delta", {})
                                        content = delta.get("content")
                                        if content is not None:
                                            yield str(content)
                                except json.JSONDecodeError:
                                    pass
                            elif line_dec.startswith('{'):
                                # Fallback Decoder: if API miraculously ignored 'stream=True'
                                try:
                                    json_data = json.loads(line_dec)
                                    if "choices" in json_data and len(json_data["choices"]) > 0:
                                        msg = json_data["choices"][0].get("message", {})
                                        if "content" in msg and msg["content"]:
                                            yield msg["content"]
                                except json.JSONDecodeError:
                                    pass
                return generate_stream()
                
            # --- Code Block for Synchronous Returns ---
            else:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "")
                return "Error: Unexpected response format from Sarvam API."
                
        # Catch network, DNS, and TLS mapping errors directly
        except requests.exceptions.RequestException as e:
            print(f"Error calling Sarvam API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print("Response text:", e.response.text)
                err_msg = f"Error: Sarvam API Failed ({e.response.status_code}): {e.response.text}"
                return [err_msg] if stream else err_msg
            err_msg = f"Error: Failed to generate response from Sarvam ({self.model}). Check backend logs."
            return [err_msg] if stream else err_msg
