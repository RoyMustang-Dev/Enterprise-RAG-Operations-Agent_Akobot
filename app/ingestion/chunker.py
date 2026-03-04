"""
Document Text Chunking Module

Responsible for slicing massive raw document strings into mathematically bounded payloads.
WARNING: This uses a legacy word-count splitter. Phase 8 will introduce 
Token-Aware `RecursiveCharacterTextSplitter` to align perfectly with the BAAI Embedding token ceilings.
"""
from typing import List, Dict, Optional
import threading

_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    return _TOKENIZER

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits massive unstructured text into smaller semantic chunks for vector embedding.
    
    This implementation utilizes LangChain's RecursiveCharacterTextSplitter paired with the BGE HuggingFace tokenizer
    to chunk perfectly by token limits.
    
    Args:
        text (str): The massive raw string extracted procedurally.
        chunk_size (int): The target boundary length of each sequence chunk in tokens.
        overlap (int): The contextual sliding window in tokens.
        
    Returns:
        List[str]: An array of discrete, vectorized-ready text segments stripped to defined bounding lengths.
    """
    if not text:
        return []
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        tokenizer = _get_tokenizer()
        
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return splitter.split_text(text)
    except Exception as e:
        print(f"Error during token-aware splitting: {e}")
        # Fallback to naive splitting if tokenizer fails
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += max(1, chunk_size - overlap)
        return chunks

