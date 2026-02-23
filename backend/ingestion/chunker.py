"""
Document Text Chunking Module

Responsible for slicing massive raw document strings into mathematically bounded payloads.
WARNING: This uses a legacy word-count splitter. Phase 8 will introduce 
Token-Aware `RecursiveCharacterTextSplitter` to align perfectly with the BAAI Embedding token ceilings.
"""
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """
    Splits massive unstructured text into smaller semantic chunks for vector embedding.
    
    This implementation currently utilizes a rudimentary word-based array splitting strategy.
    
    [KNOWN ARCHITECTURAL LIMITATION]: Because 512 *words* typically instantiate as 700+ *tokens*, 
    and our underlying sentence-transformers Embedding Model (BAAI/bge-large-en-v1.5) enforces a strict runtime hard cutoff at 512 max tokens,
    this script inadvertently causes silent data sequence truncation at the tail-end of specific densely worded chunks.
    
    Args:
        text (str): The massive raw string extracted procedurally from PyMuPDF or headless Playwright.
        chunk_size (int): The target boundary length of each sequence chunk in explicit word counts.
        overlap (int): The contextual sliding window (number of words copied into the next chronological chunk to preserve logic strings across cuts).
        
    Returns:
        List[str]: An array of discrete, vectorized-ready text segments stripped to defined bounding lengths.
    """
    if not text:
        return []
        
    # Split the raw payload string strictly on whitespace boundaries turning it into an indexed array
    words = text.split()
    chunks = []
    
    start = 0
    # Slide the procedural index window horizontally across the massive word array
    while start < len(words):
        # Determine the definitive cutoff index map for the active chunk boundary
        end = min(start + chunk_size, len(words))
        
        # Splice the indexed sub-array logic and structurally concatenate it back into a readable long string formatting
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Terminate the iterative procedural loop early if the tail end boundary of the document string is definitively reached
        if end == len(words):
            break
            
        # Nudge the starting pointer integer forward chronologically, explicitly mathematically subtracting the specific algorithmic `overlap` count
        # to ensure the end textual phrasing of chunk A is naturally duplicated at the very start of chunk B
        start += (chunk_size - overlap)
        
    return chunks
