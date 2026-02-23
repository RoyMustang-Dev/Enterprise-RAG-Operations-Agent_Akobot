"""
Multi-Format Document Loader Module

Provides dedicated extraction functions for parsing complex enterprise filetypes 
(PDF, DOCX, TXT) into raw, contiguous strings ready for the chunking pipeline.
"""
import fitz  # PyMuPDF
import docx
import os

def load_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file utilizing the PyMuPDF (fitz) library.
    
    [DESIGN DECISION]: PyMuPDF is selected over PyPDF2 due to its 10x faster 
    extraction speed and superior handling of complex multi-column enterprise layouts.
    
    Args:
        file_path (str): Absolute or relative OS path to the `.pdf` file.
        
    Returns:
        str: The extracted text content concatenated sequentially from all pages.
    """
    text = ""
    try:
        # Open the document safely leveraging context managers memory limits
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def load_docx(file_path: str) -> str:
    """
    Extracts structured text from a Microsoft Word DOCX file using python-docx.
    
    Iterates sequentially through all paragraph blocks defined in the document's XML structure.
    
    Args:
        file_path (str): Absolute path to the `.docx` file.
        
    Returns:
        str: The extracted text content, with logical paragraphs separated by precise newlines.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def load_text(file_path: str) -> str:
    """
    Extracts raw text from a plain TXT file (useful for crawled web output).
    
    Args:
        file_path (str): Absolute path to the `.txt` file.
        
    Returns:
        str: The unformatted raw text content.
    """
    text = ""
    try:
        # Enforce UTF-8 to prevent byte decoding crashes on scraped HTML artifact characters
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
    return text

def load_document(file_path: str) -> str:
    """
    Master Dispatcher function to route document loads based on file extension.
    
    Centralizes the try/catch logic and format discovery phase for the Ingestion Pipeline.
    Supports: `.pdf`, `.docx`, `.txt`
    
    Args:
        file_path (str): The absolute path pointing to the locally saved temporary file.
        
    Returns:
        str: Extracted text content converted to a unified string representation.
        
    Raises:
        ValueError: If the file extension extracted from the path is not currently supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported explicit enterprise file type: {ext}")
