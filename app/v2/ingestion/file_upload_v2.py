import os
import io
import logging
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

# Use existing Aeko capabilities
import pymupdf  # Replaced PyPDF2 with PyMuPDF for speed/accuracy
import docx
import pandas as pd
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)

class FileUploadServiceV2:
    """
    V2 File Ingestion Pipeline for PageIndex
    Extracts raw contiguous Markdown/Text from files and bypasses traditional 
    chunking/vector embeddings to instead pass the entire structure to the Tree Builder.
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id

    async def process_files(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        Receives FastAPI UploadFiles, extracts Markdown text, and returns a list
        of document payloads ready for PageIndex Tree Generation.
        """
        extracted_documents = []

        for f in files:
            filename = f.filename or "unknown"
            ext = os.path.splitext(filename)[1].lower()
            try:
                content_bytes = await f.read()
                text_content = ""

                if ext == ".pdf":
                    text_content = self._extract_pdf(content_bytes)
                elif ext in [".docx"]:
                    text_content = self._extract_docx(content_bytes)
                elif ext in [".md", ".txt"]:
                    text_content = self._extract_txt(content_bytes)
                elif ext in [".csv", ".tsv"]:
                    text_content = self._extract_csv(content_bytes, ext=ext)
                elif ext in [".xlsx"]:
                    text_content = self._extract_xlsx(content_bytes)
                else:
                    logger.warning(f"[V2 INGESTION] Skipping unsupported file format: {ext}")
                    continue

                if not text_content.strip():
                    logger.warning(f"[V2 INGESTION] Empty content extracted from {filename}")
                    continue

                extracted_documents.append({
                    "filename": filename,
                    "content": text_content,
                    "type": "file_upload"
                })
                logger.info(f"[V2 INGESTION] Successfully extracted text from {filename} ({len(text_content)} chars)")

            except Exception as e:
                logger.error(f"[V2 INGESTION] Failed to process file {filename}: {e}")

        return extracted_documents

    def _extract_pdf(self, content_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF (faster and better formatting than PyPDF2)."""
        doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        full_text = []
        for page in doc:
            full_text.append(page.get_text("text"))
        return "\n\n".join(full_text)

    def _extract_docx(self, content_bytes: bytes) -> str:
        """Extract text from DOCX."""
        doc = docx.Document(io.BytesIO(content_bytes))
        full_text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n\n".join(full_text)

    def _extract_txt(self, content_bytes: bytes) -> str:
        """Extract RAW text from TXT or MD files."""
        return content_bytes.decode("utf-8", errors="ignore")

    def _extract_csv(self, content_bytes: bytes, ext: str = ".csv") -> str:
        """Convert CSV/TSV to Markdown tables."""
        separator = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(io.BytesIO(content_bytes), sep=separator)
        return df.to_markdown(index=False)

    def _extract_xlsx(self, content_bytes: bytes) -> str:
        """Convert XLSX sheets to Markdown tables."""
        all_sheets = pd.read_excel(io.BytesIO(content_bytes), sheet_name=None)
        md_text = []
        for sheet_name, df in all_sheets.items():
            md_text.append(f"## Sheet: {sheet_name}\n")
            md_text.append(df.to_markdown(index=False))
            md_text.append("\n")
        return "\n".join(md_text)

