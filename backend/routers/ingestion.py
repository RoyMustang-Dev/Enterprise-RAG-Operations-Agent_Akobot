"""
Knowledge Ingestion API Router

Provides modular FastAPI endpoints for ingesting raw data into the Vector Database.
Supports multiple ingestion modalities including direct document uploads (PDF, DOCX) 
and asynchronous external web crawling.
"""
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import logging
import asyncio
from backend.ingestion.crawler_service import CrawlerService
from backend.ingestion.pipeline import IngestionPipeline
from urllib.parse import urlparse

# Initialize the router to namespace ingestion-specific REST operations
router = APIRouter(
    prefix="/api/v1/ingest",
    tags=["Knowledge Ingestion"],
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dependency Management
# -----------------------------------------------------------------------------
def get_pipeline() -> IngestionPipeline:
    """
    Factory function to instantiate the IngestionPipeline.
    Separated to support future dependency injection or mocking during tests.
    
    Returns:
        IngestionPipeline: Initialized instance handling chunking and vector processing.
    """
    return IngestionPipeline()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.post("/files")
async def ingest_document(
    files: List[UploadFile] = File(...),
    mode: str = Form("append") # Options: "append" (keep existing data) or "start_fresh" (wipe DB)
):
    """
    **Manual File Upload Ingestion**
    
    Accepts an array of multipart document files (e.g., .pdf, .docx).
    Saves them to temporary static storage, and executes the vector extraction pipeline.
    """
    try:
        filenames = [f.filename for f in files]
        logger.info(f"API Ingesting documents: {filenames} (Mode: {mode})")
        
        # Determine if the vector database should clear its existing contents before writing
        reset_db = True if mode == "start_fresh" else False
        pipeline = get_pipeline()
        
        # Track local filepaths and generated metadata for the vector chunker
        paths_to_process = []
        metadatas_to_process = []
        
        # Ensure the safe temporary upload directory exists
        os.makedirs("data/uploaded_docs", exist_ok=True)
        
        # Process multi-part uploads sequentially
        for uploaded_file in files:
            temp_path = os.path.join("data/uploaded_docs", uploaded_file.filename)
            content = await uploaded_file.read()
            
            # Persist byte chunks safely to disk
            with open(temp_path, "wb") as f:
                f.write(content)
                
            paths_to_process.append(temp_path)
            # Attach structural metadata dict determining document source types
            metadatas_to_process.append({"type": "file", "original_name": uploaded_file.filename})
            
        # Execute the heavy synchronous data extraction pipeline synchronously (currently blocks the event loop)
        num_chunks = pipeline.run_ingestion(paths_to_process, metadatas=metadatas_to_process, reset_db=reset_db)
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(filenames)} files.",
            "chunks_added": num_chunks,
            "mode_applied": mode
        }
    except Exception as e:
        logger.error(f"Ingestion Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process document upload.")

@router.post("/crawler")
async def trigger_crawler(
    url: str = Form(..., description="The target URL to crawl (e.g., https://example.com)"),
    max_depth: int = Form(1, description="Depth of recursion (1 = single page, >1 = follow links)"),
    mode: str = Form("append", description="Ingestion mode: 'append' adds to KB, 'start_fresh' clears the database first.")
):
    """
    **External Website Crawler Trigger**
    
    Spawns an advanced headless Playwright instance against the provided URL.
    Saves extracted structured text into the static 'crawled_docs' directory, 
    and then automatically pipes them into the semantic ingestion pipeline.
    """
    try:
        logger.info(f"API Triggering advanced crawler on {url} (Depth: {max_depth})")
            
        # 1. Directory Formatting
        # Parse the domain and path to create a clean, identifiable folder name (e.g., example.com_about)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.strip("/")
        folder_name = f"{domain}_{path}".replace("/", "_") if path else domain
        save_folder = os.path.join("data", "crawled_docs", folder_name)
        
        from fastapi.concurrency import run_in_threadpool
        import sys
        import asyncio
        
        # 2. Asynchronous Playwright Wrapping
        # Because we're executing an entirely separate event loop for the async crawler deep inside a FastAPI endpoint,
        # we must isolate it safely utilizing run_in_threadpool and explicit Windows Event loop rules.
        def _run_crawler_sync():
            # Specifically required on Windows to prevent NotImplementedError when asyncio manages subprocesses
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
            # Spin up an isolated loop specifically for Playwright
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            crawler = CrawlerService()
            try:
                # Block the thread pool string until playright completely finishes execution
                return loop.run_until_complete(
                    crawler.crawl_url(
                        url=url, 
                        save_folder=save_folder,
                        simulate=False,
                        recursive=(max_depth > 1),
                        max_depth=max_depth
                    )
                )
            finally:
                loop.close()
                
        # Fire the threadpool function without blocking FastAPI's main event loops
        result = await run_in_threadpool(_run_crawler_sync)
        
        # 3. Pipeline Ingestion
        reset_db = True if mode == "start_fresh" else False
        pipeline = get_pipeline()
        
        num_chunks = 0
        
        # If the crawl was successful and yielded valid extracted text files...
        if result.get("saved_files"):
            for filepath in result["saved_files"]:
                if filepath.endswith(".txt"):
                    # Step 4: Pass the raw text files to FAISS/Vector ingestion
                    num_chunks += pipeline.run_ingestion([filepath], metadatas=[{"type": "url", "source_url": url}], reset_db=reset_db)
                    
                    # Prevent subsequent files from a single crawl operation from wiping the DB they just wrote to
                    reset_db = False 

        return {
            "status": "success" if result["status"] == "success" else result["status"], 
            "message": f"Crawler dispatched for {url} successfully.",
            "pages_crawled": result.get("pages_crawled", 0),
            "duration_seconds": result.get("duration", 0),
            "chunks_added": num_chunks,
            "mode_applied": mode
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Crawler Error:\n{error_details}")
        # Send full trace back specifically for diagnostic debugging
        raise HTTPException(status_code=500, detail=f"Failed to dispatch crawler. Reason: {str(e)}\n\nTraceback: {error_details}")
