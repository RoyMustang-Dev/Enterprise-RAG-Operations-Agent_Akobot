"""
Master Ingestion Pipeline Orchestrator

This script logically chains the loader, chunker, embedding generation, 
and database persistence operations into a single cohesive synchronous block.
"""
import os
import time
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import QdrantStore
from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the entire document ingestion chronological workflow.
    
    Execution Flow:
    1. Loader: Reads temporary file data from disk into system RAM.
    2. Chunker: Splits the massive text array into mathematically bound segments.
    3. Embedder: Converts the text segments into high-dimensional vector representations.
    4. Vector Store: Saves the vectors alongside their descriptive metadata dictionaries for RAG retrieval.
    """
    def __init__(self, tenant_id: str = None):
        """
        Initializes the pipeline with explicit EmbeddingModel and Database backend adapters.
        """
        self.embedder = EmbeddingModel()
        profile = HardwareProbe.get_profile()
        self.batch_size = int(os.getenv("INGEST_BATCH_SIZE", profile.get("embedding_batch_size", 32)))
        logger.info(f"[INGESTION] Batch size set to {self.batch_size} for embeddings.")
        self.tenant_id = tenant_id
        
        use_multi_tenant = os.getenv("QDRANT_MULTI_TENANT", "false").lower() == "true"
        if not use_multi_tenant and tenant_id:
            # Legacy dedicated collection mode
            self.vector_store = QdrantStore(collection_name=tenant_id)
        else:
            # Multi-tenant mode (or single global base mode) utilizes base enterprise_rag container
            self.vector_store = QdrantStore()

    def run_ingestion(self, file_paths: List[str], metadatas: List[Dict] = None, reset_db: bool = False, job_tracker: dict = None, mark_completed: bool = True) -> int:
        """
        Executes the exhaustive ingestion loop mapping provided files to the semantic vector index.
        
        Args:
            file_paths (List[str]): Array of OS paths pointing to raw files meant for ingestion.
            metadatas (List[Dict], optional): Explicit dictionaries mapping custom tags (e.g. source URL).
            reset_db (bool): If True, triggers a complete destructive wipe of the Vector Database.
            job_tracker (dict, optional): Mutable dictionary explicitly passed from FastAPI to track completion ratios.
            mark_completed (bool): Whether to instantly mark the job "completed" upon function return, disabled for streaming.
                             
        Returns:
            int: The total numerical metric of successfully embedded and persisted text chunks.
        """
        if not file_paths:
            return 0

        total_chunks = 0
        all_chunks = []
        all_metadatas = []
        
        # -------------------------------------------------------------------------
        # 0. Defensive Database Truncation
        # -------------------------------------------------------------------------
        if reset_db:
            msg = "Resetting active Knowledge Base index mappings..."
            print(msg)
            if job_tracker is not None:
                job_tracker.setdefault("logs", []).append(msg)
            self.vector_store.clear()

        msg = f"Starting pipeline ingestion loop for {len(file_paths)} system files..."
        logger.info(msg)
        if job_tracker is not None:
            job_tracker.setdefault("logs", []).append(msg)
        
        # -------------------------------------------------------------------------
        # 1. & 2. Loading and Chunking Sequences (Parallelized)
        # -------------------------------------------------------------------------
        def _process_file(idx: int, path: str):
            try:
                text = load_document(path)
                if not text:
                    return idx, [], []
                chunks = chunk_text(text)
                if not chunks:
                    return idx, [], []
                base_meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                source_name = os.path.basename(path)
                ext = os.path.splitext(source_name)[1].lower().lstrip(".")
                creation_year = str(datetime.utcnow().year)
                base_meta = {
                    **base_meta,
                    "source": source_name,
                    "file_path": path,
                    "ingested_at": time.time(),
                    "document_type": ext or "txt",
                    "source_domain": base_meta.get("source_domain", "uploaded_file"),
                    "author": base_meta.get("author", "unknown"),
                    "creation_year": str(base_meta.get("creation_year", creation_year)),
                }
                metas = [base_meta.copy() for _ in chunks]
                return idx, chunks, metas
            except Exception as e:
                logger.error(f"[INGESTION] Failed processing {path}: {e}")
                return idx, [], []

        cpu_cores = HardwareProbe.get_profile().get("cpu_cores", 4)
        max_workers = max(1, min(len(file_paths), max(2, cpu_cores // 2)))
        logger.info(f"[INGESTION] Parallel loader workers={max_workers}")
        if job_tracker is not None:
            job_tracker.setdefault("logs", []).append(f"Parallel loader workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, file_path in enumerate(file_paths):
                msg = f"Processing Data File: {file_path}"
                logger.info(msg)
                if job_tracker is not None:
                    job_tracker.setdefault("logs", []).append(msg)
                futures.append(executor.submit(_process_file, i, file_path))

            for future in as_completed(futures):
                _, chunks, metas = future.result()
                if not chunks:
                    continue
                all_chunks.extend(chunks)
                all_metadatas.extend(metas)
                total_chunks += len(chunks)

        if not all_chunks:
            logger.warning("No valid data chunks generated to actively ingest.")
            return 0

        # -------------------------------------------------------------------------
        # 3. & 4. Batch Embedding and Persistence (Generator Pattern)
        # -------------------------------------------------------------------------
        batch_size = self.batch_size
        total_processed = 0
        total_to_process = len(all_chunks)
        
        if job_tracker is not None:
            if "total_chunks" not in job_tracker:
                job_tracker["total_chunks"] = 0
            if "chunks_added" not in job_tracker:
                job_tracker["chunks_added"] = 0
            job_tracker["total_chunks"] += total_to_process
            
        msg = f"Executing mathematical embedding generations for {total_to_process} system chunks in batches of {batch_size}..."
        logger.info(msg)
        if job_tracker is not None:
            job_tracker.setdefault("logs", []).append(msg)
        
        for i in range(0, total_to_process, batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metas = all_metadatas[i:i + batch_size]
            
            try:
                # Batch call the BAAI engine
                embeddings = self.embedder.generate_embeddings(batch_chunks)
                if not embeddings:
                    raise ValueError("Embedding model returned an explicitly empty array for chunks.")
                    
                # Push normalized vectors to permanent vector store backend with Exponential Backoff
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(batch_chunks, embeddings, batch_metas, tenant_id=self.tenant_id)
                        break
                    except Exception as connection_err:
                        if attempt == max_retries - 1:
                            raise connection_err
                        retry_msg = f"Network connection dropped (Attempt {attempt+1}/{max_retries}). Retrying in {2 ** attempt}s..."
                        logger.warning(retry_msg)
                        if job_tracker is not None:
                            job_tracker.setdefault("logs", []).append(retry_msg)
                        time.sleep(2 ** attempt)
                        
                total_processed += len(batch_chunks)
                
                # Update tracker so frontend sees real-time progress
                if job_tracker is not None:
                    job_tracker["chunks_added"] += len(batch_chunks)
                    
                # Explicitly flush RAM allocations incrementally to prevent out-of-memory crashes
                import gc
                del embeddings
                del batch_chunks
                del batch_metas
                gc.collect()
                
            except Exception as e:
                msg = f"Batch generation pipeline failed catastrophically at index {i}: {e}"
                logger.error(msg)
                if job_tracker is not None:
                    job_tracker.setdefault("logs", []).append(msg)
                    job_tracker["status"] = "failed"
                    job_tracker["error"] = str(e)
                raise e

        # Mark discrete job completely successful
        if job_tracker is not None and mark_completed:
            job_tracker["status"] = "completed"
            
        msg = f"Orchestration Ingestion loop complete. Successfully persisted {total_processed} exact chunks."
        logger.info(msg)
        if job_tracker is not None:
            job_tracker.setdefault("logs", []).append(msg)
        return total_processed
