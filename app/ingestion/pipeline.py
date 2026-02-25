"""
Master Ingestion Pipeline Orchestrator

This script logically chains the loader, chunker, embedding generation, 
and database persistence operations into a single cohesive synchronous block.
"""
import os
import time
from datetime import datetime
from typing import List, Dict

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import QdrantStore

class IngestionPipeline:
    """
    Orchestrates the entire document ingestion chronological workflow.
    
    Execution Flow:
    1. Loader: Reads temporary file data from disk into system RAM.
    2. Chunker: Splits the massive text array into mathematically bound segments.
    3. Embedder: Converts the text segments into high-dimensional vector representations.
    4. Vector Store: Saves the vectors alongside their descriptive metadata dictionaries for RAG retrieval.
    """
    def __init__(self):
        """
        Initializes the pipeline with explicit EmbeddingModel and Database backend adapters.
        """
        self.embedder = EmbeddingModel()
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
            print("Resetting active Knowledge Base index mappings...")
            self.vector_store.clear()

        print(f"Starting pipeline ingestion loop for {len(file_paths)} system files...")
        
        # -------------------------------------------------------------------------
        # 1. & 2. Loading and Chunking Sequences Loop
        # -------------------------------------------------------------------------
        for i, file_path in enumerate(file_paths):
            try:
                print(f"Processing Data File: {file_path}")
                # Dispatch mapping load based on file extension
                text = load_document(file_path)
                if not text:
                    print(f"Warning Sequence: No text successfully extracted natively from {file_path}")
                    continue
                    
                # Subdivide massive string payload into mathematical chunks
                chunks = chunk_text(text)
                if not chunks:
                    continue
                
                # Append default tracking metadata required by the Retrieval Tool filtering logic
                base_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                source_name = os.path.basename(file_path)
                ext = os.path.splitext(source_name)[1].lower().lstrip(".")
                creation_year = str(datetime.utcnow().year)
                base_meta.update({
                    "source": source_name,
                    "file_path": file_path,
                    "ingested_at": time.time(),
                    "document_type": ext or "txt",
                    "source_domain": base_meta.get("source_domain", "uploaded_file"),
                    "author": base_meta.get("author", "unknown"),
                    "creation_year": str(base_meta.get("creation_year", creation_year)),
                })
                
                # Flatten the data struct
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append(base_meta.copy())
                    
                total_chunks += len(chunks)
                
            except Exception as e:
                print(f"Error Pipeline extraction processing {file_path}: {e}")

        if not all_chunks:
            print("No valid data chunks generated to actively ingest.")
            return 0

        # -------------------------------------------------------------------------
        # 3. & 4. Batch Embedding and Persistence (Generator Pattern)
        # -------------------------------------------------------------------------
        batch_size = 100
        total_processed = 0
        total_to_process = len(all_chunks)
        
        if job_tracker is not None:
            if "total_chunks" not in job_tracker:
                job_tracker["total_chunks"] = 0
            if "chunks_added" not in job_tracker:
                job_tracker["chunks_added"] = 0
            job_tracker["total_chunks"] += total_to_process
            
        print(f"Executing mathematical embedding generations for {total_to_process} system chunks in batches of {batch_size}...")
        
        for i in range(0, total_to_process, batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metas = all_metadatas[i:i + batch_size]
            
            try:
                # Batch call the BAAI engine
                embeddings = self.embedder.generate_embeddings(batch_chunks)
                if not embeddings:
                    raise ValueError("Embedding model returned an explicitly empty array for chunks.")
                    
                # Push normalized vectors to permanent vector store backend
                self.vector_store.add_documents(batch_chunks, embeddings, batch_metas)
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
                print(f"Batch generation pipeline failed catastrophically at index {i}: {e}")
                if job_tracker is not None:
                    job_tracker["status"] = "failed"
                    job_tracker["error"] = str(e)
                raise e

        # Mark discrete job completely successful
        if job_tracker is not None and mark_completed:
            job_tracker["status"] = "completed"
            
        print(f"Orchestration Ingestion loop complete. Successfully persisted {total_processed} exact chunks.")
        return total_processed
