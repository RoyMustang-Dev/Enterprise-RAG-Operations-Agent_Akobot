"""
Master Ingestion Pipeline Orchestrator

This script logically chains the loader, chunker, embedding generation, 
and database persistence operations into a single cohesive synchronous block.
"""
import os
import time
from typing import List, Dict

from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_text
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore

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
        self.vector_store = FAISSStore()

    def run_ingestion(self, file_paths: List[str], metadatas: List[Dict] = None, reset_db: bool = False) -> int:
        """
        Executes the exhaustive ingestion loop mapping provided files to the semantic vector index.
        
        Args:
            file_paths (List[str]): Array of OS paths pointing to raw files meant for ingestion.
            metadatas (List[Dict], optional): Explicit dictionaries mapping custom tags (e.g. source URL).
            reset_db (bool): If True, triggers a complete destructive wipe of the Vector Database.
                             
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
                base_meta.update({
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "ingested_at": time.time()
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
        # 3. Synchronous Embedding Generation
        # -------------------------------------------------------------------------
        print(f"Executing mathematical embedding generations array for {len(all_chunks)} system chunks...")
        embeddings = []
        try:
            # Batch call the BAAI engine
            embeddings = self.embedder.generate_embeddings(all_chunks)
            if not embeddings:
                # Catch empty tensor bugs explicitly
                if len(all_chunks) > 0:
                     raise ValueError("Embedding model returned an explicitly empty array for chunks.")
        except Exception as e:
            print(f"Mass Embedding generation failed catastrophically: {e}")
            raise e
        
        # -------------------------------------------------------------------------
        # 4. Persistence Execution
        # -------------------------------------------------------------------------
        print("Pushing normalized vectors to permanent vector store backend...")
        try:
            self.vector_store.add_documents(all_chunks, embeddings, all_metadatas)
        except Exception as e:
            print(f"Vector store native database addition failed explicitly: {e}")
            raise e
        
        print(f"Orchestration Ingestion loop complete. Successfully persisted {len(all_chunks)} exact chunks.")
        return len(all_chunks)
