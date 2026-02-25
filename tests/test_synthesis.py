import os
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG)

from app.reasoning.synthesis import SynthesisEngine

engine = SynthesisEngine()
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import QdrantStore
from app.retrieval.reranker import SemanticReranker

query = "What is the Enterprise Agentic RAG Architecture?"

embedder = EmbeddingModel()
qdrant = QdrantStore()
reranker = SemanticReranker()

tensor = embedder.generate_embedding(query)
raw_chunks = qdrant.search(tensor, k=5)
refined_chunks = reranker.rerank(query, raw_chunks, top_k=5)

print([c.get('page_content') for c in refined_chunks])

res = engine.synthesize(query, refined_chunks)
print(res)
