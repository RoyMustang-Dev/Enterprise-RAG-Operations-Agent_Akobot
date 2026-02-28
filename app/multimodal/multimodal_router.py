"""
Multimodal Router

Routes file uploads through OCR or Vision, then executes an ephemeral RAG cycle.
"""
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional

from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.reranker import SemanticReranker
from app.reasoning.synthesis import SynthesisEngine
from app.reasoning.verifier import HallucinationVerifier
from app.reasoning.formatter import ResponseFormatter
from app.multimodal.file_parser import FileParser
from app.multimodal.session_vector import SessionVectorManager
from app.multimodal.vision import VisionModel

logger = logging.getLogger(__name__)


class MultimodalRouter:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.reranker = SemanticReranker()
        self.synthesis_engine = SynthesisEngine()
        self.verifier = HallucinationVerifier()
        self.formatter = ResponseFormatter()
        self.file_parser = FileParser()
        self.session_vectors = SessionVectorManager()
        self.vision = VisionModel()

    def _is_image(self, filename: str) -> bool:
        ext = os.path.splitext(filename or "")[1].lower()
        return ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

    def _question_prefers_vision(self, question: str) -> bool:
        if not question:
            return False
        q = question.lower()
        visual_keywords = [
            "describe", "scene", "objects", "colors", "what is happening",
            "what's in the image", "what is in the image", "layout", "visual"
        ]
        text_keywords = ["read", "text", "ocr", "words", "what does it say", "transcribe"]
        if any(k in q for k in text_keywords):
            return False
        return any(k in q for k in visual_keywords)

    def _choose_image_mode(self, explicit_mode: str, question: str, ocr_text: str) -> str:
        explicit = (explicit_mode or "auto").lower()
        if explicit in ["ocr", "vision"]:
            return explicit

        if not ocr_text or len(ocr_text.strip()) < 12:
            return "vision"

        if self._question_prefers_vision(question):
            return "vision"
        return "ocr"

    def ingest_files_for_session(
        self,
        question: str,
        files: list,
        session_id: Optional[str] = None,
        image_mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Ingests multiple uploaded files into a single ephemeral session collection.
        Returns the session_id and collection name to be used in unified /chat retrieval.
        """
        if not question:
            raise ValueError("Question is required.")
        if not files:
            raise ValueError("No files provided.")

        self.session_vectors.cleanup_expired()
        session_id, store = self.session_vectors.get_or_create(session_id)
        # Refresh session mapping timestamp
        try:
            from app.infra.database import upsert_session_collection
            upsert_session_collection(session_id, store.collection_name)
        except Exception:
            pass

        ingested = 0
        modes = []

        for filename, file_bytes in files:
            if not file_bytes:
                continue

            content = ""
            doc_type = "file"
            chosen_mode = "file_text"

            if self._is_image(filename):
                doc_type = "image"
                ocr_text = ""
                if image_mode in ["auto", "ocr"]:
                    try:
                        ocr_text = self.file_parser.parse_image_text(file_bytes)
                    except Exception as e:
                        if image_mode == "ocr":
                            raise
                        logger.warning(f"[MULTIMODAL] OCR failed, falling back to vision: {e}")

                chosen_mode = self._choose_image_mode(image_mode, question, ocr_text)
                if chosen_mode == "vision":
                    logger.info("[MULTIMODAL] Using vision model for semantic image understanding.")
                    vision_text = self.vision.answer(file_bytes, question=question)
                    if ocr_text:
                        content = f"{ocr_text}\n\nVISION_CONTEXT:\n{vision_text}"
                    else:
                        content = vision_text
                else:
                    content = ocr_text
            else:
                content = self.file_parser.parse(filename, file_bytes)

            content = content.strip() if content else ""
            if not content:
                continue

            chunks = chunk_text(content)
            if not chunks:
                continue

            embeddings = self.embedding_model.generate_embeddings(chunks)
            metadatas = [
                {"source": filename, "document_type": doc_type, "mode_used": chosen_mode}
                for _ in chunks
            ]
            store.add_documents(chunks, embeddings, metadatas)
            ingested += len(chunks)
            modes.append({"file": filename, "mode": chosen_mode})

        return {
            "session_id": session_id,
            "collection_name": store.collection_name,
            "chunks_added": ingested,
            "modes": modes,
        }

    def _handle_image(
        self,
        question: str,
        file_bytes: bytes,
        filename: str,
        session_id: Optional[str],
        image_mode: str,
    ) -> Dict[str, Any]:
        if not session_id:
            session_id = str(uuid.uuid4())
        ocr_text = ""
        if image_mode in ["auto", "ocr"]:
            try:
                ocr_text = self.file_parser.parse_image_text(file_bytes)
            except Exception as e:
                if image_mode == "ocr":
                    raise
                logger.warning(f"[MULTIMODAL] OCR failed, falling back to vision: {e}")

        chosen = self._choose_image_mode(image_mode, question, ocr_text)
        if chosen == "vision":
            logger.info("[MULTIMODAL] Using vision model for semantic image understanding.")
            answer = self.vision.answer(file_bytes, question=question)
            state = {
                "answer": answer,
                "sources": [{"source": filename, "score": 1.0, "text": answer[:240]}],
                "confidence": 0.7,
                "verifier_verdict": "UNVERIFIED",
                "is_hallucinated": False,
                "optimizations": {"mode_used": "vision"},
                "latency_optimizations": {},
            }
            formatted = self.formatter.construct_final_response(state)
            formatted["session_id"] = session_id
            return formatted

        logger.info("[MULTIMODAL] Using OCR text for RAG over image.")
        return self._rag_from_text(question, ocr_text, filename, session_id, mode_used="ocr")

    def _rag_from_text(
        self,
        question: str,
        content: str,
        source_name: str,
        session_id: Optional[str],
        mode_used: str,
    ) -> Dict[str, Any]:
        if not session_id:
            session_id = str(uuid.uuid4())
        if not content:
            state = {
                "answer": "DATA_NOT_FOUND",
                "sources": [],
                "confidence": 1.0,
                "verifier_verdict": "UNVERIFIED",
                "is_hallucinated": False,
                "optimizations": {"mode_used": mode_used},
                "latency_optimizations": {},
            }
            formatted = self.formatter.construct_final_response(state)
            formatted["session_id"] = session_id
            return formatted

        chunks = chunk_text(content)
        if not chunks:
            return self._rag_from_text(question, "", source_name, session_id, mode_used=mode_used)

        embeddings = self.embedding_model.generate_embeddings(chunks)
        metadatas = [{"source": source_name, "document_type": "file"} for _ in chunks]

        session_id, store = self.session_vectors.get_or_create(session_id)
        store.add_documents(chunks, embeddings, metadatas)

        retrieval_t0 = time.perf_counter()
        query_embedding = self.embedding_model.generate_embedding(question)
        retrieved = store.search(query_embedding, k=30)
        retrieval_t1 = time.perf_counter()

        if not retrieved:
            state = {
                "answer": "DATA_NOT_FOUND",
                "sources": [],
                "confidence": 1.0,
                "verifier_verdict": "UNVERIFIED",
                "is_hallucinated": False,
                "optimizations": {"mode_used": mode_used},
                "latency_optimizations": {"retrieval_time_ms": round((retrieval_t1 - retrieval_t0) * 1000, 3)},
            }
            formatted = self.formatter.construct_final_response(state)
            formatted["session_id"] = session_id
            return formatted

        rerank_t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, retrieved, top_k=5)
        rerank_t1 = time.perf_counter()

        llm_t0 = time.perf_counter()
        synth = self.synthesis_engine.synthesize(question, reranked)
        llm_t1 = time.perf_counter()

        verification = self.verifier.verify(synth.get("answer", ""), reranked)

        state = {
            "answer": synth.get("answer", ""),
            "sources": synth.get("provenance", []),
            "confidence": synth.get("confidence", 0.0),
            "verification_claims": verification.get("claims", []),
            "verifier_verdict": verification.get("overall_verdict", "UNVERIFIED"),
            "is_hallucinated": verification.get("is_hallucinated", False),
            "optimizations": {
                "tokens_input": synth.get("tokens_input", 0),
                "tokens_output": synth.get("tokens_output", 0),
                "temperature_used": synth.get("temperature_used", 0.0),
                "mode_used": mode_used,
            },
            "latency_optimizations": {
                "retrieval_time_ms": round((retrieval_t1 - retrieval_t0) * 1000, 3),
                "rerank_time_ms": round((rerank_t1 - rerank_t0) * 1000, 3),
                "llm_time_ms": round((llm_t1 - llm_t0) * 1000, 3),
            },
        }

        formatted = self.formatter.construct_final_response(state)
        formatted["session_id"] = session_id
        return formatted
