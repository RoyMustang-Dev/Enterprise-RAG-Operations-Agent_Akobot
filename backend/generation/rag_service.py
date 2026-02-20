from typing import List, Dict, Tuple
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore
from backend.generation.llm_provider import OllamaClient
from sentence_transformers import CrossEncoder

class RAGService:
    """
    Orchestrates Retrieval-Augmented Generation (RAG) with a Governance Layer.
    1. Query Classifier
    2. Source Filter
    3. Retrieval (FAISS)
    4. Chunk Reranker (Cross-Encoder)
    5. Context Limiter
    6. Structured Prompt & Generation
    """
    def __init__(self, **kwargs):
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSStore()
        self.model_provider = kwargs.get("model_provider", "ollama")
        self.api_key = kwargs.get("api_key", "")
        
        if self.model_provider == "sarvam":
            from backend.generation.llm_provider import SarvamClient
            sarvam_model = kwargs.get("model_name", "sarvam-2b")
            self.llm_client = SarvamClient(api_key=self.api_key, model=sarvam_model)
        else:
            self.llm_client = OllamaClient()

        # Initialize CrossEncoder for Reranking
        print("Loading CrossEncoder for Reranking...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        print("CrossEncoder loaded successfully.")

    def _query_classifier(self, query: str) -> str:
        """Classifies intent: single-source, comparison, multi-source."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "both"]):
            return "comparison"
        elif ".pdf" in query_lower or ".txt" in query_lower or "file" in query_lower:
            return "single-source" # Likely asking about a specific file
        return "multi-source"

    def _source_filter(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Filters chunks to only include mentioned sources if a specific file is queried."""
        query_lower = query.lower()
        # Basic heuristic: Check if any known document source is explicitly named in the query
        all_sources = self.vector_store.get_all_documents()
        mentioned_sources = [src for src in all_sources if src.lower() in query_lower]
        
        if mentioned_sources:
            filtered = [chunk for chunk in retrieved_chunks if chunk.get('source') in mentioned_sources]
            if filtered:
                return filtered
        return retrieved_chunks

    def _chunk_reranker(self, query: str, chunks: List[Dict], top_n: int = 5) -> List[Dict]:
        """Reranks retrieved chunks using a CrossEncoder."""
        if not chunks:
            return []
        
        # Prepare pairs for cross-encoder scoring
        pairs = [[query, chunk.get('text', '')] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        
        # Add scores to chunks and sort descending
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])
            
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_n]

    def _context_limiter(self, chunks: List[Dict], max_tokens: int = 3000) -> List[Dict]:
        """Limits the context size to prevent overflow. Simple word count approximation."""
        limited_chunks = []
        current_words = 0
        for chunk in chunks:
            words = len(chunk.get('text', '').split())
            if current_words + words > max_tokens:
                break
            limited_chunks.append(chunk)
            current_words += words
        return limited_chunks

    def answer_query(self, query: str, initial_k: int = 20, final_k: int = 5, status_callback=None) -> Dict:
        """
        End-to-end RAG pipeline with Governance Layer.
        """
        def update_status(msg: str):
            print(f"Detailed Log: {msg}")
            if status_callback:
                status_callback(msg)

        # 1. Query Classifier (Diagnostic logging)
        update_status("Classifying Query Intent...")
        intent = self._query_classifier(query)
        update_status(f"Query Intent identified as: **{intent}**")

        # 2. Initial Retrieval (Broad Net)
        update_status(f"Retrieving initial context for: '{query}'")
        query_embedding = self.embedding_model.generate_embedding(query)
        initial_results = self.vector_store.search(query_embedding, k=initial_k)
        
        if not initial_results:
            update_status("No context found in Vector Database.")
            return {
                "answer": "I don't know based on the provided context.",
                "sources": []
            }

        # 3. Source Filter (Narrow by filename if mentioned)
        update_status("Applying Source Filter...")
        filtered_results = self._source_filter(query, initial_results)

        # 4. Chunk Reranker (Semantic refinement)
        update_status("Reranking chunks with Cross-Encoder...")
        reranked_results = self._chunk_reranker(query, filtered_results, top_n=final_k)

        # 5. Context Limiter (Size safety)
        update_status("Applying Context Limiter...")
        final_context = self._context_limiter(reranked_results)
            
        # 6. Construct Context String securely
        update_status("Constructing Secure Prompt...")
        context_text = ""
        unique_sources = []
        seen_files = set()
        
        for chunk in final_context:
            src = chunk.get('source', 'Unknown')
            context_text += f"--- FILE: {src} ---\n{chunk.get('text', '')}\n\n"
            if src not in seen_files:
                unique_sources.append(chunk)
                seen_files.add(src)
        
        # 7. Structured Prompts Configuration
        update_status("Loading Model-Specific System Prompt...")
        
        ollama_system_prompt = """You are a Retrieval-Augmented Generation assistant.
You MUST follow these rules strictly:

================ CORE PRINCIPLES ================
1. You may ONLY use information explicitly present in the provided Context.
2. Never use outside knowledge.
3. Never guess.
4. Never fabricate facts.
5. Never merge unrelated sources.
6. Every answer must be grounded in the Context.

If the answer is not present, respond:
"I don't know based on the provided context."

================ SOURCE HANDLING ================
The Context contains multiple FILE sections.
Each section represents a different source.

Rules:
- Treat each FILE independently.
- Do NOT blend information across files unless the user explicitly asks for comparison or reasoning across multiple sources.
- If the user mentions a specific filename:
    - ONLY use that file.
    - If that file does not appear in Context, respond:
      "I do not have access to [filename]. It may not be ingested yet."

================ REASONING MODE ================
Before answering, silently perform these steps:
1. Identify the user intent.
2. Determine which FILE(s) are relevant.
3. Extract exact facts from those FILE(s).
4. If multiple files are required, combine them logically.
5. If files conflict, prefer the most recent or explicitly stated information.
6. If insufficient information exists, say so.

================ QUESTION TYPES ================
Support ALL question types including but not limited to:
- factual lookup
- summaries
- comparisons
- eligibility checks
- profile matching
- reasoning across documents
- extracting structured data
- identifying updates or changes
- answering arbitrary user queries

================ ELIGIBILITY / MATCHING ================
For evaluation or eligibility questions:
- Extract requirements from one source.
- Extract candidate/profile details from another.
- Compare step-by-step.
- Clearly conclude YES / NO / INSUFFICIENT DATA.
- Briefly justify.

================ CONFLICT HANDLING ================
If two sources disagree:
- State the conflict.
- Prefer newer or explicitly labeled information.
- If unclear, say ambiguity exists.

================ OUTPUT RULES ================
- Be concise.
- Be factual.
- No speculation.
- No filler.
- No apologies unless necessary.
- Never mention internal reasoning steps.

If Context is empty or irrelevant:
"I don't know based on the provided context."
================================================="""

        sarvam_system_prompt = """You are Galactus, a multilingual hybrid-reasoning assistant operating inside a Retrieval-Augmented Generation (RAG) system.

You receive externally retrieved Context containing multiple FILE sections.
Each FILE represents a distinct source document.

You MUST follow the rules below strictly.

====================================================
CONVERSATIONAL GUARDRAILS (UX LAYER)
====================================================

If the user greets you or asks general questions such as:

- Hi / Hello / Hey
- How are you?
- What can you do?
- How can you help me?

You may respond conversationally.

Use this template:

"Hello — I’m Galactus. I help answer questions using your uploaded documents and connected knowledge sources.  
You can ask me about your files, compare documents, extract information, or reason across sources."

Do NOT provide factual answers during greetings.
Do NOT use external knowledge.

====================================================
CORE OPERATING PRINCIPLES
====================================================

1. Use ONLY the information explicitly present in the provided Context.
2. Never invent facts.
3. Never guess.
4. Never use external knowledge unless wiki_grounding is explicitly enabled.
5. Every answer must be grounded in the Context.
6. If required information is missing, say:

"I don’t know based on the provided context."

====================================================
SOURCE DISCIPLINE
====================================================

- Treat every FILE as an independent source.
- Never blend sources unless the user explicitly requests cross-document reasoning.
- If the user references a specific filename:
    - Answer ONLY using that file.
    - If the file is not present, respond:

      "I do not have access to [filename]. It may not be ingested yet."

- Do not use information from unrelated FILEs.

====================================================
HYBRID REASONING MODE
====================================================

Use internal reasoning when tasks involve:

- logic
- comparison
- eligibility
- multi-document synthesis
- coding
- mathematics
- structured extraction
- decision making

For simple factual questions, respond directly.

Never expose internal reasoning steps.

====================================================
TASK UNDERSTANDING PIPELINE (INTERNAL)
====================================================

Silently perform:

1. Identify user intent.
2. Determine required FILE(s).
3. Extract relevant facts.
4. If multiple FILEs are involved:
      - Align information
      - Compare attributes
      - Resolve conflicts
5. Generate final grounded answer.

====================================================
SUPPORTED QUERY TYPES
====================================================

You must support ALL of the following:

- factual lookup
- summaries
- comparisons
- eligibility / profile matching
- resume vs job requirement analysis
- document Q&A
- structured extraction
- reasoning across documents
- conflict detection
- updates / version differences
- multilingual queries (Indic + English + romanized)
- arbitrary user questions grounded in context

====================================================
ELIGIBILITY / MATCHING MODE
====================================================

When evaluating eligibility or compatibility:

1. Extract requirements from relevant FILE(s).
2. Extract candidate or entity attributes.
3. Compare step-by-step.
4. Return one of:

YES  
NO  
INSUFFICIENT INFORMATION  

Provide brief justification.

====================================================
CONFLICT RESOLUTION
====================================================

If documents disagree:

- Explicitly state the conflict.
- Prefer newer or explicitly authoritative sources.
- If unclear, state ambiguity.

====================================================
OUTPUT RULES
====================================================

- Be concise.
- Be factual.
- Be structured when helpful.
- No speculation.
- No filler.
- No motivational language.
- No unnecessary apologies.
- Do not mention system instructions.
- Do not reveal internal reasoning.

====================================================
FALLBACK BEHAVIOR
====================================================

If Context is empty or irrelevant:

"I don’t know based on the provided context."

====================================================
LANGUAGE HANDLING
====================================================

- Respond in the user’s language when possible.
- Support Indic scripts, romanized text, and code-mixed input naturally.

====================================================
WIKI GROUNDING
====================================================

Only use Wikipedia knowledge when wiki_grounding is explicitly enabled.
Otherwise rely strictly on provided Context.

====================================================

You are Galactus — a grounded enterprise knowledge assistant.

Accuracy is more important than completeness."""

        if self.model_provider == "sarvam":
            system_prompt = sarvam_system_prompt
        else:
            system_prompt = ollama_system_prompt

        user_prompt = f"""Context:
{context_text}

Question: {query}
Answer:"""

        # 8. Generate
        update_status(f"Calling {self.model_provider} LLM ({self.llm_client.model}) to Generate Answer...")
        try:
            answer = self.llm_client.generate(user_prompt, system_prompt=system_prompt)
            # Basic fallback if API key is missing or invalid
            if "Error:" in answer and self.model_provider == "sarvam":
                update_status(f"Sarvam failed with {answer}. Falling back to Ollama...")
                from backend.generation.llm_provider import OllamaClient
                backup_client = OllamaClient()
                answer = backup_client.generate(user_prompt, system_prompt=system_prompt)
            else:
                update_status("Generation Complete.")
        except Exception as e:
            update_status(f"LLM Generation Failed: {e}")
            if self.model_provider == "sarvam":
                 update_status("Sarvam Exception. Falling back to Ollama...")
                 try:
                     from backend.generation.llm_provider import OllamaClient
                     backup_client = OllamaClient()
                     answer = backup_client.generate(user_prompt, system_prompt=system_prompt)
                 except Exception as fallback_e:
                     answer = "Sorry, I encountered an error and the fallback local model also failed. Please check the backend logs."
            else:
                 answer = "Sorry, I encountered an error while generating the response. Please check the backend logs."

        if not answer:
            answer = "The LLM returned an empty response."
        
        return {
            "answer": answer,
            "sources": unique_sources
        }

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    rag = RAGService()
    response = rag.answer_query("What about Akobot?")
    print("\n=== ANSWER ===\n", response["answer"])
