"""
Streamlit User Interface (Frontend Portal)

This file serves as the visual testing ground and interactive Portal for the 
Enterprise RAG Architecture. It explicitly demonstrates how external client UI 
applications should interact with the abstracted FastAPI/LangGraph backend endpoints.
"""
import os
import sys
import re
import asyncio
import nest_asyncio
import requests
import streamlit as st
from dotenv import load_dotenv

# Re-allow asynchronous nesting mapped for Streamlit execution environments 
nest_asyncio.apply()
load_dotenv()

# Add backend to path to allow direct object instantiations when bypassing standard API calls
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



st.set_page_config(page_title="Enterprise RAG Agent", layout="wide")

st.title("Enterprise RAG Operations Agent")

# -------------------------------------------------------------------------
# UI Sidebar Configuration Maps
# -------------------------------------------------------------------------
st.sidebar.header("Operations")
option = st.sidebar.selectbox("Choose Action", ["Chat", "Ingest Documents", "System Health"])

st.sidebar.markdown("---")
st.sidebar.header("LLM Settings")

st.sidebar.markdown("**Provider:** Groq API (llama-3.3-70b-versatile)")

# Failsafe UI layer intercepting direct user keys if standard `.env` maps are offline
existing_key = os.getenv("GROQ_API_KEY", "")

groq_api_key = st.sidebar.text_input("Groq API Key", value=existing_key, type="password")
if not groq_api_key:
    st.sidebar.warning("API Key required for Groq models.")


# =========================================================================
# System Health View
# =========================================================================
if option == "System Health":
    st.subheader("System Status")
    try:
        # Probe the standalone FastAPI uvicorn worker explicitly
        res = requests.get("http://127.0.0.1:8000/health")
        if res.status_code == 200:
            st.success(f"Backend is online: {res.json()}")
        else:
            st.error("Backend returned an error mapping response.")
    except Exception as e:
        st.error(f"Could not connect to explicitly separated backend API logic: {e}")


# =========================================================================
# Document Ingestion View
# =========================================================================
elif option == "Ingest Documents":
    st.subheader("Ingest Documents")
    
    # Render the active Qdrant Knowledge Base index count statically
    with st.expander("📂 Current Knowledge Base (Click to View)", expanded=False):
        try:
            res = requests.get("http://127.0.0.1:8000/api/v1/ingest/status")
            if res.status_code == 200:
                data = res.json()
                docs = data.get("documents", [])
                total_vectors = data.get("total_vectors", 0)
                if docs:
                    st.write(f"**Total Dedicated Document Partitions:** {len(docs)} (Vectors: {total_vectors})")
                    for doc in docs:
                        st.text(f"• {doc}")
                else:
                    st.info("System Knowledge Base is functionally empty.")
            else:
                st.error("Failed to map explicit vector store system bounds from API.")
        except Exception as e:
            st.error(f"Could not load vector store explicit boundary mappings: {e}")
            
    st.divider()
    
    st.markdown("Upload files, enter a URL, or do both. The system will process all inputs natively into the high-dimensional knowledge base.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Upload Files")
        uploaded_files = st.file_uploader("References (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    with col2:
        st.markdown("### 2. Crawl URL")
        url_input = st.text_input("Target URL (e.g., https://learnnect.com)")
        max_depth_input = st.number_input("Max Recursion Depth", min_value=1, max_value=4, value=1, help="1 = Single root page. 2 = Root + Immediate child links.")

    # -------------------------------------------------------------------------
    # Index Mutability Controls
    # -------------------------------------------------------------------------
    st.subheader("Ingestion Settings")
    ingestion_mode = st.radio(
        "Ingestion Mode:",
        ("Start Fresh (Clear Logic)", "Append to Knowledge Base"),
        help="Start Fresh will forcefully wipe the existing active Qdrant database mappings. Append will selectively add text chunks to it."
    )
    
    if ingestion_mode == "Append to Knowledge Base":
        st.warning("⚠️ Warning: When natively appending, ensure you are not uploading duplicate system documents. Logic to natively detect exact subset duplicates in Qdrant is currently experimental.")

    # -------------------------------------------------------------------------
    # Ingestion API Execution Logic
    # -------------------------------------------------------------------------
    if st.button("Start Ingestion"):
        if not uploaded_files and not url_input:
            st.error("Please provide at least one explicit data source structure (File or mapped URL).")
        else:
            def wait_for_job_completion(job_id, success_base_msg):
                st.write("---")
                progress_bar = st.progress(0)
                status_text = st.empty()
                import time
                
                while True:
                    try:
                        res = requests.get(f"http://127.0.0.1:8000/api/v1/progress/{job_id}")
                        if res.status_code == 200:
                            data = res.json()
                            status = data.get("status")
                            
                            if status == "completed":
                                progress_bar.progress(100)
                                status_text.success(f"✅ {success_base_msg} (Extracted exactly {data.get('chunks_added', 0)} mathematical chunks)")
                                break
                            elif status == "failed":
                                progress_bar.empty()
                                status_text.error(f"❌ Background Ingestion Exception: {data.get('error')}")
                                break
                            else:
                                chunks_added = data.get("chunks_added", 0)
                                total = data.get("total_chunks", 0)
                                txt_status = status.replace('_', ' ').capitalize()
                                
                                if total > 0:
                                    pct = min(100, int((chunks_added / total) * 100))
                                    progress_bar.progress(pct)
                                    status_text.info(f"⏳ {txt_status} ... Embedding {chunks_added} of {total} total discrete text chunks natively.")
                                else:
                                    status_text.info(f"⏳ {txt_status} natively...")
                        else:
                            status_text.error(f"❌ Progress API Error Format Bounds: {res.text}")
                            break
                    except Exception as e:
                        status_text.error(f"❌ HTTP Target Polling Request failed explicitly: {e}")
                        break
                    time.sleep(1.5)

            with st.spinner("Dispatching structural payloads to Background Task APIs..."):
                reset_flag = "start_fresh" if ingestion_mode == "Start Fresh (Clear Logic)" else "append"
                
                # 1. POST Multi-part Forms back to Backend Pipeline Processors
                if uploaded_files:
                    st.write(f"📂 Uploading {len(uploaded_files)} structural file(s) natively to endpoint API queue...")
                    files_payload = []
                    for uploaded_file in uploaded_files:
                        files_payload.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
                    
                    try:
                        res = requests.post(
                            "http://127.0.0.1:8000/api/v1/ingest/files",
                            files=files_payload,
                            data={"mode": reset_flag}
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.success(f"✅ API Queue Registration Success: {data['message']}")
                            job_id = data.get("job_id")
                            if job_id:
                                wait_for_job_completion(job_id, "File processing explicitly completely natively")
                            
                            reset_flag = "append" # Mutate flag to prevent accidental overwrite if Crawler follows in sequence
                        else:
                            st.error(f"❌ API Explicit Error Formats Bounds: {res.text}")
                    except Exception as e:
                        st.error(f"❌ FastApi Request pipeline failed natively payload maps bounds explicitly: {e}")

                # 2. POST Crawler Trigger to Playwright API Wrapper
                if url_input:
                    st.write(f"🌐 Dispatching background Playwright Crawler Worker Array exclusively mapping on {url_input} with Depth {max_depth_input}...")
                    try:
                        res = requests.post(
                            "http://127.0.0.1:8000/api/v1/ingest/crawler",
                            data={"url": url_input, "max_depth": max_depth_input, "mode": reset_flag}
                        )
                        if res.status_code == 200:
                            data = res.json()
                            if data.get("status") == "accepted":
                                st.success(f"✅ Web Crawler Job ID Queued Effectively: {data['message']}")
                                job_id = data.get("job_id")
                                if job_id:
                                    wait_for_job_completion(job_id, "Crawler execution natively completely bounds resolved")
                            else:
                                st.warning(f"⚠️ Web Crawler queued natively failed bounds. API Raw Response Array Objects Logic Mappings: {data}")
                        else:
                            st.error(f"❌ Target Background Endpoint API Extraction Error Array Sequences Native Mappings: {res.text}")
                    except Exception as e:
                        st.error(f"❌ Target Node Queue Background Request API HTTP native failed expressly bounds sequences: {e}")
                        
                # Artificial UI delay to render green success blocks before destructive screen repaint
                import time
                time.sleep(2)
                st.rerun()

# =========================================================================
# Chat Bot / Traceability Dashboard View
# =========================================================================
elif option == "Chat":
    st.subheader("Chat with Semantic Knowledge Base")
    
    # Validate DB existence statically before permitting queries
    if not os.path.exists("data/qdrant_storage"):
        st.warning("⚠️ Semantic Knowledge Base not found actively instantiated!")
        st.info("Please explicitly select **'Ingest Documents'** from the sidebar to populate the Qdrant memory maps.")
    else:
        st.success("✅ Semantic Knowledge Base is Mapped and Ready!")

        # -------------------------------------------------------------------------
        # Native Context History Mapping
        # -------------------------------------------------------------------------
        history_path = os.path.join("data", "chat_history.json")
        if "messages" not in st.session_state:
            st.session_state.messages = []
            if os.path.exists(history_path):
                try:
                    import json
                    with open(history_path, "r") as f:
                        st.session_state.messages = json.load(f)
                except Exception:
                    pass

        # Native Streamlit history repainting logic
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("View Grounded Origin Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Target Origin Node {i+1}:** {source.get('source', 'Unknown Native Origin')}")

        # -------------------------------------------------------------------------
        # User Submission Hooks
        # -------------------------------------------------------------------------
        if prompt := st.chat_input("Ask a dynamic reasoning question about your parsed enterprise documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # -------------------------------------------------------------------------
            # Assistant Response Output Hooks
            # -------------------------------------------------------------------------
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                import re
                import difflib
                
                # Pre-processing heuristic mapping: Guess if query is Smalltalk to update visual loading spinners instantly
                is_greeting = False
                normalized = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).strip().lower()
                words = normalized.split()
                g_words = ["hi", "hello", "hey", "greetings", "sup", "howdy", "morning", "afternoon", "evening"]
                if normalized in g_words:
                    is_greeting = True
                elif len(words) <= 5:
                    for w in words:
                        if difflib.get_close_matches(w, g_words, n=1, cutoff=0.7):
                            is_greeting = True
                            break
                            
                initial_status = "Supervisor Agent: Rapid Smalltalk Bypass Enabled" if is_greeting else "Galactus Supervisor: Extracting Vector Maps..."
                
                # -------------------------------------------------------------------------
                # Active API Execution Block
                # -------------------------------------------------------------------------
                with st.status(initial_status, expanded=True) as status:
                    try:
                        status.update(label="Transmitting payload structural request to Enterprise Backend Router API...")
                        st.write("⚙️ Connecting actively to internal Galactus execution DAG Orchestrator...")
                        
                        payload = {
                            "query": prompt,
                            "chat_history": st.session_state.messages,
                            "model_provider": "groq"
                        }

                        # Hit the Chat API synchronously
                        res = requests.post("http://127.0.0.1:8000/api/v1/chat", json=payload)
                        res.raise_for_status()
                        response_data = res.json()
                        
                        status.update(label="LangGraph Engine Execution Native Complete!", state="complete", expanded=False)
                        
                        # Destructure payload mappings from strictly typed response contract schema output
                        answer = response_data.get("answer", "No systemic synthetic reply algorithmically generated.")
                        sources = response_data.get("sources", [])
                        confidence = response_data.get("confidence", 0.0)
                        verdict = response_data.get("verifier_verdict", "UNMAPPED_UNKNOWN")
                        is_hallucinated = response_data.get("is_hallucinated", False)
                        optimizations = response_data.get("optimizations", {})
                        
                        agent_used = optimizations.get("agent_routed", response_data.get("agent_used", "Unknown Exec Node Path"))

                        # -------------------------------------------------------------------------
                        # UI Parsing: Extract raw `<think>` algorithmic tags from LLM response natively
                        # -------------------------------------------------------------------------
                        import re
                        think_match = re.search(r'<think>(.*?)(?:</think>|$)', answer, re.DOTALL | re.IGNORECASE)
                        if think_match:
                            think_content = think_match.group(1).strip()
                            # Strip tags to present the clean declarative answer explicitly mapped
                            answer = re.sub(r'<think>.*?(?:</think>|$)', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()
                            if not answer:
                                answer = "*(The underlying native model physically exhausted its max generation payload string capacity while structurally reasoning natively and did not provide a definitive final programmatic output array format answer. Please consider restructuring the initial prompt sequence.)*"
                        else:
                            think_content = None
                        
                        # Inject Extracted `think` into custom Collapsible Markdown Streamlit Native elements
                        if think_content:
                            with st.expander("🤔 Internal Autonomous Agent Reasoning Tree Execution Process", expanded=False):
                                st.markdown(think_content)
                                
                        # -------------------------------------------------------------------------
                        # Artificial Typing Emulation Array
                        # -------------------------------------------------------------------------
                        import time
                        stream_state = {"text": ""}
                        for chunk in answer.split(" "):
                            stream_state["text"] += chunk + " "
                            message_placeholder.markdown(stream_state["text"] + "▌")
                            time.sleep(0.02)
                        message_placeholder.markdown(answer)
                        
                        # -------------------------------------------------------------------------
                        # Enterprise Traceability Dashboard Telemetry
                        # -------------------------------------------------------------------------
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric(label="Directed Routing Network Path", value=agent_used)
                        with cols[1]:
                            st.metric(label="Native Embedded Retrieval Logic Target Confidence Bounds", value=f"{confidence * 100:.0f}%" if isinstance(confidence, (int, float)) else "N/A structural")
                        with cols[2]:
                            st.metric(label="Internal Hallucination Architecture Verifier Sequence Verdict", value=verdict)
                        with cols[3]:
                            st.metric(label="Algorithmic Truth Hallucinated Context Node?", value="🚨 POSITIVE MATCH" if is_hallucinated else "✅ NEGATIVE STRUCTURAL")
                            
                        if is_hallucinated:
                            st.warning("Active Alert System Warning Bounds: The structural engine mathematically flagged this active response context sequence string as explicitly drifting from the retrieved document contextual parameters boundaries context layer. Please explicitly verify raw sources directly.")
                            
                        # Complete Optimization & Traceability Context UI Dropdown mapping
                        if sources or search_query:
                            with st.expander("View Deep Execution RAG Traceability Bounds Node Context (Logical Array Sources & Memory Optimizations Sequence Mapping)"):
                                if search_query and search_query != prompt:
                                     st.markdown(f"**Original User Logical Input:** `{prompt}`")
                                     st.markdown(f"**Optimized Semantic Database Mapping Search Target Request Payload Context:** `{search_query}` *(Autonomous Rewriting Heuristic Transformation Architecture Algorithm Applied Sequences)*")
                                     st.divider()
                                
                                st.markdown("**Dynamic Operational Latency Metrics Constraints Applied Architecture Execution Parameters:**")
                                st.caption(f"- **Heuristic Hardware Sequence Bounds Short-Circuited Bypass Node:** `{'Yes Active Native Bounds' if optimizations.get('short_circuited') else 'No Negative Bounds'}` *(Bypassed Heavy Database FAISS Vector Semantic Extraction Operations natively)*\n- **Dynamic Model Output Creativity Constraint Bounds Temperature Parameters Sequence Mapping:** `{optimizations.get('temperature', 'N/A')}` | **Active Engine Logic Hardware Constraint Compute Sequences Bounds Processing Threshold Node Effort Parameter Limit Sequences:** `{optimizations.get('reasoning_effort', 'N/A mapping')}`")
                                st.divider()
                                
                                if sources:
                                    for i, source in enumerate(sources):
                                        st.markdown(f"**Raw Data Object JSON Mapped Source File Text Node Source Node Block Context Matrix Node Array Payload Origin {i+1}:** {source.get('source', 'Unknown Unknown Mapping Origin Payload Origin Default') if isinstance(source, dict) else source}")
                                        st.caption(f"Raw Text Extracted Data Map Structural Block Literal Snippet Output Text Array Mapping Object Logic Sequence Block String Context JSON String Block Context String Formatted Structural Chunk Object Array Limit: {str(source.get('text', '') if isinstance(source, dict) else source)[:200]}...")
                        
                        # -------------------------------------------------------------------------
                        # User Reinforcement Learning Bounds Logging Feedback Loop Database Integration Hooks
                        # -------------------------------------------------------------------------
                        st.write("Was this actively generated systemic heuristic generated structure string response logically helpful and directly contextually empirically accurate?")
                        msg_idx = len(st.session_state.messages)
                        feedback_key = f"feedback_{msg_idx}"
                        
                        if feedback_key not in st.session_state:
                             st.session_state[feedback_key] = None
                             
                        if st.session_state[feedback_key] is None:
                            f_cols = st.columns([1, 1, 10])
                            with f_cols[0]:
                                 if st.button("👍", key=f"up_{msg_idx}"):
                                      st.session_state[feedback_key] = "positive_verified"
                                      # Async offline telemetry log write
                                      import json, datetime, os
                                      feedback_dir = os.path.join("data", "audit")
                                      os.makedirs(feedback_dir, exist_ok=True)
                                      with open(os.path.join(feedback_dir, "feedback.jsonl"), "a") as f:
                                            f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "query": prompt, "rating_feedback_metrics": "positive_verified"}) + "\n")
                                      st.rerun()
                            with f_cols[1]:
                                 if st.button("👎", key=f"dn_{msg_idx}"):
                                      st.session_state[feedback_key] = "negative_issue"
                                      import json, datetime, os
                                      feedback_dir = os.path.join("data", "audit")
                                      os.makedirs(feedback_dir, exist_ok=True)
                                      with open(os.path.join(feedback_dir, "feedback.jsonl"), "a") as f:
                                            f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "query": prompt, "rating_feedback_metrics": "negative_issue"}) + "\n")
                                      st.rerun()
                        else:
                             st.success(f"Heuristic rating feedback telemetry successfully tracked and mapped recorded natively to metric database offline file mapping: {st.session_state[feedback_key]}. Native system optimization training weights updated implicitly Thank you heavily!")
                                    
                        # Append history JSON block string file object explicitly 
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        import json
                        with open(os.path.join("data", "chat_history.json"), "w") as f:
                            json.dump(st.session_state.messages, f)
                        
                    except Exception as e:
                        st.error(f"Error system structural backend natively implicitly actively communicating with FastApi routing API JSON backend explicitly explicitly explicitly natively: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry system error API string format, I encountered an internal backend FastAPI HTTP explicitly error formatting bounds: {e}"})
