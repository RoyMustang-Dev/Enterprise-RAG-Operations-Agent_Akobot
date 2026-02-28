"""
Enterprise RAG Operations Agent - Pure API Client UI
This Streamlit module contains ZERO backend logic. It acts purely as a thin client
that interfaces exclusively with the FastAPI backend through standard HTTP calls.
"""
import os
import streamlit as st
import requests
import json
import uuid

# =========================================================================
# System Setup & Configuration
# =========================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Enterprise AI Client", layout="wide", page_icon="🤖")
st.title("Enterprise Multimodal AI Assistant")

# Initialize Session State Variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================================
# Sidebar Configurations
# =========================================================================
st.sidebar.header("⚙️ Settings")
model_provider = st.sidebar.selectbox(
    "AI Provider", 
    ["auto", "groq", "openai", "anthropic", "gemini"], 
    help="Choose which AI provider powers the chat. 'Auto' dynamically routes based on heuristics."
)
image_mode = st.sidebar.selectbox(
    "Image Processing", 
    ["auto", "ocr", "vision"], 
    help="How should uploaded images be read? 'Auto' attempts reading text first, then uses Vision if needed."
)
reranker_profile = st.sidebar.selectbox(
    "Reranker Profile",
    ["auto", "accurate", "fast", "off"],
    help="Select the cross-encoder reranking strategy for RAG. 'accurate' uses large models, 'fast' uses base models."
)
enable_streaming = st.sidebar.toggle(
    "Stream Chat Responses", 
    value=True, 
    help="Shows the AI's answer word-by-word as it's generated."
)

st.sidebar.divider()
st.sidebar.header("🔐 User Identity")
tenant_id = st.sidebar.text_input("Tenant ID", value="default_enterprise", help="Your organization's workspace name.")
user_id = st.sidebar.text_input("User ID", value="admin_user", help="Your unique user identifier.")

st.sidebar.divider()
st.sidebar.header("🛠️ System Health")
if st.sidebar.button("Check API Connection"):
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            st.sidebar.success(f"🟢 Connected properly to Backend!")
        else:
            st.sidebar.error(f"🟡 Connection Issue (Code: {res.status_code})")
    except Exception as e:
        st.sidebar.error(f"🔴 Offline: Cannot reach the backend API.")

# =========================================================================
# View Separation
# =========================================================================
tab_chat, tab_ingest = st.tabs(["💬 Chat with AI", "📚 Manage Knowledge Base"])

# =========================================================================
# TAB 1: Multimodal Dynamic Chat View
# =========================================================================
with tab_chat:
    st.info("💬 Tip: You can type a question, record a voice query, or upload files directly into this chat session. Files uploaded here are temporary and only last for 24 hours.")
    
    # 1. Paint Existing Message Native History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display attached API execution metadata (only for assistant interactions)
            meta = msg.get("meta")
            if meta:
                with st.expander("📊 View Detailed RAG Traceability (Sources, Stats)"):
                    st.json(meta)
                    
            # Native Feedback Metrics UI Logic
            if msg["role"] == "assistant" and "feedback" not in msg:
                f_cols = st.columns([1, 1, 10])
                if f_cols[0].button("👍 Useful", key=f"up_{idx}"):
                    msg["feedback"] = "positive"
                    requests.post(f"{API_BASE_URL}/api/v1/feedback", json={"session_id": st.session_state.session_id, "rating": "up"})
                    st.rerun()
                if f_cols[1].button("👎 Hallucinated", key=f"dn_{idx}"):
                    msg["feedback"] = "negative"
                    requests.post(f"{API_BASE_URL}/api/v1/feedback", json={"session_id": st.session_state.session_id, "rating": "down"})
                    st.rerun()
            
            # Text-to-Speech Output Execution UI Mapping 
            if msg["role"] == "assistant":
                if st.button("🔊 Play Audio (TTS)", key=f"tts_{idx}"):
                    with st.spinner("Generating speech..."):
                        try:
                            # Strip any system tags like <think> out before piping to Text To Speech audio synthesis
                            clean_text = msg["content"]
                            import re
                            clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
                            res = requests.post(f"{API_BASE_URL}/api/v1/tts", data={"text": clean_text})
                            res.raise_for_status()
                            st.audio(res.content, format="audio/wav")
                        except Exception as e:
                            st.error(f"TTS Synthesis Error: {e}")

    st.markdown("---")
    
    # 2. Input Composition Blocks (File, Voice, Text Forms)
    col_files, col_voice = st.columns(2)
    with col_files:
        chat_files = st.file_uploader(
            "📎 Upload Temporary Session Files (PDF/DOCX/TXT/Images)",
            accept_multiple_files=True,
            help="Files uploaded here form a temporary memory layer for the duration of exactly 24 hours."
        )
    with col_voice:
        audio_val = st.audio_input("🎙️ Record Voice Query")
        
    prompt = st.chat_input("Ask a question about your documents, images, or knowledge base...")
    
    # Track audio state to prevent duplicate submissions on Streamlit UI reruns
    if "last_processed_audio_id" not in st.session_state:
        st.session_state.last_processed_audio_id = None
        
    # Generate a unique hash for the audio buffer to track whether we've already transcribed it
    audio_hash = hash(audio_val.getvalue()) if audio_val else None

    # Pre-process Microphone Input (Only if we have new audio and no text prompt)
    transcribed_text = None
    if audio_val and not prompt and st.session_state.last_processed_audio_id != audio_hash:
        with st.spinner("Transcribing your audio using AI..."):
            try:
                files_payload = {"audio_file": ("voice.wav", audio_val.getvalue(), "audio/wav")}
                t_res = requests.post(f"{API_BASE_URL}/api/v1/transcribe", files=files_payload)
                if t_res.status_code == 200:
                    transcribed_text = t_res.json().get("transcript", "")
                    st.session_state.last_processed_audio_id = audio_hash # Mark as processed
                else:
                    st.error(f"Transcription Failed: {t_res.text}")
            except Exception as e:
                st.error(f"Transcription Error: {e}")

    final_query = prompt or transcribed_text
    
    # 3. Main Chat API Execution Call
    if final_query:
        # Commit User Text Input Request To Canvas
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)
            
        with st.chat_message("assistant"):
            try:
                # Compile strict contextual message payload for API
                history_json = json.dumps([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]])
                params = {
                    "query": final_query,
                    "model_provider": model_provider,
                    "session_id": st.session_state.session_id,
                    "image_mode": image_mode,
                    "reranker_profile": reranker_profile,
                    "stream": "true" if enable_streaming else "false",
                    "chat_history": history_json
                }
                
                headers = {"x-tenant-id": tenant_id, "x-user-id": user_id}
                
                # Bundle multi-part files attached during prompt payload composition mapping
                files_payload = []
                if chat_files:
                    for f in chat_files:
                        files_payload.append(("files", (f.name, f.getvalue(), f.type)))
                        
                with st.spinner("Connecting structural pipelines to remote execution graph..."):
                    if enable_streaming:
                        res = requests.post(f"{API_BASE_URL}/api/v1/chat", data=params, files=files_payload, headers=headers, stream=True)
                        res.raise_for_status()
                        
                        placeholder = st.empty()
                        full_answer = ""
                        meta_data = {}
                        is_meta = False
                        
                        for line in res.iter_lines():
                            if line:
                                dec = line.decode("utf-8")
                                if dec.startswith("event: meta"):
                                    is_meta = True
                                elif dec.startswith("data: "):
                                    payload_str = dec[6:]
                                    if is_meta:
                                        meta_data = json.loads(payload_str)
                                        is_meta = False
                                    else:
                                        # Process raw text fragment map formatting
                                        chunk = payload_str
                                        full_answer += chunk
                                        placeholder.markdown(full_answer + "▌")
                                else:
                                    # Fallback payload structural fragment processing (Handles unescaped newline carriage returns cleanly in SSE streams natively)
                                    if not is_meta and dec.strip() != "":
                                        full_answer += "\\n" + dec
                                        placeholder.markdown(full_answer + "▌")
                                        
                        placeholder.markdown(full_answer)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_answer,
                            "meta": meta_data
                        })
                        
                        if meta_data:
                            with st.expander("📊 View Detailed Traceability Bounds & Pipeline Optimizations"):
                                st.json(meta_data)
                                
                    else:
                        # Blocking Request API execution bounds
                        res = requests.post(f"{API_BASE_URL}/api/v1/chat", data=params, files=files_payload, headers=headers)
                        res.raise_for_status()
                        data = res.json()
                        ans = data.get("answer", "")
                        st.markdown(ans)
                        
                        meta = {k: v for k, v in data.items() if k != "answer"}
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "meta": meta
                        })
                        
                        with st.expander("📊 View Detailed Traceability Bounds & Pipeline Optimizations"):
                            st.json(meta)
                            
            except Exception as e:
                st.error(f"Critical Native Pipeline Communication Exception API Node Structure Bounds Blocked: {e}")

# =========================================================================
# TAB 2: Global Background Data Ingestion Hub
# =========================================================================
with tab_ingest:
    st.subheader("📚 Global Knowledge Base Data Ingestion")
    st.info("Upload documents or crawl websites here to add them to the main AI knowledge base. This data will be permanently available to all users across all chat sessions.")
    
    st.markdown("**Current Database Status:**")
    if st.button("Check Active Database Status"):
        try:
             res = requests.get(f"{API_BASE_URL}/api/v1/ingest/status", headers={"x-tenant-id": tenant_id})
             if res.status_code == 200:
                 st.json(res.json())
             else:
                 st.error(f"Failed to fetch database status: {res.text}")
        except Exception as e:
             st.error(f"Error connecting to backend: {e}")
             
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
         st.markdown("### 📄 Process Documents")
         st.caption("Upload PDF, TXT, or DOCX files to extract their text and add them to the vector database.")
         global_files = st.file_uploader("Select Documents", accept_multiple_files=True, key="global_files")
         mode = st.radio("Insertion Strategy", ["append", "overwrite"], help="'Append' adds new data. 'Overwrite' deletes the entire database before adding these files!")
         if st.button("Upload and Process Files"):
             if global_files:
                 with st.spinner("Submitting files to the background queue..."):
                     payload = [("files", (f.name, f.getvalue(), f.type)) for f in global_files]
                     res = requests.post(f"{API_BASE_URL}/api/v1/ingest/files", files=payload, data={"mode": mode}, headers={"x-tenant-id": tenant_id})
                     if res.status_code == 200:
                         job_id = res.json().get("job_id")
                         st.success(f"✅ Success! Your files are being processed in the background.\n\n**Copy this Job ID to track progress:** `{job_id}`")
                     else:
                         st.error(f"Failed to submit files: {res.text}")
             else:
                 st.warning("Please upload at least one file before clicking this button.")

    with c2:
         st.markdown("### 🕸️ Web Crawler")
         st.caption("Provide a website URL for the AI to read, scrape, and learn from.")
         url = st.text_input("Website URL", help="Example: https://en.wikipedia.org/wiki/Artificial_intelligence")
         depth = st.number_input("Crawl Depth", min_value=1, max_value=4, value=1, help="Depth 1 = Just the main page. Depth 2 = The page and all its links.")
         if st.button("Start Web Crawler"):
             if url:
                 with st.spinner("Starting crawler..."):
                     res = requests.post(f"{API_BASE_URL}/api/v1/ingest/crawler", data={"url": url, "max_depth": depth, "mode": "append"}, headers={"x-tenant-id": tenant_id})
                     if res.status_code == 200:
                         job_id = res.json().get("job_id")
                         st.success(f"✅ Success! The crawler has started in the background.\n\n**Copy this Job ID to track progress:** `{job_id}`")
                     else:
                         st.error(f"Failed to start crawler: {res.text}")
             else:
                  st.warning("Please enter a valid URL before starting the crawler.")
                     
    st.divider()
    st.markdown("### 🔎 Track Background Job Status")
    st.info("Paste the Job ID you received above to see the live progress of your file ingestion or web crawler.")
    job_trace_id = st.text_input("Enter Job ID:")
    if st.button("Check Progress"):
        if job_trace_id:
            with st.spinner("Fetching job status..."):
                res = requests.get(f"{API_BASE_URL}/api/v1/progress/{job_trace_id}")
                if res.status_code == 200:
                    data = res.json()
                    status = data.get("status", "unknown").upper()
                    st.markdown(f"**Current Status:** `{status}`")
                    st.json(data)
                else:
                    st.error("Could not find a job with that ID. Please check the ID and try again.")
        else:
            st.warning("Please enter a Job ID to check progress.")
