import io
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

# --- CONFIGURAZIONE ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api_gateway:8000")
DB_URL = os.getenv("SYNC_DATABASE_URL", "postgresql+psycopg2://user:password@db:5432/logs")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

st.set_page_config(
    page_title="Log Analysis Dashboard",
    layout="wide",
    page_icon="🤖"
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "Chat"
if "processing_file" not in st.session_state:
    st.session_state.processing_file = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigazione")
    page_options = {
        "💬 Chat": "Chat",
        "🗂️ Database": "Postgres",
        "🔍 Vettori": "Qdrant"
    }
    for label, page in page_options.items():
        if st.button(label, use_container_width=True):
            st.session_state.page = page

    st.markdown("---")
    st.subheader("Cronologia Chat")
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.caption(f"👤 {msg['content'][:50]}...")
        elif msg["role"] == "assistant":
            st.caption(f"🤖 {msg['content'][:50]}...")

# --- MAIN CONTENT ---
if st.session_state.page == "Chat":
    st.title("🤖 Intelligent Log Analysis Chat")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Reasoning come lista di step
                if "reasoning" in message and message["reasoning"]:
                    with st.expander("Reasoning step-by-step"):
                        if isinstance(message["reasoning"], list):
                            st.markdown("\n".join(f"- {step}" for step in message["reasoning"]))
                        else:
                            st.markdown(str(message["reasoning"]))
                # Analisi aggregata
                if "analysis" in message and message["analysis"]:
                    with st.expander("Analisi aggregata"):
                        st.markdown(message["analysis"])
                # Visualizza contesto e fonti solo per risposte RAG (assistant, senza analysis)
                if (
                    message["role"] == "assistant"
                    and not ("analysis" in message and message["analysis"])
                ):
                    with st.expander("Contesto utilizzato per la risposta"):
                        if "sources" in message and message["sources"]:
                            for idx, chunk in enumerate(message["sources"]):
                                st.markdown(f"**Fonte {idx+1}:** {chunk.get('source', 'sconosciuta')}")
                                st.markdown(f"**Relevance Score:** {chunk.get('score', 'N/A'):.4f}")
                                st.code(chunk.get("text", "")[:800] + ("..." if len(chunk.get("text", "")) > 800 else ""))
                        else:
                            st.markdown("Nessun contesto disponibile.")

    if st.session_state.processing_file:
        st.info("🔄 Processing file in background...")

    # --- Chat input con upload file ---
    user_input = st.chat_input(
        "Scrivi la tua domanda o carica un file...",
        accept_file=True,
        file_type=None,
        disabled=False
    )

    if user_input:
        # Messaggio testuale
        if user_input.text:
            st.session_state.messages.append({"role": "user", "content": user_input.text})
            st.rerun()

        # Upload file
        if user_input.files and not st.session_state.processing_file:
            uploaded_file = user_input.files[0]
            st.session_state.processing_file = True
            try:
                file_bytes = uploaded_file.read()
                response = requests.post(
                    f"{API_GATEWAY_URL}/ingest",
                    data=file_bytes,
                    timeout=None
                )
                result = response.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ File '{uploaded_file.name}' processato con successo.",
                    "reasoning": result.get("reasoning", []),
                    "analysis": result.get("analysis", "") or result.get("aggregated_analysis", "")
                })
                st.session_state.processing_file = False
                st.success(f"File '{uploaded_file.name}' processato con successo")
                st.rerun()
            except Exception as e:
                st.error(f"Errore durante l'upload: {str(e)}")
                st.session_state.processing_file = False

    # --- Gestione risposta assistant alla chat (separata) ---
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        last_user_msg = st.session_state.messages[-1]["content"]
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
            if msg["role"] in ["user", "assistant"]
        ]
        try:
            response = requests.post(
                f"{API_GATEWAY_URL}/chat",
                json={"question": last_user_msg, "history": history[:-1]},
                timeout=60
            )
            result = response.json()
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "reasoning": result.get("reasoning", []),
                "analysis": result.get("analysis", "") or result.get("aggregated_analysis", ""),
                "sources": result.get("sources", []),
                "context": result.get("context", "")
            })
            st.rerun()
        except Exception as e:
            st.error(f"Errore durante la generazione: {str(e)}")

elif st.session_state.page == "Postgres":
    st.title("🗂️ Log Storage - PostgreSQL")
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(DB_URL)
        st.subheader("Database Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_logs = pd.read_sql("SELECT COUNT(*) FROM logs", engine).iloc[0, 0]
            st.metric("Total Logs", total_logs)
        with col2:
            last_update = pd.read_sql("SELECT MAX(timestamp) FROM logs", engine).iloc[0, 0]
            last_update_str = str(last_update) if pd.notnull(last_update) else "N/A"
            st.metric("Last Update", last_update_str)
        with col3:
            common_errors = pd.read_sql(
                "SELECT category, COUNT(*) FROM logs GROUP BY category ORDER BY COUNT DESC LIMIT 1",
                engine
            )
            common_cat = str(common_errors.iloc[0, 0]) if not common_errors.empty else "N/A"
            st.metric("Most Common Category", common_cat)
        st.subheader("Log Details")
        df = pd.read_sql("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 500", engine)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Database error: {str(e)}")

elif st.session_state.page == "Qdrant":
    st.title("🔍 Vectors - Qdrant")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        selected_collection = st.selectbox(
            "Select Collection",
            [col.name for col in collections]
        )
        collection_info = client.get_collection(selected_collection)
        st.subheader(f"Collection: {selected_collection}")
        st.metric("Vectors Count", collection_info.vectors_count)
        st.metric("Vector Size", f"{collection_info.config.params.vectors.size}D")
        st.subheader("Explore Vectors")
        points_result = client.scroll(
            collection_name=selected_collection, 
            limit=50, 
            with_payload=True
        )
        points = points_result[0]
        for point in points:
            with st.expander(f"Vector {point.id}"):
                st.json(point.payload)
    except Exception as e:
        st.error(f"Qdrant error: {str(e)}")
