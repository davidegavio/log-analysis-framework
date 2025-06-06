import io
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from utils.logger import logger, truncate_payload

load_dotenv()

app = FastAPI()
PREPROCESSING_URL = os.getenv("PREPROCESSING_URL")
RAG_CHAT_URL = os.getenv("RAG_CHAT_URL")

@app.post("/ingest")
async def ingest_log(request: Request):
    logger.info(f"api_gateway/app/main.py - ingest_log() - Starting log ingestion")
    try:
        data = await request.body()
        logger.debug(f"api_gateway/app/main.py - ingest_log() - Received raw data, size: {len(data)} bytes")
        files = {
            "file": ("log_file", io.BytesIO(data))
        }
        logger.info(f"api_gateway/app/main.py - ingest_log() - Forwarding to preprocessing at {PREPROCESSING_URL}")
        async with httpx.AsyncClient() as client:
            resp = await client.post(PREPROCESSING_URL, files=files, timeout=None)
            logger.debug(f"api_gateway/app/main.py - ingest_log() - Preprocessing response status: {resp.status_code}")
            logger.debug(f"api_gateway/app/main.py - ingest_log() - Preprocessing response: {truncate_payload(resp.json())}")
        return resp.json()
    except Exception as e:
        logger.error(f"api_gateway/app/main.py - ingest_log() - Error: {str(e)}")
        return {"error": str(e)}

@app.post("/chat")
async def rag_chat(request: Request):
    logger.info(f"api_gateway/app/main.py - rag_chat() - Starting chat request")
    data = await request.json()
    logger.debug(f"api_gateway/app/main.py - rag_chat() - Received chat request: {truncate_payload(data)}")
    logger.info(f"api_gateway/app/main.py - rag_chat() - Forwarding to RAG chat at {RAG_CHAT_URL}")
    async with httpx.AsyncClient() as client:
        resp = await client.post(RAG_CHAT_URL, json=data, timeout=60.0)
        logger.debug(f"api_gateway/app/main.py - rag_chat() - RAG chat response status: {resp.status_code}")
        logger.debug(f"api_gateway/app/main.py - rag_chat() - RAG chat response: {truncate_payload(resp.json())}")
    return resp.json()
