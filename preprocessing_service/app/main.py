import base64
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from mistralai import Mistral
from qdrant_client import QdrantClient, models

from utils.logger import logger, truncate_payload

load_dotenv()

app = FastAPI()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
DISTANCE = os.getenv("DISTANCE", "Cosine")

api_key = os.getenv("MISTRAL_API_KEY")
model = os.getenv("MISTRAL_EMBED_MODEL")

logger.info(f"preprocessing_service/app/main.py - init - Initializing with model: {model}, collection: {COLLECTION_NAME}")
mistral_client = Mistral(api_key=api_key)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
logger.info(f"preprocessing_service/app/main.py - init - Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
logger.debug(f"preprocessing_service/app/main.py - init - Using chunk size: {CHUNK_SIZE}")

def chunk_bytes(data: bytes, chunk_size: int = CHUNK_SIZE):
    logger.debug(f"preprocessing_service/app/main.py - chunk_bytes() - Chunking data of size: {len(data)} bytes with chunk_size: {chunk_size}")
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    logger.debug(f"preprocessing_service/app/main.py - chunk_bytes() - File split into {len(chunks)} chunks")
    return chunks

@app.post("/preprocess")
async def preprocess_file(file: UploadFile = File(...)):
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Starting file preprocessing: {file.filename}")
    content = await file.read()
    logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - File content read, size: {len(content)} bytes")
    logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - File preview: {truncate_payload(content.decode('utf-8', errors='replace'))}")

    # Send the file to the orchestrator (as bytes)
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Sending file to orchestrator at {ORCHESTRATOR_URL}")
    async with httpx.AsyncClient() as client_http:
        response = await client_http.post(ORCHESTRATOR_URL, content=content, timeout=None)
        logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - Orchestrator response status: {response.status_code}")
        logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - Orchestrator response: {truncate_payload(response.text)}")
        # Supponiamo che l'orchestrator restituisca un JSON con 'reasoning' e 'aggregated_analysis'
        try:
            orchestrator_result = response.json()
        except Exception:
            orchestrator_result = {}

    # Chunk the file
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Starting file chunking")
    chunks = chunk_bytes(content)
    logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - Created {len(chunks)} chunks")

    # Create embeddings
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Calculating embeddings for {len(chunks)} chunks")
    embeddings_batch_response = mistral_client.embeddings.create(
        model=model,
        inputs=[chunk.decode("utf-8", errors="ignore") for chunk in chunks]
    )
    embeddings = []
    if embeddings_batch_response:
        embeddings = [data.embedding for data in embeddings_batch_response.data]
        logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - Generated {len(embeddings)} embeddings")
    else:
        logger.warning(f"preprocessing_service/app/main.py - preprocess_file() - No embeddings returned from API")

    # Prepare payload with base64 encoded chunks
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Preparing payload for Qdrant")
    payloads = [{"chunk": base64.b64encode(chunk).decode("utf-8")} for chunk in chunks]
    logger.debug(f"preprocessing_service/app/main.py - preprocess_file() - Created {len(payloads)} payload entries")

    # Insertion into Qdrant
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Creating collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=DISTANCE)
        )
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Upserting {len(embeddings)} vectors into Qdrant")
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": idx,
            "vector": embeddings[idx],
            "payload": payloads[idx]
        } for idx in range(len(chunks))]
    )
    logger.info(f"preprocessing_service/app/main.py - preprocess_file() - Preprocessing completed successfully")
    return {
        "status": "success",
        "reasoning": orchestrator_result.get("reasoning", []),
        "analysis": orchestrator_result.get("aggregated_analysis", ""),
        "chunks": len(chunks),
        "embeddings": len(embeddings)
    }
