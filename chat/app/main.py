import base64
import os
import time
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from mistralai import Mistral
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from utils.logger import logger, truncate_payload

load_dotenv()

# --- CONFIG ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "preprocessed_files")
QA_COLLECTION_NAME = os.getenv("QA_COLLECTION_NAME", "chat_conversations")
EMBEDDING_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))  # Mistral embeddings dimension
DISTANCE = models.Distance.COSINE

logger.info("chat/main.py - Initializing chat service")
logger.debug(f"chat/main.py - QDRANT_HOST: {QDRANT_HOST}")
logger.debug(f"chat/main.py - QDRANT_PORT: {QDRANT_PORT}")
logger.debug(f"chat/main.py - COLLECTION_NAME: {COLLECTION_NAME}")
logger.debug(f"chat/main.py - QA_COLLECTION_NAME: {QA_COLLECTION_NAME}")
logger.debug(f"chat/main.py - EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.debug(f"chat/main.py - LLM_MODEL: {LLM_MODEL}")
logger.debug(f"chat/main.py - VECTOR_SIZE: {VECTOR_SIZE}")

# --- CLIENTS ---
logger.info("chat/main.py - Connecting to Qdrant and Mistral")
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("chat/main.py - Successfully connected to Qdrant")
except Exception as e:
    logger.error(f"chat/main.py - Failed to connect to Qdrant: {str(e)}")
    raise

try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    logger.info("chat/main.py - Successfully initialized Mistral client")
except Exception as e:
    logger.error(f"chat/main.py - Failed to initialize Mistral client: {str(e)}")
    raise

# --- FASTAPI APP ---
app = FastAPI()

def check_create_collection(collection_name: str):
    """Creates a collection if it doesn't exist"""
    logger.info(f"chat/main.py - check_create_collection() - Checking if collection {collection_name} exists")
    
    try:
        existing_collections = [c.name for c in qdrant_client.get_collections().collections]
        logger.debug(f"chat/main.py - check_create_collection() - Existing collections: {existing_collections}")
        
        if collection_name not in existing_collections:
            logger.info(f"chat/main.py - check_create_collection() - Creating new collection: {collection_name}")
            
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE
                )
            )
            logger.info(f"chat/main.py - check_create_collection() - Successfully created collection: {collection_name}")
        else:
            logger.info(f"chat/main.py - check_create_collection() - Collection {collection_name} already exists")
            
    except Exception as e:
        logger.error(f"chat/main.py - check_create_collection() - Error creating collection: {str(e)}")
        raise

@app.on_event("startup")
async def initialize_collections():
    """Initialize all necessary collections"""
    logger.info("chat/main.py - initialize_collections() - Initializing collections on startup")
    
    try:
        check_create_collection(COLLECTION_NAME)
        check_create_collection(QA_COLLECTION_NAME)
        logger.info("chat/main.py - initialize_collections() - All collections initialized successfully")
    except Exception as e:
        logger.error(f"chat/main.py - initialize_collections() - Error initializing collections: {str(e)}")
        raise

class ChatRequest(BaseModel):
    question: str
    history: list = []

def get_embedding(text: str):
    logger.info("chat/main.py - get_embedding() - Generating embedding for text")
    logger.debug(f"chat/main.py - get_embedding() - Text sample: {truncate_payload(text)}")
    try:
        result = mistral_client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[text]
        )
        logger.debug(f"chat/main.py - get_embedding() - Embedding generated with dimensionality: {len(result.data[0].embedding)}")
        return result.data[0].embedding
    except Exception as e:
        logger.error(f"chat/main.py - get_embedding() - Error generating embedding: {str(e)}")
        raise
    
def detect_intent(question: str) -> str:
    """
    Uses the LLM to classify the intent of the query.
    Returns 'document_search' or 'general_chat'.
    """
    intent_prompt = (
        "Classify the intent of the following question as 'document_search' if the user wants to search for information "
        "in documents or as 'general_chat' for any other request.\n\n"
        f"Question: {question}\n"
        "Reply only with one of the two labels: document_search or general_chat."
    )
    try:
        response = mistral_client.chat.complete(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": intent_prompt}],
            temperature=0
        )
        intent = response.choices[0].message.content.strip().lower()
        if "document_search" in intent:
            return "document_search"
        return "general_chat"
    except Exception as e:
        logger.error(f"Intent detection failed: {str(e)}")
        return "general_chat"  # fallback


def decode_chunk(chunk_b64: str):
    logger.debug(f"chat/main.py - decode_chunk() - Decoding base64 chunk of length: {len(chunk_b64) if chunk_b64 else 0}")
    try:
        if not chunk_b64:
            logger.warning("chat/main.py - decode_chunk() - Empty chunk provided")
            return ""
            
        decoded = base64.b64decode(chunk_b64).decode('utf-8', errors='replace')
        logger.debug(f"chat/main.py - decode_chunk() - Successfully decoded chunk with length: {len(decoded)}")
        return decoded
    except Exception as e:
        logger.error(f"chat/main.py - decode_chunk() - Decoding error: {str(e)}")
        return ""
    
def call_mistral_with_retry(mistral_client, model, prompt, max_retries=3, wait_time=2):
    """Chiamata LLM con retry/backoff per gestire rate limit"""
    for attempt in range(max_retries):
        try:
            return mistral_client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 429 or "429" in str(e):
                wait = wait_time
                # Se c'è header Retry-After, usalo
                if hasattr(e, "raw_response") and hasattr(e.raw_response, "headers"):
                    retry_after = e.raw_response.headers.get("Retry-After")
                    if retry_after and str(retry_after).isdigit():
                        wait = int(retry_after)
                logger.warning(f"Rate limit hit (429). Retry attempt {attempt+1} in {wait} seconds...")
                time.sleep(wait)
            else:
                logger.error(f"Unexpected error during LLM call: {str(e)}")
                raise
    raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please wait and try again.")

def retrieve_context(query: str, top_k: int = 3, intent: str = "general_chat"):
    logger.info(f"Retrieving context from both collections for top_k={top_k} and intent={intent}")
    embedding = get_embedding(query)
    
    if intent == "document_search":
        # Cerca solo nella collection principale
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=.7
        )
    else:
        # Cerca in entrambe le collection (come prima)
        hits_main = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=.7
        )
        hits_qa = qdrant_client.search(
            collection_name=QA_COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=.7
        )
        hits = list(hits_main) + list(hits_qa)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = hits[:top_k]

    # Costruisci la lista delle sources
    context_chunks = []
    for hit in hits:
        payload = hit.payload or {}
        chunk_b64 = ""
        if "chunk" in payload:
            chunk_b64 = payload.get("chunk", "")
        elif "question" in payload and "answer" in payload:
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            combined_qa = f"Question: {question}\nAnswer: {answer}"
            chunk_b64 = base64.b64encode(combined_qa.encode('utf-8')).decode('utf-8')
            logger.debug(f"chat/main.py - retrieve_context() - Created QA context: {truncate_payload(combined_qa)}")

        context_chunks.append({
            "text": decode_chunk(chunk_b64) if chunk_b64 else "",
            "source": payload.get("source", "unknown"),
            "score": round(hit.score, 2)
        })
    valid_chunks = [c for c in context_chunks if c["text"].strip()]
    context_chunks = valid_chunks[:top_k]
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    logger.info(f"chat/main.py - retrieve_context() - Filtered to {len(context_chunks)} valid chunks")
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    logger.debug(f"Assembled context with length: {len(context_text)}")
    
    return context_text, context_chunks


def build_prompt(question: str, context: str, history: list):
    logger.info("chat/main.py - build_prompt() - Building prompt for LLM")
    logger.debug(f"chat/main.py - build_prompt() - Question: {truncate_payload(question)}")
    logger.debug(f"chat/main.py - build_prompt() - Context length: {len(context)}")
    logger.debug(f"chat/main.py - build_prompt() - History entries: {len(history)}")
    history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in history])
    
    prompt = f"""Answer the question below. 
                If the provided context contains sufficient information, use ONLY that to answer.
                If the context is NOT sufficient or does not contain the answer, respond based on your general knowledge.                
                History:
                {history_str}

                Context:
                {context}

                Question: {question}
                Answer:"""
    
    logger.debug(f"chat/main.py - build_prompt() - Prompt length: {len(prompt)}")
    return prompt
                
                
                
def chunk_in_answer(chunk_text, answer, window=80):
    logger.debug(f"chat/main.py - chunk_in_answer() - Checking if chunk text appears in answer")
    logger.debug(f"chat/main.py - chunk_in_answer() - Chunk length: {len(chunk_text)}, Answer length: {len(answer)}, Window: {window}")
    
    chunk_text = chunk_text.strip().lower()
    answer = answer.lower()
    
    if not chunk_text:
        logger.debug("chat/main.py - chunk_in_answer() - Empty chunk text, returning False")
        return False
    
    for i in range(0, len(chunk_text) - window + 1, window):
        window_text = chunk_text[i:i+window]
        if window_text in answer:
            logger.debug(f"chat/main.py - chunk_in_answer() - Found match at position {i}, window: {truncate_payload(window_text)}")
            return True
    
    logger.debug("chat/main.py - chunk_in_answer() - No matches found")
    return False


@app.post("/chat")
async def chat(request: Request):
    logger.info("chat/main.py - chat() - Processing chat request")
    try:
        data = await request.json()
        question = data["question"]
        intent = detect_intent(question)
        logger.info(f"chat/main.py - chat() - Detected intent: {intent}")
        history = data.get("history", [])
        logger.debug(f"chat/main.py - chat() - Question: {truncate_payload(question)}")
        logger.debug(f"chat/main.py - chat() - History entries: {len(history)}")

        # Retrieval context
        logger.info("chat/main.py - chat() - Retrieving relevant context")
        context, sources = retrieve_context(question, intent=intent)
        logger.info(f"chat/main.py - chat() - Retrieved {len(sources)} context chunks")

        # Generate answer
        logger.info("chat/main.py - chat() - Building prompt and calling LLM")
        prompt = build_prompt(question, context, history)
        logger.debug("chat/main.py - chat() - Calling Mistral API with retry/backoff logic")        # --- Retry logic here ---
        response = call_mistral_with_retry(
            mistral_client,
            LLM_MODEL,
            prompt,
            max_retries=3,
            wait_time=2
        )

        answer = response.choices[0].message.content
        logger.info("chat/main.py - chat() - Received response from LLM")
        logger.debug(f"chat/main.py - chat() - Answer length: {len(answer)}")

        # Check if context was used in the answer
        logger.debug("chat/main.py - chat() - Checking if context was used in the answer")
        context_used = bool(context.strip()) and len(sources) > 0

        logger.info(f"chat/main.py - chat() - Context used in answer: {context_used}")

        # Save conversation
        logger.info("chat/main.py - chat() - Saving conversation to vector store")
        check_create_collection(QA_COLLECTION_NAME)
        conversation_id = str(uuid4())
        logger.debug(f"chat/main.py - chat() - Generated conversation ID: {conversation_id}")

        embedding = get_embedding(question)
        qdrant_client.upsert(
            collection_name=QA_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=conversation_id,
                    vector=embedding,
                    payload={
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "sources": sources if context_used else [],
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )
        logger.info("chat/main.py - chat() - Conversation saved successfully")

        return {
            "answer": answer,
            "sources": sources if context_used else [],
            "context": context
        }

    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"chat/main.py - chat() - Error processing chat request: {str(e)}")
        raise