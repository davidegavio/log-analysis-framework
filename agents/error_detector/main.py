import asyncio
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from utils.logger import logger, truncate_payload

load_dotenv()

app = FastAPI()

def call_mistral_with_retry(mistral_client, model, prompt, max_retries=3, wait_time=2):
    """LLM call with retry/backoff to handle rate limits"""
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
                # If Retry-After header exists, use it
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

@app.post("/analyze")
async def analyze(request: Request):
    logger.info(f"error_detector/main.py - analyze() - Starting error detection analysis")
    
    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_MODEL_CODE")
    logger.debug(f"error_detector/main.py - analyze() - Using model: {model}")
    
    mistral_client = Mistral(api_key=api_key)
    
    log = await request.json()
    logger.debug(f"error_detector/main.py - analyze() - Received log data: {truncate_payload(log)}")
    
    prompt = f"""
        You are an agent specialized in software error analysis based on log files.
        Your task is to identify and summarize all errors present, highlight the most critical or frequent ones,
        recognize recurring patterns, and suggest possible causes and corrective actions.
        Input: a series of logs that may contain error messages, warnings, stack traces, status codes, and other relevant information.
        Goals:
            - Analyze the logs to identify all errors.
            - Highlight the most critical or frequent errors.
            - Identify the main possible causes of the errors.
            - Suggest corrective actions or areas for further investigation.
            - Provide a clear and structured summary of the analysis.

        Respond in a structured format, including:
            - Overall summary of detected errors.
            - Details of the most critical or frequent errors.
            - Possible causes and correlations.
            - Recommendations for resolution or mitigation.

        INPUT: {log}
    """
    
    logger.debug(f"error_detector/main.py - analyze() - Created prompt with length: {len(prompt)}")
    logger.info(f"error_detector/main.py - analyze() - Sending request to Mistral API")
    try:
        # --- Retry logic here ---
        response = call_mistral_with_retry(
            mistral_client,
            model,
            prompt,
            max_retries=3,
            wait_time=2
        )
        logger.debug(f"error_detector/main.py - analyze() - Received response from API")
    except SDKError as e:
        err_msg = str(e)
        logger.warning(f"error_detector/main.py - analyze() - SDK error: {err_msg}")
        
        if "Invalid model" in err_msg or "invalid_model" in err_msg:
            fallback = os.getenv("MISTRAL_LLM_MODEL")
            logger.info(f"error_detector/main.py - analyze() - Falling back to model: {fallback}")
            response = mistral_client.chat.complete(
                model=fallback,
                messages=[{"role": "user", "content": prompt}]
            )
            logger.debug(f"error_detector/main.py - analyze() - Received response from fallback API")
        else:
            logger.error(f"error_detector/main.py - analyze() - Unhandled SDK error: {err_msg}")
            raise

    content = response.choices[0].message.content
    logger.info(f"error_detector/main.py - analyze() - Analysis completed, content length: {len(content)}")
    logger.debug(f"error_detector/main.py - analyze() - Content summary: {truncate_payload(content)}")
    
    return {"error_type": content}
