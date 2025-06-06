import asyncio
import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from mistralai import Mistral

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

def extract_json(text):
    # Find the first JSON object between braces
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    logger.warning(f"performance_analyzer/main.py - extract_json() - No JSON object found in text: {truncate_payload(text)}")
    raise ValueError("No JSON object found in the text.")

@app.post("/analyze")
async def analyze(request: Request):
    logger.info(f"performance_analyzer/main.py - analyze() - Starting performance analysis")
    
    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_MODEL_PERF")
    logger.debug(f"performance_analyzer/main.py - analyze() - Using model: {model}")
    
    mistral_client = Mistral(api_key=api_key)
    
    log = await request.json()
    logger.debug(f"performance_analyzer/main.py - analyze() - Received log data: {truncate_payload(log)}")

    prompt = f"""
        You are an agent specialized in software performance analysis based on system and application logs.
        Your task is to identify performance issues such as slowdowns, timeouts, excessive resource usage (CPU, memory, I/O), bottlenecks, and anomalies.
        Input: a series of logs that may contain error messages, warnings, performance information, time metrics, and other relevant data.
        Goals:
            - Analyze the logs to identify patterns or events indicating performance issues.
            - Highlight the most probable causes of slowdowns or malfunctions.
            - Suggest possible corrective actions or areas for further investigation.
            - Provide a clear and detailed summary of the analysis.

        Respond in a structured format, including:
            - Overall summary of detected performance insights.
            - Details of critical or anomalous events.
            - Possible causes and correlations.
            - Recommendations for performance improvements.

        INPUT: {log}
    """
    
    logger.debug(f"performance_analyzer/main.py - analyze() - Created prompt with length: {len(prompt)}")
    logger.info(f"performance_analyzer/main.py - analyze() - Sending request to Mistral API")
      # --- Retry logic here ---
    response = call_mistral_with_retry(
        mistral_client,
        model,
        prompt,
        max_retries=3,
        wait_time=2
    )
    
    logger.debug(f"performance_analyzer/main.py - analyze() - Received response from API")
    content = response.choices[0].message.content
    
    logger.info(f"performance_analyzer/main.py - analyze() - Analysis completed, content length: {len(content)}")
    logger.debug(f"performance_analyzer/main.py - analyze() - Content summary: {truncate_payload(content)}")
    
    return {"perf_issue": content}
