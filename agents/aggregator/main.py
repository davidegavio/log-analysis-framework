import os
import time

from dotenv import load_dotenv

load_dotenv()

import json
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from mistralai import Mistral

from utils.logger import logger, truncate_payload

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
async def aggregate_analysis(request: Request) -> Dict[str, Any]:
    """Endpoint for aggregated analysis with Mistral"""
    logger.info("aggregator/main.py - aggregate_analysis() - Starting aggregation request")
    
    try:
        # Retrieve data from request
        request_data = await request.json()
        logger.debug(f"aggregator/main.py - Received request data: {truncate_payload(request_data)}")
        
        # Extract main components
        batch_results = request_data.get("batch_results", [])
        total_batches = request_data.get("total_batches", len(batch_results))
        metadata = request_data.get("metadata", {})

        # Initialize the Mistral client
        api_key = os.getenv("MISTRAL_API_KEY")
        model = os.getenv("MISTRAL_MODEL_LARGE")
        mistral_client = Mistral(api_key=api_key)
        
        # Build context for the LLM
        context = "\n".join([
            f"Batch {idx+1}:\n{json.dumps(res, ensure_ascii=False)}"
            for idx, res in enumerate(batch_results)
        ])

        # Build the prompt
        prompt = f"""
        You are an expert log analyst. Synthesize the following results obtained from specialized agents:

        {context}

        Produce a final report in English that includes:
        1. General overview of identified issues
        2. Classification of errors by severity
        3. Recurring patterns
        4. Priority operational recommendations
        5. Suggestions for further investigation

        Format the report in Markdown with appropriate headings.
        """

        # Call Mistral API
        logger.info("aggregator/main.py - Calling Mistral API for analysis")
        # --- Retry logic here ---
        response = call_mistral_with_retry(
            mistral_client,
            model,
            prompt,
            max_retries=3,
            wait_time=2
        )

        if not response.choices:
            logger.warning("aggregator/main.py - Empty response from Mistral API")
            raise HTTPException(status_code=500, detail="Empty response from AI model")

        analysis = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.info("aggregator/main.py - Analysis completed successfully")
        logger.debug(f"aggregator/main.py - Generated analysis: {truncate_payload(analysis)}")
        
        return {
            "status": "success",
            "analysis": analysis,
            "metadata": {
                "total_batches": total_batches,
                "model": model,
                "tokens_used": tokens_used,
                **metadata
            }
        }
    except json.JSONDecodeError as e:
        logger.error(f"aggregator/main.py - JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e

    except HTTPException:
        raise  # Re-raise already handled exceptions

    except Exception as e:
        logger.error(f"aggregator/main.py - Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}") from e
