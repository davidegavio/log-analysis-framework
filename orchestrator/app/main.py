import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from workflow import analyze_log

from utils.logger import logger, truncate_payload

load_dotenv()

app = FastAPI()

@app.post("/process")
async def process_log(request: Request):
    logger.info(f"orchestrator/app/main.py - process_log() - Starting log processing")
    
    log = await request.body()
    logger.debug(f"orchestrator/app/main.py - process_log() - Received log data, size: {len(log)} bytes")
    logger.debug(f"orchestrator/app/main.py - process_log() - Log preview: {truncate_payload(log.decode('utf-8', errors='replace'))}")
    
    logger.info(f"orchestrator/app/main.py - process_log() - Calling analyze_log function")
    result = await analyze_log(log)
    
    logger.info(f"orchestrator/app/main.py - process_log() - Analysis completed")
    logger.debug(f"orchestrator/app/main.py - process_log() - Analysis result: {truncate_payload(result)}")
    
    return result
