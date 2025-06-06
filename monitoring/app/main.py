import json
import os

from database import SessionLocal, engine
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from models import Base, SystemEvent

from utils.logger import logger, truncate_payload

load_dotenv()

app = FastAPI()

@app.on_event("startup")
async def startup():
    logger.info(f"monitoring/app/main.py - startup() - Initializing database")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"monitoring/app/main.py - startup() - Database initialized successfully")

@app.post("/log")
async def log_entry(request: Request):
    logger.info(f"monitoring/app/main.py - log_entry() - Logging new system event")
    
    data = await request.json()
    logger.debug(f"monitoring/app/main.py - log_entry() - Received event data: {truncate_payload(data)}")
    
    async with SessionLocal() as session:
        entry = SystemEvent(
            source=data.get("source", "unknown"),
            component=data.get("component"),
            log_level=data.get("log_level", "info"),
            event_type=data.get("event_type"),
            details=data.get("details", {}),
            raw_data=json.dumps(data.get("raw_data", {}))
        )
        
        session.add(entry)
        await session.commit()
        logger.info(f"monitoring/app/main.py - log_entry() - Event logged successfully")
    
    return {"status": "logged"}
