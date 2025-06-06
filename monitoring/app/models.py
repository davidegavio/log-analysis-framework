from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SystemEvent(Base):
    __tablename__ = "system_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(50), nullable=False)  # Es: 'orchestrator', 'agent:error_detector'
    component = Column(String(50))  # Es: 'reasoning', 'action', 'aggregator'
    log_level = Column(String(20))  # Es: 'debug', 'info', 'error'
    event_type = Column(String(50))  # Es: 'log_analysis', 'agent_call', 'system_alert'
    details = Column(JSON)  # Campo generico per tutti i dati
    raw_data = Column(Text)  # Campo per dump completo non strutturato
