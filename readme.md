# Log Analysis Project

A sophisticated microservices-based system for analyzing log files using AI agents and RAG (Retrieval-Augmented Generation) techniques to extract insights, detect errors, and analyze performance issues.

## Architecture Overview

The system consists of several microservices, each with a specific responsibility:

### Core Services

- **API Gateway**: Entry point for all client interactions, routing requests to appropriate services
- **Preprocessing Service**: Handles log file ingestion, chunking, and vector embeddings generation
- **Orchestrator**: Coordinates the analysis workflow and delegates tasks to specialized agents
- **RAG Service**: Provides a chat interface for querying log data with context retrieval
- **Chat Service**: Backend for the RAG-powered conversational interface
- **Monitoring Service**: Collects and stores system events and analysis results

### Specialized Agents

- **Error Detector**: Identifies and analyzes errors in log files
- **Performance Analyzer**: Detects performance issues, slowdowns, and resource usage patterns
- **Aggregator**: Consolidates insights from specialized agents into a comprehensive analysis

### Data Storage

- **PostgreSQL**: Stores structured log entries, analysis results, and system events
- **Qdrant**: Vector database for semantic search and retrieval of log chunks

## Key Features

- **AI-Powered Analysis**: Uses Mistral AI models for error detection, performance analysis, and natural language understanding
- **Vector Search**: Implements RAG techniques for context-aware conversations about log data
- **Distributed Processing**: Microservices architecture with containerized components
- **Workflow Orchestration**: LangGraph-based workflow for coordinated analysis
- **Interactive Dashboard**: Streamlit-based UI for log visualization and chat interaction

## Technical Stack

- **Backend**: FastAPI and Python for all microservices
- **Frontend**: Streamlit for the dashboard and chat interface
- **Databases**: PostgreSQL (structured data) and Qdrant (vector database)
- **AI Models**: Mistral AI for embeddings and text generation
- **Containerization**: Docker and Docker Compose
- **Communication**: REST APIs between services

## Service Details

### API Gateway

Entry point for all client requests, routing to appropriate services:
- `/ingest`: Forwards log files to the preprocessing service
- `/chat`: Routes chat requests to the RAG service

### Preprocessing Service

Handles log file ingestion and preparation:
- Splits logs into chunks
- Generates embeddings using Mistral AI
- Stores vector embeddings in Qdrant
- Forwards logs to the orchestrator for analysis

### Orchestrator

Coordinates the analysis workflow using LangGraph:
- Analyzes logs to determine which agents to call
- Dispatches logs to specialized agents
- Aggregates analysis results
- Reports to monitoring service

### RAG Service

Provides an interactive UI for log analysis:
- Chat interface for querying log data
- Log file upload
- Database visualization
- Vector store exploration

### Chat Service

Backend for RAG-based conversations:
- Retrieves relevant context from vector stores
- Generates answers using Mistral AI
- Maintains conversation history

### Monitoring Service

Tracks system events and stores analysis results:
- Records events from all services
- Stores structured data in PostgreSQL
- Provides access to historical analysis

### Specialized Agents

Perform targeted analysis on log data:
- **Error Detector**: Identifies errors, exceptions, and failures
- **Performance Analyzer**: Detects slowdowns, timeouts, and resource issues
- **Aggregator**: Combines and synthesizes analysis from other agents

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Mistral AI API key (set in `.env` file)

### Environment Setup

Create a `.env` file with the following variables:

```
# API Keys
MISTRAL_API_KEY=your_mistral_api_key

# Models
MISTRAL_EMBED_MODEL=mistral-embed
MISTRAL_MODEL_LARGE=mistral-large-latest
MISTRAL_MODEL_CODE=mistral-large-latest
MISTRAL_MODEL_PERF=mistral-large-latest

# Service URLs
API_GATEWAY_URL=http://api_gateway:8000
PREPROCESSING_URL=http://preprocessing:8002/preprocess
ORCHESTRATOR_URL=http://orchestrator:8001/process
RAG_CHAT_URL=http://chat:8008/chat

# Vector DB
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=preprocessed_files
QA_COLLECTION_NAME=chat_conversations
VECTOR_SIZE=1024
DISTANCE=Cosine
CHUNK_SIZE=1024

# Database
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/logs
SYNC_DATABASE_URL=postgresql+psycopg2://user:password@db:5432/logs
```

### Running the System

Start all services using Docker Compose:

```bash
docker-compose up -d
```

### Accessing the UI

The RAG service UI is available at: http://localhost:8004

## Usage Examples

### Uploading a Log File

1. Navigate to the RAG service UI (http://localhost:8004)
2. Use the chat input to upload a log file
3. The system will preprocess the file, analyze it, and provide an initial summary

### Querying Log Data

After uploading logs, you can ask questions about them:
- "What errors are present in the logs?"
- "Are there any performance issues?"
- "Show me all authentication failures"
- "What are the most critical issues found?"

## Workflow Details

1. User uploads a log file through the RAG service UI
2. API Gateway forwards the file to the Preprocessing Service
3. Preprocessing Service:
   - Chunks the file
   - Generates embeddings
   - Stores vectors in Qdrant
   - Sends the file to the Orchestrator
4. Orchestrator:
   - Uses AI to determine which agents to call
   - Dispatches logs to Error Detector and/or Performance Analyzer
   - Collects analysis results
   - Sends aggregated results to the Aggregator agent
5. Results are returned to the user through the UI

## Contributing

To contribute to this project:

1. Clone the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.