# Chroma DB API

A FastAPI-based service for storing, retrieving, and searching document chunks in a Chroma vector database.

## Overview

This project provides an API for:
- Storing document chunks with metadata
- Retrieving specific documents by ID
- Deleting documents
- Semantic search across stored documents

The API uses the Chroma vector database to efficiently store and query document embeddings.

## Requirements

- Python 3.8+
- Required packages listed in `app/requirements.txt`

## Setup

1. Clone the repository:
```bash
git clone https://github.com/logon2390/chroma-db.git
cd chroma-db
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r app/requirements.txt
```

5. Create a `.env` file in the root directory (optional, defaults will be used if not provided):
```
API_HOST=0.0.0.0
API_PORT=9000
CHROMA_DB_DIR=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Running the Application

Start the API server:
```bash
python -m app.main
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --port 9000 --reload
```

The API will be available at http://localhost:9000

## API Endpoints

- **GET /** - Root endpoint with API information
- **GET /health** - Health check endpoint
- **POST /api/v1/store** - Store document chunks
- **GET /api/v1/documents/{doc_id}** - Retrieve a document by ID
- **DELETE /api/v1/documents/{doc_id}** - Delete a document by ID
- **GET /api/v1/search?query={query}&limit={limit}** - Search documents

## API Documentation

After starting the server, visit:
- Swagger UI: http://localhost:9000/docs
- ReDoc: http://localhost:9000/redoc 