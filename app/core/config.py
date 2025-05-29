import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9000"))

# Vector Store Settings
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Get all config as dictionary
def get_settings() -> Dict[str, Any]:
    return {
        "api": {
            "host": API_HOST,
            "port": API_PORT,
        },
        "vector_store": {
            "db_dir": CHROMA_DB_DIR,
            "embedding_model": EMBEDDING_MODEL,
        }
    } 