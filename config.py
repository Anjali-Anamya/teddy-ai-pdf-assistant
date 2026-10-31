
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# PDF directory
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "data1.pdf"

# ChromaDB directory
CHROMA_PATH = BASE_DIR / "chroma_db"

# Gemini models
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.8-flash"

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever configuration
RETRIEVER_K = 4
RETRIEVER_FETCH_K = 10

# LLM configuration
TEMPERATURE = 0
