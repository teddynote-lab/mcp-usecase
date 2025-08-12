from pathlib import Path

# Path settings
DATA_DIR = Path(__file__).parent.parent / "data"
VECTOR_DIR = Path(__file__).parent / "vector_db"

# Default settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
