from pathlib import Path

#Direktori Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_STORAGE_DIR = DATA_DIR / "bps-datapdf"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"
CHUNKS_JSON_PATH = DATA_DIR / "chunks.json"
INDEX_NAME = "index"

#Konfigurasi Model
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"
LLM_REPO_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

#KONFIGURASI CHUNKS
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS_FINAL_FILTER = 512

#Konfigurasi Retriever dan Reranker
RETRIEVER_SEARCH_K = 5
RERANKER_TOP_N = 3
