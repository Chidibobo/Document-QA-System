from dotenv import load_dotenv
load_dotenv(override=True)
import os

class Config:
    # Hugging Face API Configuration
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    
    # Model configurations
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Storage paths
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/vector_store/faiss_index.bin")
    CHUNK_METADATA_PATH = os.getenv("CHUNK_METADATA_PATH", "data/vector_store/chunk_metadata.pkl")
    
    # Chunking settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
    OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", "50"))
