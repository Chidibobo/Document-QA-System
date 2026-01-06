from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict
from src.config.logger import get_logger
from src.config.setup import Config

logger = get_logger(__name__)


def embed_and_index_chunks(chunks: List[Dict], 
                           model_name: str = None,
                           index_path: str = None,
                           metadata_path: str = None):
    # Use config defaults if not provided
    model_name = model_name or Config.EMBEDDING_MODEL
    index_path = index_path or Config.FAISS_INDEX_PATH
    metadata_path = metadata_path or Config.CHUNK_METADATA_PATH
    """
    Embed document chunks and create a FAISS index for retrieval.
    
    Args:
        chunks: List of chunk dicts from chunk_document()
        model_name: HuggingFace embedding model name
        index_path: Where to save the FAISS index
        metadata_path: Where to save chunk metadata
    
    Returns:
        tuple: (faiss_index, embedding_model)
    """
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.debug(f"Successfully loaded embedding model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
        raise
    
    # Extract just the text from chunks
    texts = [chunk['text'] for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    
    try:
        # Encode all chunks into embeddings
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        logger.debug(f"Successfully embedded {len(texts)} chunks")
    except Exception as e:
        logger.error(f"Failed to embed chunks: {str(e)}")
        raise
    
    # Get embedding dimension
    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Create FAISS index
    # Using IndexFlatIP for cosine similarity 
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    logger.info(f"Added {index.ntotal} vectors to FAISS index")
    
    # Save FAISS index
    try:
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {index_path}: {str(e)}")
        raise
    
    # Save chunk metadata 
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved chunk metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save chunk metadata to {metadata_path}: {str(e)}")
        raise
    
    return index, model


def load_index_and_metadata(index_path: str = None,
                            metadata_path: str = None,
                            model_name: str = None):
    """
    Load a saved FAISS index and its metadata.
    
    Returns:
        tuple: (faiss_index, chunks, embedding_model)
    """
    # Use config defaults if not provided
    index_path = index_path or Config.FAISS_INDEX_PATH
    metadata_path = metadata_path or Config.CHUNK_METADATA_PATH
    model_name = model_name or Config.EMBEDDING_MODEL
    logger.info(f"Loading FAISS index from {index_path}")
    try:
        # Load FAISS index
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {index_path}: {str(e)}")
        raise
    
    # Load chunk metadata
    try:
        with open(metadata_path, 'rb') as f:
            chunks = pickle.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to load chunk metadata from {metadata_path}: {str(e)}")
        raise
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.debug(f"Successfully loaded embedding model")
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
        raise
    
    return index, chunks, model


def search_chunks(query: str, 
                 index: faiss.Index,
                 chunks: List[Dict],
                 model: SentenceTransformer,
                 top_k: int = 5):
    """
    Search for relevant chunks given a query.
    
    Args:
        query: The search query/question
        index: FAISS index
        chunks: List of chunk metadata
        model: Embedding model
        top_k: Number of results to return
    
    Returns:
        List of relevant chunks with scores
    """
    logger.info(f"Searching for relevant chunks for query: '{query[:50]}...'")
    logger.debug(f"Search parameters: top_k={top_k}")
    
    try:
        # Embed the query
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        logger.debug("Query embedding generated successfully")
    except Exception as e:
        logger.error(f"Failed to embed query: {str(e)}")
        raise
    
    # Search FAISS index
    try:
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        logger.debug(f"FAISS search completed, found {len(indices[0])} results")
    except Exception as e:
        logger.error(f"FAISS search failed: {str(e)}")
        raise
    
    # Get the actual chunks
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'chunk': chunks[idx],
            'score': float(score),
            'text': chunks[idx]['text'],
            'metadata': chunks[idx]['metadata']
        })
    
    logger.info(f"Retrieved {len(results)} chunks with scores ranging from {results[-1]['score']:.4f} to {results[0]['score']:.4f}")
    return results


# Example usage:
if __name__ == "__main__":
    # Assuming you have chunks from the previous chunking function
    # chunks = chunk_document(doc_result, max_tokens=512)
    
    # Step 1: Embed and index (do this once per document)
    # index, model = embed_and_index_chunks(
    #     chunks,
    #     model_name="BAAI/bge-large-en-v1.5"
    # )
    
    # Step 2: Load saved index (for subsequent queries)
    # index, chunks, model = load_index_and_metadata()
    
    # Step 3: Search
    # query = "What is the main topic of this document?"
    # results = search_chunks(query, index, chunks, model, top_k=3)
    # 
    # for i, result in enumerate(results, 1):
    #     print(f"\n=== Result {i} (Score: {result['score']:.4f}) ===")
    #     print(f"Section: {result['metadata']['section']}")
    #     print(f"Text: {result['text'][:200]}...")
    
    pass