from typing import List, Dict
from huggingface_hub import InferenceClient
from src.config.logger import get_logger
from src.config.setup import Config

logger = get_logger(__name__)


class DocumentQA:
    """
    Question-answering system using Hugging Face Inference API.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 api_token: str = None):
        """
        Initialize the QA engine with HuggingFace Inference API.
        
        Args:
            model_name: HuggingFace model name for answer generation
            api_token: HuggingFace API token (uses env var if not provided)
        """
        self.model_name = model_name or Config.LLM_MODEL
        self.api_token = api_token or Config.HUGGING_FACE_TOKEN
        
        if not self.api_token:
            logger.error("No Hugging Face API token provided. Set HUGGING_FACE_TOKEN in .env file")
            raise ValueError("HUGGING_FACE_TOKEN is required. Get one at https://huggingface.co/settings/tokens")
        
        logger.info(f"Initializing DocumentQA with model: {self.model_name}")
        
        try:
            self.client = InferenceClient(
                model=self.model_name,
                token=self.api_token
            )
            logger.info("Successfully initialized HuggingFace Inference Client")
        except Exception as e:
            logger.error(f"Failed to initialize Inference Client: {str(e)}")
            raise
    
    def answer_question(self, 
                       question: str, 
                       retrieved_chunks: List[Dict],
                       max_new_tokens: int = 512,
                       temperature: float = 0.7) -> Dict:
        """
        Generate an answer based on retrieved chunks using the API.
        
        Args:
            question: User's question
            retrieved_chunks: List of relevant chunks from FAISS search
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with answer and metadata
        """
        logger.info(f"Answering question: '{question[:50]}...'")
        logger.debug(f"Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, num_chunks={len(retrieved_chunks)}")
        
        try:
            # Build context from retrieved chunks
            logger.debug("Building context from retrieved chunks")
            context = self._build_context(retrieved_chunks)
            
            # Create messages for chat completion
            messages = self._create_messages(question, context)
            
            # Generate answer via API
            logger.debug("Calling HuggingFace Inference API...")
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated answer with {len(answer)} characters")
            
            return {
                'question': question,
                'answer': answer,
                'sources': [chunk['metadata'] for chunk in retrieved_chunks],
                'num_chunks_used': len(retrieved_chunks)
            }
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Combine retrieved chunks into context string."""
        logger.debug(f"Building context from {len(chunks)} chunks")
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            section = chunk['metadata'].get('section', 'Unknown')
            text = chunk['text']
            context_parts.append(f"[Source {i} - {section}]\n{text}")
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Built context with {len(context)} characters")
        return context
    
    def _create_messages(self, question: str, context: str) -> List[Dict]:
        """Create messages for chat completion API."""
        system_message = """You are a helpful assistant that answers questions based on the provided context. 
Be concise and accurate. If the context doesn't contain enough information to answer the question, say so."""
        
        user_message = f"""Context:
{context}

Question: {question}

Answer:"""
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
