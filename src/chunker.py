from typing import List, Dict
import re
from src.config.logger import get_logger
logger = get_logger(__name__)

from transformers import AutoTokenizer

# Initialize tokenizer once at the top
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

def chunk_document(doc_result, max_tokens: int = 512, overlap_tokens: int = 50) -> List[Dict]:
    """
    Chunk a Docling document result using semantic boundaries with token limits.
    
    Args:
        doc_result: The result from DocumentConverter.convert()
        max_tokens: Maximum tokens per chunk (default: 512)
        overlap_tokens: Token overlap between chunks (default: 50)
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    try:
        logger.info(f"Attempting to Chunk Docling Document")
        # Export to markdown to preserve structure
        full_text = doc_result.document.export_to_markdown()

        if full_text:
            logger.info(f"Splitting texts")
            # Split by headers first (semantic boundaries)
            sections = split_by_headers(full_text)
            
            for section_idx, section in enumerate(sections):
                section_text = section['text']
                section_header = section['header']
                
                # If section is small enough, keep it as one chunk
                if estimate_tokens(section_text) <= max_tokens:
                    chunks.append({
                        'text': section_text,
                        'metadata': {
                            'section': section_header,
                            'chunk_index': len(chunks),
                            'section_index': section_idx
                        }
                    })
                else:
                    # Split large sections by paragraphs
                    paragraphs = section_text.split('\n\n')
                    current_chunk = ""
                    
                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue
                        
                        test_chunk = current_chunk + '\n\n' + para if current_chunk else para
                        
                        if estimate_tokens(test_chunk) <= max_tokens:
                            current_chunk = test_chunk
                        else:
                            # Save current chunk if it exists
                            if current_chunk:
                                chunks.append({
                                    'text': current_chunk,
                                    'metadata': {
                                        'section': section_header,
                                        'chunk_index': len(chunks),
                                        'section_index': section_idx
                                    }
                                })
                            
                            # Handle paragraphs that are too long on their own
                            if estimate_tokens(para) > max_tokens:
                                # Split by sentences as fallback
                                sub_chunks = split_by_sentences(para, max_tokens)
                                for sub_chunk in sub_chunks:
                                    chunks.append({
                                        'text': sub_chunk,
                                        'metadata': {
                                            'section': section_header,
                                            'chunk_index': len(chunks),
                                            'section_index': section_idx
                                        }
                                    })
                                current_chunk = ""
                            else:
                                current_chunk = para
                    
                    # Don't forget the last chunk
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk,
                            'metadata': {
                                'section': section_header,
                                'chunk_index': len(chunks),
                                'section_index': section_idx
                            }
                        })
            
            # Add overlap between chunks
            if overlap_tokens > 0:
                chunks = add_overlap(chunks, overlap_tokens)
    
        return chunks
    except Exception as e:
        logger.error(f"Error Chunking text: {str(e)}")
        return None    


def split_by_headers(text: str) -> List[Dict]:
    """Split text by markdown headers while preserving structure."""
    logger.debug("Splitting text by markdown headers")
    
    sections = []
    lines = text.split('\n')
    current_section = {'header': 'Introduction', 'text': ''}
    
    for line in lines:
        # Check if line is a header (markdown style)
        if line.startswith('#'):
            # Save previous section
            if current_section['text'].strip():
                sections.append(current_section)
            
            # Start new section
            header_text = line.lstrip('#').strip()
            current_section = {'header': header_text, 'text': ''}
        else:
            current_section['text'] += line + '\n'
    
    # Add the last section
    if current_section['text'].strip():
        sections.append(current_section)
    
    result = sections if sections else [{'header': 'Document', 'text': text}]
    logger.debug(f"Split text into {len(result)} sections")
    return result


def split_by_sentences(text: str, max_tokens: int) -> List[str]:
    """Split text by sentences when paragraphs are too long."""
    logger.debug(f"Splitting long paragraph by sentences (max_tokens={max_tokens})")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
        
        if estimate_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.debug(f"Split paragraph into {len(chunks)} sentence-based chunks")
    return chunks


def add_overlap(chunks: List[Dict], overlap_tokens: int) -> List[Dict]:
    """Add overlapping text between consecutive chunks."""
    logger.debug(f"Adding overlap of {overlap_tokens} tokens between {len(chunks)} chunks")
    if len(chunks) <= 1:
        logger.debug("Only one chunk, skipping overlap")
        return chunks
    
    overlapped_chunks = []
    
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        
        # Add overlap from previous chunk
        if i > 0:
            prev_text = chunks[i-1]['text']
            prev_words = prev_text.split()
            overlap_size = min(overlap_tokens, len(prev_words))
            overlap = ' '.join(prev_words[-overlap_size:])
            text = f"...{overlap}\n\n{text}"
        
        overlapped_chunks.append({
            'text': text,
            'metadata': chunk['metadata']
        })
    
    logger.debug(f"Added overlap to {len(overlapped_chunks)} chunks")
    return overlapped_chunks


def estimate_tokens(text: str) -> int:
    """Count tokens using the actual model tokenizer."""
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    logger.debug(f"Estimated {token_count} tokens for text of {len(text)} characters")
    return token_count


