"""
Streamlit Document Q&A Application
Upload documents and ask questions about their content
"""
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import tempfile
import os
import time

from src.config.logger import get_logger
from src.config.setup import Config
from src.doc_preprocessor import convert_to_docling
from src.chunker import chunk_document
from src.embedder import embed_and_index_chunks, search_chunks
from src.qa_engine import DocumentQA

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, distinctive look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #00D4AA;
        --secondary-color: #7B61FF;
        --accent-color: #FF6B6B;
        --bg-dark: #0A0E17;
        --bg-card: #131A2B;
        --text-primary: #F0F4F8;
        --text-secondary: #8B9DC3;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-dark) 0%, #0D1321 50%, #151D32 100%);
    }
    
    /* Main title styling */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Outfit', sans-serif;
        font-weight: 300;
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .stExpander {
        background: var(--bg-card);
        border: 1px solid rgba(123, 97, 255, 0.2);
        border-radius: 12px;
    }
    
    /* File uploader */
    .stFileUploader {
        background: var(--bg-card);
        border-radius: 12px;
        border: 2px dashed rgba(0, 212, 170, 0.3);
        padding: 1rem;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-card);
        border: 1px solid rgba(123, 97, 255, 0.3);
        border-radius: 8px;
        color: var(--text-primary);
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-dark) 100%);
        border-right: 1px solid rgba(123, 97, 255, 0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.1), rgba(123, 97, 255, 0.1));
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Source card */
    .source-card {
        background: var(--bg-card);
        border-left: 3px solid var(--secondary-color);
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    /* Status indicators */
    .status-ready {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    .status-processing {
        color: var(--secondary-color);
        font-weight: 600;
    }
    
    /* Metrics styling */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(123, 97, 255, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--primary-color);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--primary-color) transparent transparent transparent;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid var(--accent-color);
        border-radius: 8px;
    }
    
    /* Chat history styling */
    .chat-question {
        background: rgba(123, 97, 255, 0.1);
        border-radius: 12px 12px 4px 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--secondary-color);
    }
    
    .chat-answer {
        background: rgba(0, 212, 170, 0.1);
        border-radius: 12px 12px 12px 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--primary-color);
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'embed_model' not in st.session_state:
        st.session_state.embed_model = None
    if 'qa_engine' not in st.session_state:
        st.session_state.qa_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None


def process_document(uploaded_file):
    """Process an uploaded document"""
    logger.info(f"Processing uploaded document: {uploaded_file.name}")
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Step 1: Convert document
        with st.spinner(" Converting document..."):
            logger.info("Step 1: Converting document with Docling")
            doc = convert_to_docling(tmp_path)
            if doc is None:
                st.error("Failed to convert document. Please check the file format.")
                logger.error("Document conversion failed")
                return False
        
        # Step 2: Chunk the document
        with st.spinner("Chunking document..."):
            logger.info("Step 2: Chunking document")
            # Create a result-like object that chunk_document expects
            class DocResult:
                def __init__(self, document):
                    self.document = document
            
            doc_result = DocResult(doc)
            chunks = chunk_document(doc_result, max_tokens=Config.MAX_TOKENS, overlap_tokens=Config.OVERLAP_TOKENS)
            
            if chunks is None or len(chunks) == 0:
                st.error("Failed to chunk document. The document might be empty.")
                logger.error("Document chunking failed or produced no chunks")
                return False
            
            st.session_state.chunks = chunks
            logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings and index
        with st.spinner("Creating embeddings and index..."):
            logger.info("Step 3: Embedding chunks and creating FAISS index")
            
            # Create temp paths for index and metadata
            index_path = tempfile.mktemp(suffix='.bin')
            metadata_path = tempfile.mktemp(suffix='.pkl')
            
            index, model = embed_and_index_chunks(
                chunks,
                model_name=Config.EMBEDDING_MODEL,  
                index_path=index_path,
                metadata_path=metadata_path
            )
            
            st.session_state.faiss_index = index
            st.session_state.embed_model = model
            logger.info("Embeddings and index created successfully")
        
        # Step 4: Initialize QA engine (lazy - only when needed)
        st.session_state.document_processed = True
        st.session_state.document_name = uploaded_file.name
        st.session_state.chat_history = []
        
        logger.info("Document processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        st.error(f"Error processing document: {str(e)}")
        return False
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def answer_question(question: str, top_k: int = 5):
    """Answer a question using the processed document"""
    logger.info(f"Answering question: {question}")
    
    try:
        # Search for relevant chunks
        with st.spinner("Searching relevant sections..."):
            results = search_chunks(
                question,
                st.session_state.faiss_index,
                st.session_state.chunks,
                st.session_state.embed_model,
                top_k=top_k
            )
            logger.info(f"Found {len(results)} relevant chunks")
        
        # Initialize QA engine if not already done
        if st.session_state.qa_engine is None:
            with st.spinner("Connecting to HuggingFace API..."):
                logger.info("Initializing QA engine with HuggingFace API")
                st.session_state.qa_engine = DocumentQA()
        
        # Generate answer
        with st.spinner("Generating answer..."):
            response = st.session_state.qa_engine.answer_question(question, results)
            logger.info("Answer generated successfully")
        
        return response, results
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        st.error(f"Error: {str(e)}")
        return None, None


def render_sidebar():
    """Render the sidebar"""
    with st.sidebar:
        st.markdown("### Settings")
        
        st.caption("☁️ Using HuggingFace Inference API")
        if not Config.HUGGING_FACE_TOKEN:
            st.warning("⚠️ Set HUGGING_FACE_TOKEN in .env file")
        
        st.markdown("---")
        
        # Number of chunks to retrieve
        top_k = st.slider(
            "Number of sources to search",
            min_value=1,
            max_value=10,
            value=5,
            help="How many document sections to consider when answering"
        )
        
        st.markdown("---")
        
        # Document status
        st.markdown("### Status")
        
        if st.session_state.document_processed:
            st.markdown(f"**Document:** {st.session_state.document_name}")
            st.metric("Chunks", len(st.session_state.chunks) if st.session_state.chunks else 0)
            st.metric("Questions Asked", len(st.session_state.chat_history))
            
            if st.button("Clear Document", use_container_width=True):
                st.session_state.document_processed = False
                st.session_state.chunks = None
                st.session_state.faiss_index = None
                st.session_state.embed_model = None
                st.session_state.qa_engine = None
                st.session_state.chat_history = []
                st.session_state.document_name = None
                st.rerun()
        else:
            st.info("No document loaded")
        
        st.markdown("---")
        
        # Info section
        st.markdown("### About")
        st.markdown("""
        This app allows you to:
        1. Upload PDF documents
        2. Ask questions about the content
        3. Get AI-generated answers with sources
        
        **Supported formats:** PDF, DOCX, TXT
        """)
        
        return top_k


def render_chat_history():
    """Render the chat history"""
    if st.session_state.chat_history:
        st.markdown("### Conversation History")
        
        for i, (q, a, sources) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-question">
                <strong>Q:</strong> {q}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-answer">
                <strong>A:</strong> {a}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📚 Sources ({len(sources)} sections)"):
                for j, source in enumerate(sources):
                    section = source['metadata'].get('section', 'Unknown')
                    score = source.get('score', 0)
                    text_preview = source['text'][:200] + "..." if len(source['text']) > 200 else source['text']
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {j+1}</strong> - {section} (Score: {score:.3f})<br>
                        <em>{text_preview}</em>
                    </div>
                    """, unsafe_allow_html=True)


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-title"> Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload documents and ask questions powered by AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    top_k = render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, DOCX, or TXT file to analyze",
            disabled=st.session_state.document_processed
        )
        
        if uploaded_file and not st.session_state.document_processed:
            if st.button("Process Document", use_container_width=True):
                success = process_document(uploaded_file)
                if success:
                    st.success(f"Document processed successfully! Created {len(st.session_state.chunks)} searchable sections.")
                    st.rerun()
    
    with col2:
        st.markdown("### Ask Questions")
        
        if st.session_state.document_processed:
            question = st.text_input(
                "Enter your question",
                placeholder="What is this document about?",
                key="question_input"
            )
            
            if st.button("Get Answer", use_container_width=True) and question:
                start_time = time.time()
                response, sources = answer_question(question, top_k)
                elapsed_time = time.time() - start_time
                
                if response:
                    # Add to chat history
                    st.session_state.chat_history.append((question, response['answer'], sources))
                    
                    # Display answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <strong>Answer:</strong><br>
                        {response['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"Generated in {elapsed_time:.2f}s using {response['num_chunks_used']} sources")
                    
                    # Show sources
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources):
                            section = source['metadata'].get('section', 'Unknown')
                            score = source.get('score', 0)
                            st.markdown(f"**Source {i+1}** - {section} (Relevance: {score:.3f})")
                            st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
                            st.markdown("---")
        else:
            st.info("Please upload and process a document first")
    
    # Chat history at bottom
    st.markdown("---")
    render_chat_history()


if __name__ == "__main__":
    main()
