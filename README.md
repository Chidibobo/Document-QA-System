# 📚 Document Q&A System

A RAG (Retrieval-Augmented Generation) powered document question-answering system. Upload PDFs, DOCX, or TXT files and ask questions about their content using AI.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Document Processing**: Supports PDF, DOCX, and TXT files via Docling
- **Smart Chunking**: Semantic chunking with token-aware splitting and overlap
- **Vector Search**: FAISS-powered similarity search for relevant context retrieval
- **AI Answers**: Uses HuggingFace Inference API for answer generation (no GPU required)
- **Source Citations**: See which document sections were used to generate answers
- **Modern UI**: Clean Streamlit interface with dark theme

## 🏗️ Architecture

```
Document → Docling Parser → Chunker → Embeddings → FAISS Index
                                                        ↓
                    Question → Embed → Search → Context + LLM → Answer
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Document-QA-System.git
cd Document-QA-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
# Required: Get your token at https://huggingface.co/settings/tokens
HUGGING_FACE_TOKEN=hf_your_token_here

# Optional: Model configurations (defaults shown)
LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Storage paths
FAISS_INDEX_PATH=data/vector_store/faiss_index.bin
CHUNK_METADATA_PATH=data/vector_store/chunk_metadata.pkl

# Optional: Chunking settings
MAX_TOKENS=512
OVERLAP_TOKENS=50
```

### 4. Run the app

```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
Document-QA-System/
├── src/
│   ├── app.py              # Streamlit web application
│   ├── doc_preprocessor.py # Document conversion (Docling)
│   ├── chunker.py          # Text chunking with semantic boundaries
│   ├── embedder.py         # Embedding generation & FAISS indexing
│   ├── qa_engine.py        # Question answering via HuggingFace API
│   └── config/
│       ├── setup.py        # Configuration management
│       └── logger.py       # Logging configuration
├── data/
│   ├── pdfs/               # Store your PDF files
│   └── vector_store/       # FAISS index storage
├── logs/                   # Application logs
├── requirements.txt
├── .env                    # Your configuration (create this)
└── README.md
```

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGING_FACE_TOKEN` | - | **Required.** Your HuggingFace API token |
| `LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | LLM for answer generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for text embeddings |
| `FAISS_INDEX_PATH` | `data/vector_store/faiss_index.bin` | Where to save the vector index |
| `CHUNK_METADATA_PATH` | `data/vector_store/chunk_metadata.pkl` | Where to save chunk metadata |
| `MAX_TOKENS` | `512` | Maximum tokens per chunk |
| `OVERLAP_TOKENS` | `50` | Token overlap between chunks |

## 📖 Usage

### Web Interface

1. **Upload**: Click "Upload Document" and select a PDF, DOCX, or TXT file
2. **Process**: Click "Process Document" to extract, chunk, and index the content
3. **Ask**: Type your question and click "Get Answer"
4. **Review**: View the AI-generated answer and source citations

### Programmatic Usage

```python
from src.doc_preprocessor import convert_to_docling
from src.chunker import chunk_document
from src.embedder import embed_and_index_chunks, search_chunks
from src.qa_engine import DocumentQA

# 1. Convert document
doc = convert_to_docling("path/to/document.pdf")

# 2. Chunk the document
class DocResult:
    def __init__(self, document):
        self.document = document

chunks = chunk_document(DocResult(doc), max_tokens=512)

# 3. Create embeddings and index
index, model = embed_and_index_chunks(chunks)

# 4. Search for relevant chunks
results = search_chunks("What is this document about?", index, chunks, model)

# 5. Generate answer
qa = DocumentQA()
response = qa.answer_question("What is this document about?", results)
print(response['answer'])
```

## 🔍 How It Works

### 1. Document Processing
Documents are parsed using [Docling](https://github.com/DS4SD/docling), which extracts text while preserving structure from PDFs, DOCX, and other formats.

### 2. Chunking
Text is split into chunks using semantic boundaries:
- Splits on markdown headers first
- Falls back to paragraph boundaries
- Sentence-level splitting for very long paragraphs
- Configurable token limits with overlap for context continuity

### 3. Embedding & Indexing
Chunks are embedded using sentence-transformers and stored in a FAISS index for fast similarity search.

### 4. Retrieval & Generation
When you ask a question:
1. Your question is embedded using the same model
2. FAISS finds the most similar chunks
3. Relevant chunks are sent as context to the LLM
4. The LLM generates an answer based on the context

## 🛠️ Troubleshooting

### "No module named 'src'"
Make sure you're running from the project root:
```bash
cd Document-QA-System
streamlit run src/app.py
```

### "HUGGING_FACE_TOKEN is required"
Create a `.env` file with your HuggingFace token. Get one at https://huggingface.co/settings/tokens

### Slow document processing
- For text-based PDFs, OCR is disabled by default
- If you have scanned PDFs, modify `convert_to_docling(path, use_ocr=True)`

### Embedding model errors
If using `nvidia/llama-embed-nemotron-8b`, the code automatically enables `trust_remote_code=True`. For faster processing, use the default `sentence-transformers/all-MiniLM-L6-v2`.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
