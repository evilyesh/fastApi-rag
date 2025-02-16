# RAG (Retrieval-Augmented Generation) Demo Project

This project demonstrates a basic RAG implementation using:
- **Llama.cpp** for text generation/embeddings
- **ChromaDB** for vector storage/retrieval
- **FastAPI** backend with WebSocket support
- Simple HTML/JS frontend

## Core Components

### `app.py` (FastAPI Server)
```
Handles:
- File uploads (`.txt` only) → chunking → ChromaDB storage
- WebSocket chat interface
- Static file serving
```

### `LlamaHelper` Class
```
Manages:
- Text chunking (basic sentence/paragraph splitting)
- Document embedding storage in ChromaDB
- Context-aware response generation using:
  1. ChromaDB similarity search
  2. Llama.cpp API for LLM completion
```

### `ChromaDBClient` Class
```
Wrapper for:
- Collection management
- Document CRUD operations
- Vector similarity queries
```

### Frontend (`index.html`)
```
Features:
- Real-time chat interface
- Drag-n-drop file upload
- Progress indicators
- Ctrl+Enter support
```

## Workflow
1. User uploads document → split into chunks
2. Chunks stored in ChromaDB with embeddings
3. Chat queries:
   - Find relevant chunks via semantic search
   - Augment LLM prompt with context
   - Generate final response

## Usage Example
```python
# Start server
uvicorn.run(app, host="0.0.0.0", port=8000)

# Usage flow:
1. Upload text file via /upload
2. Chat via WebSocket (/ws)
3. Responses generated using document context
```

> **Note**: Requires running Llama.cpp server separately on port 8989