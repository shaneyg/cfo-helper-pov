# CFO Helper

## Overview

CFO Helper is a financial document analysis application built with Streamlit. It provides an AI-powered assistant for analyzing financial documents using LlamaIndex for document parsing and retrieval-augmented generation (RAG), with OpenAI's GPT-4o as the underlying language model. The application allows users to upload financial documents, which are parsed, embedded, and stored for intelligent querying.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: Streamlit
- **Layout**: Wide layout with expanded sidebar
- **Purpose**: Provides a web interface for document upload and AI-assisted financial queries

### Backend/AI Pipeline
- **Document Processing**: LlamaParse for parsing uploaded documents (likely PDFs)
- **Vector Store**: LlamaIndex with PGVectorStore for storing document embeddings in PostgreSQL
- **LLM**: OpenAI GPT-4o with temperature 0.0 for deterministic responses
- **Embeddings**: OpenAI text-embedding-3-small model
- **Async Handling**: nest_asyncio applied for async compatibility within Streamlit

### Data Storage
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **Tables**:
  - `documents`: Tracks uploaded files with filename, file hash (for deduplication), and upload timestamp
  - `document_vectors`: Stores vector embeddings for RAG retrieval
- **Connection**: SQLAlchemy with connection pooling (pool_pre_ping enabled)
- **Local Storage**: JSON-based storage in `/storage` directory for document indices (docstore, vector store, graph store)

### Key Design Decisions
1. **File Deduplication**: Uses SHA-256 hash to prevent duplicate document processing
2. **Caching**: Database engine cached with `@st.cache_resource` for connection reuse
3. **Environment Validation**: Application validates all required environment variables before startup
4. **Vector Search**: PostgreSQL with pgvector chosen over external vector databases for simplified infrastructure

## External Dependencies

### APIs & Services
- **OpenAI API**: Required for GPT-4o LLM and text-embedding-3-small embeddings
- **LlamaParse**: Document parsing service for extracting text from financial documents

### Database
- **PostgreSQL**: Primary database with pgvector extension
- **Required Environment Variables**:
  - `DATABASE_URL`: Full connection string
  - `PGDATABASE`, `PGHOST`, `PGPASSWORD`, `PGPORT`, `PGUSER`: Individual connection parameters

### Key Python Packages
- `streamlit`: Web application framework
- `llama-index-core`: Core RAG functionality
- `llama-index-llms-openai`: OpenAI LLM integration
- `llama-index-embeddings-openai`: OpenAI embeddings integration
- `llama-index-vector-stores-postgres`: PostgreSQL vector store
- `llama-parse`: Document parsing
- `sqlalchemy`: Database ORM and connection management
- `nest-asyncio`: Async compatibility layer