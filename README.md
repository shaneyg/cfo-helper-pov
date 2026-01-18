CFO Financial Assistant: A Reference Architecture for Financial RAG
The Strategic Context
- This repository serves as a Proof of Concept (PoC) for an enterprise Retrieval-Augmented Generation (RAG) system designed for CFO-level financial analysis.
- The goal of this project was to validate an  infrastructure pattern for a system grounded in Financial Integrity and Audit-Ready Data.

Core Capabilities
- Contextual Financial Intelligence: Analyzes financial statements (PDFs) while preserving structural integrity.
- Identity Hallucination Mitigation: Utilizes MD5 hash-synchronization to anchor vector chunks to specific document metadata, preventing the model from "guessing" executive roles or financial figures.
- Multi-Cloud Ready: Currently hardened for Microsoft Azure (Container Apps + PostgreSQL/pgvector), with architectural patterns transferable to GCP and AWS.

The Technical Stack
- Orchestration: LlamaIndex for high-precision data chunking and retrieval.
- Interface: Streamlit for a lightweight, executive-ready UI.
- Database: Azure PostgreSQL with the pgvector extension for high-dimensional vector storage.

Deployment: Containerized via Docker and deployed through a GitHub Actions CI/CD pipeline to Azure Container Apps.

Why This Pattern?
This "Pattern" was developed to solve the "Three Hurdles of Enterprise AI":
- Security: Data remains within the client's Azure security perimeter.
- Accuracy: RAG logic ensures the model is "grounded" in the uploaded Annual Reviews, not its training data.
- Portability: The Docker-based architecture ensures the solution can be deployed across any modern cloud environment.

Developed by Shane Groeger as a technical stress-test of modern RAG infrastructure patterns.
