# Dual Stage RAG: C++ HNSW + Python Cross-Encoder

A local, multi-document semantic search engine. This project bypasses standard LangChain wrappers and cloud vector databases to implement a bare-metal retrieval architecture. 

It handles dual-stage retrieval by bridging a C++ HNSW graph/BM25 index with a Python deep-learning Reranker via JSON IPC, capable of searching over 1.5 million words (50,000+ vector chunks) on local CPU hardware.

## Architecture

The system operates in two distinct phases:

**Stage 1: Retrieval (C++)**
* **Sparse Index:** Okapi BM25 implementation for exact-keyword matching.
* **Dense Index:** Custom Hierarchical Navigable Small World (HNSW) graph built dynamically in RAM using `hnswlib`. 
* **Fusion:** Reciprocal Rank Fusion (RRF) combines the sparse and dense scores to pull the top `k` candidates.

**Stage 2: Reranking (Python)**
* **Embedding:** HuggingFace `all-MiniLM-L6-v2` (384D) generates initial chunk vectors.
* **Cross-Encoder:** `ms-marco-MiniLM-L-6-v2` evaluates the exact semantic relationship between the query and the top candidates returned by the C++ engine, scoring and sorting the final output.

## Prerequisites
* Python 3.9+
* Clang or GCC compiler (supporting C++17)
* Mac/Linux environment (or WSL on Windows)

## Quick Start (Local Setup)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rounaktiwari27/cpp-search-engine.git](https://github.com/rounaktiwari27/cpp-search-engine.git)
   cd cpp-search-engine