# ⚡ Dual-Stage Retrieval-Augmented Generation (RAG) Engine

A bare-metal, local, multi-document semantic search engine built from scratch. 

Unlike standard AI projects that rely on high-level wrappers (like LangChain) and cloud-hosted vector databases (like Pinecone), this project implements a **custom C++ retrieval backend** communicating with a **Python deep-learning frontend** via JSON Inter-Process Communication (IPC). It is capable of indexing and searching over 1.5 million words (50,000+ vector chunks) strictly on local CPU hardware.

---

## 🏗️ System Architecture

The engine utilizes a "Neural-Sparse Fusion" approach, guaranteeing that both highly specific keywords and ambiguous semantic philosophies are retrieved accurately.

### 1. Ingestion & Vectorization (Python)
* **Multi-Document Parsing:** Accepts multiple massive `.txt` files concurrently.
* **Chunking Strategy:** Context-aware newline chunking.
* **Bi-Encoder Generation:** Utilizes HuggingFace's `all-MiniLM-L6-v2` to generate genuine 384-dimensional dense math embeddings for every chunk.
* **Memory Management:** Implements strict `st.session_state` hashing to lock the dataset into memory, preventing redundant `O(N)` neural network calculations on subsequent searches.

### 2. Stage 1: Fast Retrieval (C++ Microservice)
The Python frontend serializes the query vector and corpus into JSON and triggers the compiled C++ binary (`hybrid_engine`).
* **Sparse Index (Lexical):** A custom-built Okapi BM25 index handles exact-keyword matching and term-frequency normalization.
* **Dense Index (Semantic):** A Hierarchical Navigable Small World (HNSW) graph is built dynamically in RAM using `hnswlib`. It computes L2 (Euclidean) distances across the 384D space in $O(\log N)$ time.
* **Reciprocal Rank Fusion (RRF):** The C++ engine mathematically merges the BM25 lexical scores with the HNSW semantic distances to extract the top 60 candidate chunks.

### 3. Stage 2: Deep Reranking (Python)
* **Cross-Encoder Validation:** The top candidates are passed back to Python, where `ms-marco-MiniLM-L-6-v2` evaluates the *exact* contextual relationship between the user's query and the retrieved paragraphs, scoring and sorting the absolute best top 5 results for the UI.

---

## 🛠️ Tech Stack

* **Backend:** C++17, `hnswlib` (Approximate Nearest Neighbors), `nlohmann/json` (IPC).
* **Frontend & ML:** Python 3.9+, Streamlit, SentenceTransformers (PyTorch), Pandas.
* **Optimization:** `-O3` Compiler Flags, Session State Locking.

---

## 🚀 Quick Start (Local Setup)

This repository is designed to be "plug-and-play" for Unix-based systems (Mac/Linux).

**1. Clone the repository**
```bash
git clone [https://github.com/rounaktiwari27/cpp-search-engine.git](https://github.com/rounaktiwari27/cpp-search-engine.git)
cd cpp-search-engine