# ⚡ Neural-Sparse Fusion Engine (Dual-Stage RAG)

![C++17](https://img.shields.io/badge/C++-17-blue.svg?style=for-the-badge&logo=c%2B%2B)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-yellow.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?style=for-the-badge&logo=streamlit)
![Architecture](https://img.shields.io/badge/Architecture-Local_Bare_Metal-success.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A high-performance, multi-document semantic search engine built entirely from scratch.

This project intentionally bypasses high-level AI wrappers (like LangChain) and cloud-hosted vector databases (like Pinecone) to implement a **custom C++ retrieval microservice** communicating with a **Python deep-learning frontend** via JSON Inter-Process Communication (IPC). It is designed to demonstrate full-stack systems engineering, memory management, and algorithm optimization on local CPU hardware.

---

## System Architecture & Data Flow

Standard keyword searches fail to understand context, while standard vector searches often miss hyper-specific nouns. This architecture guarantees that both specific lexical keywords and ambiguous semantic queries are retrieved accurately before being reranked by a deep neural network.

```mermaid
graph TD
    classDef python fill:#3776AB,stroke:#fff,stroke-width:2px,color:#fff;
    classDef cpp fill:#00599C,stroke:#fff,stroke-width:2px,color:#fff;
    classDef data fill:#FF9900,stroke:#fff,stroke-width:2px,color:#fff;

    A[User Search Query] -->|Raw Text| B(Python Bi-Encoder: all-MiniLM-L6-v2)
    B -->|Generates 384D Vector| C[(data/query.json)]
    
    C -->|JSON IPC Handoff| D{C++ Hybrid Engine}
    
    D -->|Lexical Path| E[Okapi BM25 Index]
    D -->|Semantic Path| F[HNSW Graph / hnswlib]
    E -->|Keyword Match| G[Reciprocal Rank Fusion]
    F -->|L2 Euclidean Dist| G
    
    G -->|Top 60 Candidates| H[(C++ stdout)]
    
    H -->|JSON IPC Handoff| I{Python: Stage 2 Reranking}
    
    I -->|Pairs: Query + Chunk| J[Cross-Encoder: ms-marco-MiniLM]
    J -->|Contextual Scoring| K[Top 5 Results]
    
    K --> L[Streamlit UI Dashboard]

    class A,B,I,J,K,L python;
    class D,E,F,G,H cpp;
    class C data;
```

---

## Technical Approach

### 1. Memory-Locked Ingestion
Python reads massive `.txt` files, generates 384-dimensional mathematical vectors using a HuggingFace Bi-Encoder, and locks the dataset into RAM to prevent redundant processing.

### 2. The IPC Bridge
Python serializes the user's query into JSON and hands it off to a compiled C++ binary running in a completely separate memory space.

### 3. Dual-Stage C++ Retrieval
The C++ microservice runs the query against two distinct indexes simultaneously:

- **Sparse Index (BM25):** Hunts for exact keyword matches (e.g., character names, dates).
- **Dense Index (HNSW):** Navigates a mathematical graph to find semantic "vibes" and context.

### 4. Mathematical Fusion (RRF)
C++ combines the scores from both indexes using Reciprocal Rank Fusion and sends the top candidates back to Python.

### 5. Deep Reranking
A Python Cross-Encoder neural network reads the top candidates, evaluates their exact narrative relationship to the query, and outputs the absolute best matches to the UI.

---

## Algorithmic Engine (Under the Hood)

### The Sparse Index (BM25)
Implemented in C++, this handles exact-keyword matching and penalizes overly common stop-words using Term Frequency-Inverse Document Frequency (TF-IDF) logic.

### The Dense Index (HNSW Graph)
Semantic similarity is calculated using a Hierarchical Navigable Small World (HNSW) graph dynamically constructed in RAM via `hnswlib`. Distances are computed using the L2 (Euclidean) metric across a 384-dimensional space:

$$d(p, q) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2}$$

### Reciprocal Rank Fusion (RRF)
To bridge the "Lexical Gap," the C++ engine mathematically merges the BM25 and HNSW results using the RRF algorithm:

$$RRF = \frac{1}{k + R_{BM25}} + \frac{1}{k + R_{HNSW}}$$

---

## JSON IPC Protocol (Internal API)

Because Python and C++ run in completely separate memory spaces, they communicate strictly through serialized JSON.

**Python → C++ Payload** (`data/query.json`):
```json
{
  "text": "the existential realization that nature is indifferent",
  "vector": [0.012, -0.045, 0.881, 0.334]
}
```

**C++ → Python Response** (stdout):
```json
[
  {
    "final_rank": 1,
    "rrf_score": 0.0315,
    "bm25_rank": 45,
    "hnsw_rank": 2,
    "text": "Before him is a dead sea that stretches in azure calm..."
  }
]
```

---

## Quick Start (Local Setup Guide)

### 1. Clone the repository
```bash
git clone https://github.com/rounaktiwari27/cpp-search-engine.git
cd cpp-search-engine
```

### 2. Run the automated build script
This script isolates the Python environment, installs dependencies, and compiles the C++ microservice with `-O3` optimization for maximum CPU performance.
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Launch the UI Dashboard
```bash
source venv/bin/activate
streamlit run app.py
```

---

## Architectural Evaluation & Roadmap

### System Merits

- **Flawless Contextual Recall:** The dense vector HNSW graph maps abstract themes to contextual paragraphs without relying on exact keywords.
- **Resilient Memory Management:** Dynamic `st.session_state` locking allows massive multi-document ingestion without crashing the Streamlit UI.
- **Blazing Fast Local Execution:** The C++ binary executes $O(\log N)$ spatial navigation strictly on local CPU hardware.

### Engineering Bottlenecks (Future Scaling)

- **Dynamic Graph Latency:** The C++ engine currently rebuilds the HNSW graph dynamically in RAM from scratch for every search. Production systems must serialize this graph to a `.bin` file for $O(1)$ memory loading to achieve sub-second latency.
- **Model Dimensionality Limits:** The system uses a 384-dimensional MiniLM model to allow local CPU execution. Enterprise scaling requires upgrading to a 1536D embedding model.
- **Cloud Hosting Constraints:** Deploying this specific architecture to a 512MB free-tier cloud server (e.g., Vercel) will trigger an Out-Of-Memory (OOM) crash due to the heavy RAM requirement of local vector graph construction.