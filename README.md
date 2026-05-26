# ⚡ Neural-Sparse Fusion Engine (Dual-Stage RAG)

![C++17](https://img.shields.io/badge/C++-17-blue.svg?style=for-the-badge&logo=c%2B%2B)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-yellow.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?style=for-the-badge&logo=streamlit)
![Architecture](https://img.shields.io/badge/Architecture-Local_Bare_Metal-success.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A high-performance, multi-document semantic search engine built from scratch. 

This project intentionally bypasses high-level wrappers (like LangChain) and cloud-hosted vector databases (like Pinecone) to implement a **custom C++ retrieval backend** communicating with a **Python deep-learning frontend** via JSON Inter-Process Communication (IPC). It is designed to demonstrate full-stack systems engineering, capable of indexing and searching over 1.5 million words (50,000+ vector chunks) strictly on local CPU hardware.

---

## 🏗️ System Architecture & Data Flow

The engine utilizes a "Neural-Sparse Fusion" approach. It guarantees that both highly specific lexical keywords (names, dates) and ambiguous semantic philosophies are retrieved accurately before being reranked by a deep neural network.

```mermaid
graph TD
    %% Define Styles
    classDef python fill:#3776AB,stroke:#fff,stroke-width:2px,color:#fff;
    classDef cpp fill:#00599C,stroke:#fff,stroke-width:2px,color:#fff;
    classDef data fill:#FF9900,stroke:#fff,stroke-width:2px,color:#fff;

    %% Nodes
    A[User Search Query] -->|Raw Text| B(Python: all-MiniLM-L6-v2)
    B -->|Generates 384D Vector| C[(data/query.json)]
    
    C -->|JSON IPC Handoff| D{C++ Hybrid Engine}
    
    subgraph Stage 1: Fast Retrieval [C++ Microservice]
        D -->|Lexical Path| E[Okapi BM25 Index]
        D -->|Semantic Path| F[HNSW Graph / hnswlib]
        E -->|Keyword Match| G[Reciprocal Rank Fusion]
        F -->|L2 Euclidean Dist| G
    end
    
    G -->|Top 60 Candidates| H[(C++ stdout)]
    
    H -->|JSON IPC Handoff| I{Python: Stage 2 Reranking}
    
    subgraph Stage 2: Deep Reranking [Python]
        I -->|Pairs: Query + Chunk| J[ms-marco-MiniLM-L-6-v2]
        J -->|Cross-Encoder Scoring| K[Top 5 Results]
    end
    
    K --> L[Streamlit UI]

    %% Apply Styles
    class A,B,I,J,K,L python;
    class D,E,F,G,H cpp;
    class C data;