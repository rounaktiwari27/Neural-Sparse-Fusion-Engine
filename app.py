import streamlit as st
import subprocess
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

# Cache BOTH models to prevent memory reloading
@st.cache_resource
def load_models():
    # Bi-Encoder for Stage 1 (Fast Vectorization for HNSW)
    bi = SentenceTransformer('all-MiniLM-L6-v2')
    # Cross-Encoder for Stage 2 (Deep Context Reranking)
    cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return bi, cross

bi_encoder, reranker = load_models()

st.set_page_config(page_title="True Semantic Hybrid RAG", layout="wide")
st.title("⚡ Enterprise Hybrid RAG: True Vector Search")

st.sidebar.header("📂 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload corpus (.txt)", type=["txt"])

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    chunks = [chunk.strip() for chunk in raw_text.split('\n\n') if len(chunk.strip()) > 10]
    
    with st.sidebar:
        with st.spinner("Generating 384-Dimension Dense Embeddings..."):
            # Generate real math vectors for the HNSW graph
            vectors = bi_encoder.encode(chunks).tolist()
            
            corpus_data = [{"id": i, "text": chunk, "vector": vec} for i, (chunk, vec) in enumerate(zip(chunks, vectors))]
            
            os.makedirs("data", exist_ok=True)
            with open("data/corpus.json", "w") as f:
                json.dump(corpus_data, f, indent=4)
                
    st.sidebar.success(f"Ingested & Vectorized {len(chunks)} chunks.")
else:
    st.sidebar.warning("Upload a .txt corpus to begin.")

query = st.text_input("Search query:")

if st.button("Search") and query:
    if not os.path.exists("data/corpus.json"):
        st.error("Missing index data. Upload a corpus first.")
    else:
        # 1. PREPARE THE HANDOFF FOR C++
        query_vector = bi_encoder.encode(query).tolist()
        with open("data/query.json", "w") as f:
            json.dump({"text": query, "vector": query_vector}, f)

        # 2. RUN C++ MICROSERVICE 
        with st.spinner("Executing C++ Retrieval Layer (True HNSW + BM25)..."):
            process = subprocess.run(['./hybrid_engine'], capture_output=True, text=True)
            
        try:
            cpp_results = json.loads(process.stdout)
            
            st.subheader("Stage 1: C++ Retrieval Breakdown")
            table_data = [{
                "Final Rank": res["final_rank"],
                "RRF Score": round(res["rrf_score"], 4),
                "BM25 Rank": f"#{res['bm25_rank']} ({round(res['bm25_score'], 2)})",
                "HNSW Rank": f"#{res['hnsw_rank']} (L2 Dist: {round(res['hnsw_dist'], 2)})",
                "Snippet": res["text"][:80] + "..." 
            } for res in cpp_results]
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # 3. ML RERANKING
            with st.spinner("Executing Python Cross-Encoder Reranking..."):
                pairs = [[query, doc["text"]] for doc in cpp_results]
                scores = reranker.predict(pairs)
                
                for i, doc in enumerate(cpp_results):
                    doc["cross_encoder_score"] = float(scores[i])
                    
                final_results = sorted(cpp_results, key=lambda x: x["cross_encoder_score"], reverse=True)

            st.subheader("Stage 2: Final Cross-Encoder Selection")
            for res in final_results:
                st.markdown(f"**ML Confidence:** `{res['cross_encoder_score']:.4f}` | **C++ Rank:** #{res['final_rank']}")
                st.info(res['text'])

        except json.JSONDecodeError:
            st.error("IPC Failure: Could not parse C++ engine output.")
            st.code(process.stdout)