import streamlit as st
import subprocess
import json
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------
# STAGE 2: THE EXPERT FILTER (Cross-Encoder)
# ---------------------------------------------------------
@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

reranker = load_reranker()

# ---------------------------------------------------------
# THE USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="Hybrid RAG Engine", layout="wide")
st.title("⚡ C++ Hybrid Search & Python Reranker")

query = st.text_input("Enter your search query (e.g., 'network offline'):")

if st.button("Search") and query:
    
    # 1. CALL THE C++ MICROSERVICE
    with st.spinner("Running C++ Hybrid Fusion (BM25 + HNSW)..."):
        process = subprocess.run(
            ['./hybrid_engine', query],
            capture_output=True,
            text=True
        )
        
    try:
        # Read the JSON output from C++
        cpp_results = json.loads(process.stdout)
        
        st.subheader("Stage 1: C++ Hybrid 'Fast Net' Results")
        st.caption("Documents retrieved via BM25 & HNSW, merged using Reciprocal Rank Fusion (RRF).")
        st.json(cpp_results)

        # 2. RUN THE PYTHON CROSS-ENCODER
        with st.spinner("Applying ML Cross-Encoder Reranking..."):
            pairs = [[query, doc["text"]] for doc in cpp_results]
            scores = reranker.predict(pairs)
            
            for i, doc in enumerate(cpp_results):
                doc["cross_encoder_score"] = float(scores[i])
                
            final_results = sorted(cpp_results, key=lambda x: x["cross_encoder_score"], reverse=True)

        # 3. DISPLAY FINAL RESULTS
        st.subheader("Stage 2: Final ML Reranked Output")
        st.caption("The Cross-Encoder's final decision based on deep semantic context.")
        
        for res in final_results:
            st.markdown(f"**Score:** `{res['cross_encoder_score']:.4f}` | **Doc ID:** {res['doc_id']}")
            st.info(res['text'])
            
    except json.JSONDecodeError:
        st.error("Failed to parse C++ output. Ensure the C++ engine prints strictly JSON.")
        st.code(process.stdout)