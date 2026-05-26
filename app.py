import streamlit as st
import subprocess
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

@st.cache_resource
def load_models():
    bi = SentenceTransformer('all-MiniLM-L6-v2')
    cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return bi, cross

bi_encoder, reranker = load_models()

st.set_page_config(page_title="Dual Stage Retriever", layout="wide")
st.title("⚡ Dual Stage Retriever")

st.sidebar.header("📂 Document Ingestion")
# 1. UPGRADE: accept_multiple_files=True allows dropping multiple books at once
uploaded_files = st.sidebar.file_uploader("Upload corpus (.txt)", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    # 2. UPGRADE: Create a unique signature for this exact combination of books
    batch_name = "-".join(sorted([f.name for f in uploaded_files]))
    
    # 3. UPGRADE: Only run ingestion if this specific combination hasn't been processed yet
    if "processed_batch" not in st.session_state or st.session_state.processed_batch != batch_name:
        raw_text = ""
        
        # Loop through every uploaded book and merge them into one massive string
        for f in uploaded_files:
            raw_text += f.read().decode("utf-8").replace('\r\n', '\n') + "\n\n"
            
        chunks = [chunk.strip() for chunk in raw_text.split('\n\n') if len(chunk.strip()) > 10]
        
        with st.sidebar:
            with st.spinner(f"Generating 384D Embeddings for {len(chunks)} chunks..."):
                vectors = bi_encoder.encode(chunks).tolist()
                corpus_data = [{"id": i, "text": chunk, "vector": vec} for i, (chunk, vec) in enumerate(zip(chunks, vectors))]
                
                os.makedirs("data", exist_ok=True)
                with open("data/corpus.json", "w") as f:
                    json.dump(corpus_data, f, indent=4)
                    
        # Lock the state with the batch signature
        st.session_state.processed_batch = batch_name
        st.sidebar.success(f"Ingested & Vectorized {len(chunks)} chunks from {len(uploaded_files)} files.")
    else:
        st.sidebar.success("Library locked and loaded. Queries are instant.")
else:
    st.sidebar.warning("Upload .txt corpora to begin.")

query = st.text_input("Search query:")

if st.button("Search") and query:
    if not os.path.exists("data/corpus.json"):
        st.error("Missing index data.")
    else:
        query_vector = bi_encoder.encode(query).tolist()
        with open("data/query.json", "w") as f:
            json.dump({"text": query, "vector": query_vector}, f)

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
            
            st.dataframe(pd.DataFrame(table_data), width='stretch')

            with st.spinner("Applying ML Reranking..."):
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