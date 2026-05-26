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

st.set_page_config(page_title="Neural-Sparse Fusion Engine", layout="wide")
st.title("⚡ Neural-Sparse Fusion Engine")

# --- SIDEBAR: DOCUMENT INGESTION ---
st.sidebar.header("📂 Document Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload corpus (.txt)", type=["txt"], accept_multiple_files=True)

# GRACEFUL RESET: If the user clears the uploader, reset the session memory
if not uploaded_files and "processed_batch" in st.session_state:
    del st.session_state["processed_batch"]
    
if uploaded_files:
    batch_name = "-".join(sorted([f.name for f in uploaded_files]))
    
    if "processed_batch" not in st.session_state or st.session_state.processed_batch != batch_name:
        raw_text = ""
        
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
                    
        st.session_state.processed_batch = batch_name
        st.sidebar.success(f"Ingested & Vectorized {len(chunks)} chunks from {len(uploaded_files)} files.")
    else:
        st.sidebar.success("Library locked and loaded. Queries are instant.")
else:
    st.sidebar.warning("Upload .txt corpora to begin.")


# --- MAIN DASHBOARD AREA ---
query = st.text_input("🔍 Semantic Search Query:")

# Display a clean dashboard warning if no data is loaded yet
if not os.path.exists("data/corpus.json") or not uploaded_files:
    st.info("👋 Welcome to the Neural-Sparse Fusion Engine. Please upload your .txt corpora in the sidebar to initialize the C++ backend.")

elif st.button("Execute Search") and query:
    # 1. PREPARE THE HANDOFF FOR C++
    query_vector = bi_encoder.encode(query).tolist()
    with open("data/query.json", "w") as f:
        json.dump({"text": query, "vector": query_vector}, f)

    # 2. RUN C++ MICROSERVICE 
    with st.spinner("Executing C++ Retrieval Layer (O(log N) HNSW Graph Navigation)..."):
        process = subprocess.run(['./hybrid_engine'], capture_output=True, text=True)
        
    try:
        cpp_results = json.loads(process.stdout)
        
        st.subheader("⚙️ Stage 1: C++ Retrieval Breakdown (RRF Fusion)")
        table_data = [{
            "Final Rank": res["final_rank"],
            "RRF Score": round(res["rrf_score"], 4),
            "BM25 Rank": f"#{res['bm25_rank']} ({round(res['bm25_score'], 2)})",
            "HNSW Rank": f"#{res['hnsw_rank']} (L2 Dist: {round(res['hnsw_dist'], 2)})",
            "Snippet": res["text"][:80] + "..." 
        } for res in cpp_results]
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # 3. ML RERANKING
        with st.spinner("Applying Deep Neural Reranking (Cross-Encoder)..."):
            pairs = [[query, doc["text"]] for doc in cpp_results]
            scores = reranker.predict(pairs)
            
            for i, doc in enumerate(cpp_results):
                doc["cross_encoder_score"] = float(scores[i])
                
            final_results = sorted(cpp_results, key=lambda x: x["cross_encoder_score"], reverse=True)

        st.subheader("🧠 Stage 2: Final Semantic Selection")
        for res in final_results:
            st.markdown(f"**ML Confidence:** `{res['cross_encoder_score']:.4f}` | **Initial C++ Rank:** #{res['final_rank']}")
            st.info(res['text'])

    except json.JSONDecodeError:
        st.error("🚨 IPC Failure: C++ microservice crashed or returned invalid JSON.")
        st.code(process.stdout)

# --- SYSTEM METRICS SIDEBAR ---
if os.path.exists("data/corpus.json") and uploaded_files:
    with open("data/corpus.json", "r") as f:
        try:
            corpus_len = len(json.load(f))
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 System Metrics")
            st.sidebar.markdown(f"- **Nodes in Graph:** `{corpus_len}`")
            st.sidebar.markdown("- **Vector Dimensions:** `384`")
            st.sidebar.markdown("- **Distance Metric:** `L2 (Euclidean)`")
            st.sidebar.markdown("- **Reranker Limit:** `Top 5 candidates`")
        except:
            pass