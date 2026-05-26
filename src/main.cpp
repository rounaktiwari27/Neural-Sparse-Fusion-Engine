#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <iomanip>

#include "../include/BM25.hpp"
#include "../third_party/hnswlib/hnswlib/hnswlib.h"

// Generates simulated dense embeddings
std::vector<float> generate_dummy_vector(int dim, float base_val) {
    std::vector<float> vec(dim);
    for (int i = 0; i < dim; i++) {
        vec[i] = base_val + (i * 0.01f); 
    }
    return vec;
}

// Reciprocal Rank Fusion (RRF) Algorithm
std::vector<std::pair<float, int>> compute_rrf(
    const std::vector<std::pair<float, int>>& bm25_ranked,
    const std::vector<std::pair<float, int>>& hnsw_ranked,
    int k = 60) {
    
    std::unordered_map<int, float> rrf_scores;
    
    for (size_t rank = 0; rank < bm25_ranked.size(); ++rank) {
        int doc_id = bm25_ranked[rank].second;
        rrf_scores[doc_id] += 1.0f / (k + rank + 1); 
    }
    
    for (size_t rank = 0; rank < hnsw_ranked.size(); ++rank) {
        int doc_id = hnsw_ranked[rank].second;
        rrf_scores[doc_id] += 1.0f / (k + rank + 1);
    }
    
    std::vector<std::pair<float, int>> final_results;
    for (const auto& pair : rrf_scores) {
        final_results.push_back({pair.second, pair.first});
    }
    std::sort(final_results.rbegin(), final_results.rend());
    
    return final_results;
}

int main(int argc, char* argv[]) {
    // 1. Enforce Command Line Arguments
    if (argc < 2) {
        // Output a clean JSON error if no query is passed
        std::cout << "{\"error\": \"No query provided. Usage: ./hybrid_engine '<query>'\"}\n";
        return 1;
    }
    std::string text_query = argv[1];

    std::vector<std::string> corpus = {
        "Error 404 The requested webpage was not found",
        "Server timeout The system is offline due to connectivity issues",
        "HTTP Status Codes explain network responses"
    };

    // 2. Initialize Indexes (SILENTLY)
    BM25 bm25_index(corpus);
    
    int dim = 16; 
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, corpus.size(), 16, 200);

    for (size_t i = 0; i < corpus.size(); i++) {
        std::vector<float> doc_vector = generate_dummy_vector(dim, static_cast<float>(i));
        alg_hnsw->addPoint(doc_vector.data(), i); 
    }

    std::vector<float> dense_query = generate_dummy_vector(dim, 0.5f); 

    // 3. Execute Hybrid Query
    std::vector<float> raw_bm25 = bm25_index.score(text_query); 
    std::vector<std::pair<float, int>> bm25_ranked;
    for (int i = 0; i < raw_bm25.size(); i++) {
        bm25_ranked.push_back({raw_bm25[i], i});
    }
    std::sort(bm25_ranked.rbegin(), bm25_ranked.rend()); 

    auto hnsw_result = alg_hnsw->searchKnn(dense_query.data(), corpus.size());
    std::vector<std::pair<float, int>> hnsw_ranked;
    while (!hnsw_result.empty()) {
        auto pair = hnsw_result.top(); 
        hnsw_ranked.push_back({pair.first, pair.second}); 
        hnsw_result.pop();
    }
    std::reverse(hnsw_ranked.begin(), hnsw_ranked.end());

    std::vector<std::pair<float, int>> final_fused_results = compute_rrf(bm25_ranked, hnsw_ranked);

    // 4. Output Pure JSON
    std::cout << "[\n";
    for (size_t i = 0; i < final_fused_results.size(); i++) {
        int doc_id = final_fused_results[i].second;
        float score = final_fused_results[i].first;
        
        std::cout << "  {\n";
        std::cout << "    \"rank\": " << i + 1 << ",\n";
        std::cout << "    \"doc_id\": " << doc_id << ",\n";
        std::cout << "    \"rrf_score\": " << std::fixed << std::setprecision(5) << score << ",\n";
        std::cout << "    \"text\": \"" << corpus[doc_id] << "\"\n";
        std::cout << "  }";
        
        // Add comma to all but the last element
        if (i < final_fused_results.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "]\n";

    delete alg_hnsw;
    return 0;
}