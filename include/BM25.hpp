#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <sstream>

class BM25 {
private:
    // Default Okapi BM25 parameters for term frequency saturation and length normalization
    float k1 = 1.5f;
    float b = 0.75f;
    size_t corpus_size = 0;
    float avg_doc_len = 0.0f;
    
    // Inverted index mapping: word -> {doc_id -> term_frequency}
    std::unordered_map<std::string, std::unordered_map<int, int>> inverted_index;
    std::unordered_map<int, int> doc_lengths;
    std::unordered_map<std::string, int> document_freq; 

    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            tokens.push_back(word);
        }
        return tokens;
    }

public:
    BM25(const std::vector<std::string>& corpus) {
        corpus_size = corpus.size();
        size_t total_length = 0;

        for (size_t i = 0; i < corpus_size; i++) {
            std::vector<std::string> tokens = tokenize(corpus[i]);
            doc_lengths[i] = tokens.size();
            total_length += tokens.size();

            std::unordered_map<std::string, int> term_counts;
            for (const auto& token : tokens) {
                term_counts[token]++;
            }

            for (const auto& pair : term_counts) {
                inverted_index[pair.first][i] = pair.second;
                document_freq[pair.first]++;
            }
        }
        avg_doc_len = (corpus_size > 0) ? static_cast<float>(total_length) / corpus_size : 0.0f;
    }

    std::vector<float> score(const std::string& query) {
        std::vector<float> scores(corpus_size, 0.0f);
        std::vector<std::string> tokens = tokenize(query);

        for (const auto& token : tokens) {
            if (document_freq.find(token) == document_freq.end()) continue;

            int n_q = document_freq[token];
            float idf = std::log(1.0f + (corpus_size - n_q + 0.5f) / (n_q + 0.5f));

            for (const auto& doc_pair : inverted_index[token]) {
                int doc_id = doc_pair.first;
                int tf = doc_pair.second;
                int d_len = doc_lengths[doc_id];

                float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * (static_cast<float>(d_len) / avg_doc_len)));
                scores[doc_id] += idf * tf_norm;
            }
        }
        return scores;
    }
};