# retrieval/hybrid_retriever.py

import numpy as np


class HybridRetriever:
    def __init__(self, dense_indexer, bm25_retriever, final_chunks):
        self.dense_indexer = dense_indexer
        self.bm25_retriever = bm25_retriever
        self.final_chunks = final_chunks

    def search(self, query, embedder, alpha=0.6, top_k=5):
        # Dense search
        query_embedding = embedder.model.encode(
            query,
            normalize_embeddings=True
        )

        dense_scores, dense_indices = self.dense_indexer.search(
            query_embedding,
            top_k=top_k * 2
        )

        # Sparse search
        sparse_scores, sparse_indices = self.bm25_retriever.search(
            query,
            top_k=top_k * 2
        )

        # Normalize scores
        dense_scores = np.array(dense_scores)
        sparse_scores = np.array(sparse_scores)

        if dense_scores.max() != 0:
            dense_scores = dense_scores / dense_scores.max()

        if sparse_scores.max() != 0:
            sparse_scores = sparse_scores / sparse_scores.max()

        # Combine into dictionary
        combined_scores = {}

        for score, idx in zip(dense_scores, dense_indices):
            combined_scores[idx] = alpha * score

        for score, idx in zip(sparse_scores, sparse_indices):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * score
            else:
                combined_scores[idx] = (1 - alpha) * score

        # Sort final scores
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        final_indices = [item[0] for item in sorted_results]
        final_scores = [item[1] for item in sorted_results]

        return final_scores, final_indices