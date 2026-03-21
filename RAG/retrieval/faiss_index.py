# retrieval/faiss_index.py

import faiss
import numpy as np


class FaissIndexer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def build_index(self, embeddings):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

    def search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector]).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)
        return scores[0], indices[0]