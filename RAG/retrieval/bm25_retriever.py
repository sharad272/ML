# retrieval/bm25_retriever.py

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.corpus = [chunk["text"] for chunk in chunks]

        # Tokenize corpus
        tokenized_corpus = [doc.split() for doc in self.corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        top_scores = [scores[i] for i in top_indices]

        return top_scores, top_indices