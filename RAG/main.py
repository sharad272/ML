import os
import numpy as np

from chunking.structure_aware_chunking import extract_chunks
from chunking.recursive_chunking import recursive_split_chunk
from embedding.embedding import Embedder
from retrieval.faiss_index import FaissIndexer
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever


def load_supported_files(root_dir):
    # Collect all retrievable source files from the project tree.
    supported_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") or file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                supported_files.append(full_path)

    # Always include root-level file.pdf when present.
    default_pdf = os.path.join(root_dir, "file.pdf")
    if os.path.exists(default_pdf) and default_pdf not in supported_files:
        supported_files.append(default_pdf)

    return supported_files


def read_pdf_text(file_path):
    # Read PDF pages and merge extracted text into one document string.
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "PDF support requires 'pypdf'. Install it with: pip install pypdf"
        )

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# Cross-encoder reranking (kept commented intentionally).
# def load_cross_encoder(model_name="BAAI/bge-reranker-base"):
#     try:
#         from sentence_transformers import CrossEncoder
#     except ImportError:
#         print(
#             "Cross-encoder unavailable (sentence-transformers import failed). "
#             "Falling back to bi-encoder reranking."
#         )
#         return None
#
#     try:
#         return CrossEncoder(model_name)
#     except Exception as e:
#         print(
#             f"Cross-encoder model load failed: {e}. "
#             "Falling back to bi-encoder reranking."
#         )
#         return None


def rerank_results(
    query,
    candidate_indices,
    all_chunks,
    embedder,
    top_k=3
):
    # Re-score retrieved candidates using bi-encoder similarity.
    if not candidate_indices:
        return []

    reranked = []
    query_embedding = embedder.model.encode(
        query,
        normalize_embeddings=True
    )

    for idx in candidate_indices:
        chunk_embedding = embedder.model.encode(
            all_chunks[idx]["text"],
            normalize_embeddings=True
        )
        score = float(np.dot(query_embedding, chunk_embedding))
        reranked.append((score, idx))

    # Cross-encoder reranking path (kept commented intentionally).
    # if cross_encoder is not None:
    #     pairs = [(query, all_chunks[idx]["text"]) for idx in candidate_indices]
    #     scores = cross_encoder.predict(pairs)
    #     reranked = [(float(score), idx) for score, idx in zip(scores, candidate_indices)]

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked[:top_k]


def precision_at_k(retrieved_indices, relevant_indices, k):
    # Share of top-k results that are relevant.
    if k <= 0:
        return 0.0
    top_k = retrieved_indices[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for idx in top_k if idx in relevant_indices)
    return hits / len(top_k)


def recall_at_k(retrieved_indices, relevant_indices, k):
    # Share of relevant results recovered in top-k.
    if not relevant_indices:
        return 0.0
    top_k = retrieved_indices[:k]
    hits = sum(1 for idx in top_k if idx in relevant_indices)
    return hits / len(relevant_indices)


def mrr(retrieved_indices, relevant_indices):
    # Reciprocal rank of first relevant hit.
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_metrics(retrieved_indices, relevant_indices, k_values=(1, 3, 5)):
    # Aggregate retrieval quality metrics for selected cutoffs.
    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_indices, relevant_indices, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_indices, relevant_indices, k)
    metrics["mrr"] = mrr(retrieved_indices, relevant_indices)
    return metrics


def main():
    root_directory = "."   # current project folder

    # Store every chunk that will be embedded and indexed.
    all_chunks = []

    # Discover all candidate files for ingestion.
    supported_files = load_supported_files(root_directory)
    print(f"Found {len(supported_files)} supported files (.py/.pdf)")

    # Process each discovered file into chunk objects.
    for file_path in supported_files:
        # if file_path.endswith(".py"):
        #     with open(file_path, "r", encoding="utf-8") as f:
        #         content = f.read()

        #     structure_chunks = extract_chunks(content, file_path)

        #     for chunk in structure_chunks:
        #         sub_chunks = recursive_split_chunk(chunk)
        #         all_chunks.extend(sub_chunks)

        if file_path.endswith(".pdf"):
            # 1) Extract raw PDF text.
            content = read_pdf_text(file_path)
            # 2) Create a base chunk with metadata.
            pdf_chunk = {
                "text": content,
                "metadata": {
                    "file_path": file_path,
                    "type": "PDFDocument",
                    "name": os.path.basename(file_path),
                    "language": "text"
                }
            }
            # 3) Split large text into overlap-aware smaller chunks.
            sub_chunks = recursive_split_chunk(pdf_chunk)
            # 4) Add sub-chunks to the global corpus.
            all_chunks.extend(sub_chunks)

    print(f"Total chunks from all files: {len(all_chunks)}")

    # Encode chunks into dense vectors.
    embedder = Embedder()
    all_chunks = embedder.embed_chunks(all_chunks)

    # Build matrix format expected by FAISS.
    embedding_matrix = np.array(
        [chunk["embedding"] for chunk in all_chunks]
    ).astype("float32")

    # Build dense ANN index for semantic retrieval.
    dimension = embedding_matrix.shape[1]
    indexer = FaissIndexer(dimension)
    indexer.build_index(embedding_matrix)

    print("FAISS index built successfully!")

    # Build sparse lexical retriever.
    bm25_retriever = BM25Retriever(all_chunks)

    # Combine dense and sparse scores with weighted fusion.
    hybrid = HybridRetriever(indexer, bm25_retriever, all_chunks)
    # cross_encoder = load_cross_encoder()

    # Query to test retrieval + reranking.
    query = "How did Pablo Neruda know that somebody behind him was looking at him?"
    # Retrieve broader candidate set from hybrid retriever.
    scores, indices = hybrid.search(query, embedder, alpha=0.6, top_k=10)
    # Reorder top candidates with bi-encoder reranking.
    reranked_results = rerank_results(
        query=query,
        candidate_indices=indices,
        all_chunks=all_chunks,
        embedder=embedder,
        top_k=3
    )

    print("\nHybrid Results:")
    for score, idx in zip(scores, indices):
        print(f"\nScore: {score}")
        print("Metadata:", all_chunks[idx]["metadata"])
        print("-" * 50)

    print("\nReranked Results:")
    for score, idx in reranked_results:
        print(f"\nRerank Score: {score}")
        print("Metadata:", all_chunks[idx]["metadata"])
        print("-" * 50)

    # Metrics evaluation using labeled relevant chunk indices.
    # Fill this set when running offline evaluation. Example: {2, 7}
    relevant_indices = set()

    if relevant_indices:
        reranked_indices = [idx for _, idx in reranked_results]
        metrics = evaluate_retrieval_metrics(
            retrieved_indices=reranked_indices,
            relevant_indices=relevant_indices,
            k_values=(1, 3)
        )
        print("\nEvaluation Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
    else:
        print(
            "\nEvaluation skipped: set `relevant_indices` in code to compute "
            "Recall@k / Precision@k / MRR."
        )


if __name__ == "__main__":
    main()