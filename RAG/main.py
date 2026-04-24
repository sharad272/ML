import os
import sys
import numpy as np
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from chunking.structure_aware_chunking import extract_chunks
from chunking.recursive_chunking import recursive_split_chunk
from embedding.embedding import Embedder
from retrieval.faiss_index import FaissIndexer
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

if load_dotenv is not None:
    load_dotenv()


def get_rag_llm_model():
    # Pick a default instruction model that performs well with grounded context.
    return os.getenv("HF_RAG_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def build_rag_context(
    reranked_results,
    all_chunks,
    max_chunks=3,
    max_chars_per_chunk=1200,
    query=None
):
    # Convert top retrieved chunks into a compact, citation-like context block.
    context_blocks = []
    seen_signatures = set()
    stopwords = {
        "a", "an", "the", "of", "to", "in", "at", "on", "and", "or", "for",
        "was", "were", "is", "are", "who", "what", "when", "where", "how"
    }
    query_terms = set()
    if query:
        query_terms = {
            token.strip(".,!?;:'\"()[]{}").lower()
            for token in query.split()
            if token.strip(".,!?;:'\"()[]{}")
        }
        query_terms = {token for token in query_terms if token not in stopwords}

    def select_best_window(text):
        if len(text) <= max_chars_per_chunk or not query_terms:
            return text
        lowered = text.lower()
        window_size = max_chars_per_chunk
        step = max(120, window_size // 6)
        best_start = 0
        best_score = -1.0
        max_start = max(0, len(text) - window_size)

        for start in range(0, max_start + 1, step):
            end = min(len(text), start + window_size)
            window = lowered[start:end]
            term_hits = sum(1 for term in query_terms if term in window)
            term_ratio = term_hits / len(query_terms)
            score = term_hits + (0.4 * term_ratio)
            if score > best_score:
                best_score = score
                best_start = start

        final_end = min(len(text), best_start + window_size)
        return text[best_start:final_end]

    # Keep reranker score primary, with a small boost for lexical coverage.
    candidates = []
    for base_score, idx in reranked_results:
        chunk_text = all_chunks[idx].get("text", "")
        overlap_hits = sum(1 for term in query_terms if term in chunk_text.lower()) if query_terms else 0
        overlap_ratio = overlap_hits / len(query_terms) if query_terms else 0.0
        adjusted_score = (0.85 * base_score) + (0.15 * overlap_ratio)
        candidates.append((adjusted_score, idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    rank = 1
    for _, idx in candidates:
        if rank > max_chunks:
            break
        chunk = all_chunks[idx]
        metadata = chunk.get("metadata", {})
        source_name = metadata.get("name") or metadata.get("file_path", "unknown")
        text = chunk.get("text", "")

        # Skip near-duplicate chunks so context budget goes to diverse evidence.
        signature = " ".join(text.lower().split())[:280]
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        text = select_best_window(text)
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        context_blocks.append(f"[{rank}] Source: {source_name}\n{text}")
        rank += 1
    return "\n\n".join(context_blocks)


def generate_rag_answer(query, context, model_name=None):
    # Call Hugging Face Inference API with token auth and grounded prompt.
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN in environment or .env file."
        )

    if model_name is None:
        model_name = get_rag_llm_model()

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError(
            "Missing dependency 'huggingface_hub'. Install it with: "
            "pip install huggingface_hub"
        )

    client = InferenceClient(token=hf_token)
    system_prompt = (
        "You are a RAG assistant. Answer only using the retrieved context. "
        "If the answer is not in context, clearly say you do not have enough information."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Give a concise answer and cite sources as [1], [2], etc."
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return completion.choices[0].message.content, model_name


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

    stopwords = {
        "a", "an", "the", "of", "to", "in", "at", "on", "and", "or", "for",
        "was", "were", "is", "are", "who", "what", "when", "where", "how"
    }
    query_terms = {
        token.strip(".,!?;:'\"()[]{}").lower()
        for token in query.split()
        if token.strip(".,!?;:'\"()[]{}")
    }
    query_terms = {token for token in query_terms if token not in stopwords}

    reranked = []
    query_embedding = embedder.model.encode(
        query,
        normalize_embeddings=True
    )

    for idx in candidate_indices:
        chunk_text = all_chunks[idx]["text"]
        chunk_embedding = embedder.model.encode(
            chunk_text,
            normalize_embeddings=True
        )
        semantic_score = float(np.dot(query_embedding, chunk_embedding))
        chunk_lower = chunk_text.lower()
        overlap_hits = sum(1 for term in query_terms if term in chunk_lower)
        overlap_ratio = overlap_hits / len(query_terms) if query_terms else 0.0
        # Blend semantic similarity with lexical overlap so exact-fact chunks survive.
        score = 0.8 * semantic_score + 0.2 * overlap_ratio
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
    # Require explicit user-provided files (no automatic local directory scanning).
    provided_files = [path for path in sys.argv[1:] if os.path.exists(path)]
    if not provided_files:
        print(
            "No user-provided context files found.\n"
            "Provide file paths explicitly, e.g.:\n"
            "python main.py ./file.pdf ./notes.txt ./module.py\n"
            "Or use streamlit_app.py to upload files from frontend."
        )
        return

    all_chunks = []
    print(f"Using {len(provided_files)} user-provided file(s).")

    for file_path in provided_files:
        if file_path.endswith(".pdf"):
            content = read_pdf_text(file_path)
            base_chunk = {
                "text": content,
                "metadata": {
                    "file_path": file_path,
                    "type": "PDFDocument",
                    "name": os.path.basename(file_path),
                    "language": "text"
                }
            }
            all_chunks.extend(recursive_split_chunk(base_chunk))
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            continue

        if file_path.endswith(".py"):
            try:
                structure_chunks = extract_chunks(content, file_path)
                for chunk in structure_chunks:
                    all_chunks.extend(recursive_split_chunk(chunk))
                continue
            except Exception:
                # Fall back to generic chunking for invalid Python files.
                pass

        generic_chunk = {
            "text": content,
            "metadata": {
                "file_path": file_path,
                "type": "GenericDocument",
                "name": os.path.basename(file_path),
                "language": os.path.splitext(file_path)[1].lstrip(".") or "text"
            }
        }
        all_chunks.extend(recursive_split_chunk(generic_chunk))

    print(f"Total chunks from all files: {len(all_chunks)}")
    if not all_chunks:
        print("No readable content extracted from provided files.")
        return

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
    query = "Who was the woman found in the car at Havana Riviera Hotel?"
    # Retrieve broader candidate set from hybrid retriever.
    scores, indices = hybrid.search(query, embedder, alpha=0.6, top_k=10)
    # Reorder top candidates with bi-encoder reranking.
    reranked_results = rerank_results(
        query=query,
        candidate_indices=indices,
        all_chunks=all_chunks,
        embedder=embedder,
        top_k=10
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

    if reranked_results:
        context = build_rag_context(
            reranked_results,
            all_chunks,
            max_chunks=5,
            query=query
        )
        try:
            answer, model_used = generate_rag_answer(query, context)
            print("\nRAG Answer:")
            print(f"Model: {model_used}")
            print(answer)
        except Exception as e:
            print(f"\nRAG generation skipped: {e}")
    else:
        print("\nRAG generation skipped: no retrieved chunks.")

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