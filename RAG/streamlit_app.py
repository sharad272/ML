import io
import os
from typing import List, Tuple

import numpy as np
import streamlit as st

from chunking.recursive_chunking import recursive_split_chunk
from chunking.structure_aware_chunking import extract_chunks
from embedding.embedding import Embedder
from main import build_rag_context, generate_rag_answer, rerank_results
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_index import FaissIndexer
from retrieval.hybrid_retriever import HybridRetriever


def _uploads_signature(uploaded_files) -> Tuple[Tuple[str, int], ...]:
    """Stable signature of current frontend-uploaded files."""
    if not uploaded_files:
        return tuple()
    return tuple(sorted((f.name, int(f.size or 0)) for f in uploaded_files))


def _decode_file_bytes(data: bytes) -> str:
    """Decode arbitrary file bytes into best-effort text."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_pdf_from_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("PDF support requires 'pypdf'. Install with: pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunks_from_uploaded_file(uploaded_file) -> List[dict]:
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()
    data = uploaded_file.getvalue()

    if extension == ".pdf":
        text = _read_pdf_from_bytes(data)
        base_chunk = {
            "text": text,
            "metadata": {
                "file_path": filename,
                "type": "PDFDocument",
                "name": filename,
                "language": "text",
            },
        }
        return recursive_split_chunk(base_chunk)

    text = _decode_file_bytes(data)
    if not text.strip():
        return []

    if extension == ".py":
        try:
            structure_chunks = extract_chunks(text, filename)
            sub_chunks = []
            for chunk in structure_chunks:
                sub_chunks.extend(recursive_split_chunk(chunk))
            if sub_chunks:
                return sub_chunks
        except Exception:
            # Fall through to generic chunking if AST parsing fails.
            pass

    language = extension[1:] if extension else "text"
    base_chunk = {
        "text": text,
        "metadata": {
            "file_path": filename,
            "type": "GenericDocument",
            "name": filename,
            "language": language,
        },
    }
    return recursive_split_chunk(base_chunk)


def build_index_from_uploads(uploaded_files) -> Tuple[List[dict], Embedder, HybridRetriever]:
    all_chunks = []
    for uploaded_file in uploaded_files:
        all_chunks.extend(chunks_from_uploaded_file(uploaded_file))

    if not all_chunks:
        raise ValueError("No readable content found in uploaded files.")

    embedder = Embedder()
    all_chunks = embedder.embed_chunks(all_chunks)

    embedding_matrix = np.array([chunk["embedding"] for chunk in all_chunks]).astype("float32")
    indexer = FaissIndexer(embedding_matrix.shape[1])
    indexer.build_index(embedding_matrix)

    bm25_retriever = BM25Retriever(all_chunks)
    hybrid = HybridRetriever(indexer, bm25_retriever, all_chunks)
    return all_chunks, embedder, hybrid


def run_query(query: str, all_chunks: List[dict], embedder: Embedder, hybrid: HybridRetriever):
    scores, indices = hybrid.search(query, embedder, alpha=0.6, top_k=10)
    reranked_results = rerank_results(
        query=query,
        candidate_indices=indices,
        all_chunks=all_chunks,
        embedder=embedder,
        top_k=10,
    )
    context = build_rag_context(
        reranked_results,
        all_chunks,
        max_chunks=5,
        query=query,
    )
    answer, model_used = generate_rag_answer(query, context)
    return answer, model_used, scores, indices, reranked_results, context


def main():
    st.set_page_config(page_title="RAG Multi-File QA", layout="wide")
    st.title("RAG QA over Multiple Uploaded Files")
    st.caption("Upload PDFs, Python files, code files, or other text-like files, then ask questions.")
    st.info("Context is taken only from files uploaded in this UI. No local disk files are used.")

    uploaded_files = st.file_uploader(
        "Upload one or more files",
        accept_multiple_files=True,
        type=None,
    )

    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = None
        st.session_state.embedder = None
        st.session_state.hybrid = None
        st.session_state.uploaded_names = []
        st.session_state.built_signature = tuple()

    current_signature = _uploads_signature(uploaded_files)
    uploads_changed = current_signature != st.session_state.built_signature

    if st.button("Build / Rebuild Index", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload files before building the index.")
        else:
            with st.spinner("Building index from uploaded files..."):
                try:
                    all_chunks, embedder, hybrid = build_index_from_uploads(uploaded_files)
                    st.session_state.all_chunks = all_chunks
                    st.session_state.embedder = embedder
                    st.session_state.hybrid = hybrid
                    st.session_state.uploaded_names = [f.name for f in uploaded_files]
                    st.session_state.built_signature = current_signature
                    st.success(f"Indexed {len(st.session_state.uploaded_names)} file(s) into {len(all_chunks)} chunks.")
                except Exception as exc:
                    st.error(f"Index build failed: {exc}")

    if st.session_state.all_chunks is not None:
        st.write("Indexed files:", ", ".join(st.session_state.uploaded_names))
        if uploads_changed:
            st.warning("Uploaded files changed. Rebuild index to use the latest frontend context.")
        query = st.text_input("Ask a question", value="")
        run_disabled = uploads_changed or not st.session_state.built_signature
        if st.button("Run RAG", use_container_width=True, disabled=run_disabled):
            if not query.strip():
                st.warning("Enter a question first.")
            elif uploads_changed:
                st.warning("Please rebuild index after changing uploaded files.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        answer, model_used, scores, indices, reranked_results, context = run_query(
                            query,
                            st.session_state.all_chunks,
                            st.session_state.embedder,
                            st.session_state.hybrid,
                        )
                        st.subheader("Answer")
                        st.write(f"Model: `{model_used}`")
                        st.write(answer)

                        with st.expander("Reranked chunks used for context", expanded=False):
                            for score, idx in reranked_results[:5]:
                                st.write(f"Score: {score:.4f}")
                                st.json(st.session_state.all_chunks[idx]["metadata"])

                        with st.expander("Context sent to LLM", expanded=False):
                            st.text(context)

                        with st.expander("Hybrid candidate scores", expanded=False):
                            for score, idx in zip(scores, indices):
                                st.write(f"Score: {float(score):.4f} | Chunk idx: {int(idx)}")
                    except Exception as exc:
                        st.error(f"RAG failed: {exc}")


if __name__ == "__main__":
    main()
