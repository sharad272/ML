# embedding/embedding.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        texts = []

        for chunk in chunks:
            metadata = chunk["metadata"]

            enriched_text = f"""
            File: {metadata.get('file_path')}
            Type: {metadata.get('type')}
            Name: {metadata.get('name')}

            Code:
            {chunk['text']}
            """

            texts.append(enriched_text)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks