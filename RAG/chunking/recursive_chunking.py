# chunking/recursive_chunking.py

def recursive_split_chunk(chunk, max_lines=50, overlap=10):
    text = chunk["text"]
    metadata = chunk["metadata"]

    lines = text.splitlines()

    if len(lines) <= max_lines:
        return [chunk]

    sub_chunks = []
    start = 0

    while start < len(lines):
        end = start + max_lines
        sub_text = "\n".join(lines[start:end])

        sub_chunks.append({
            "text": sub_text,
            "metadata": {
                **metadata,
                "chunk_part": f"{start}-{end}"
            }
        })

        start += max_lines - overlap

    return sub_chunks