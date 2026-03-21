# chunking/structure_aware_chunking.py

import ast

def extract_chunks(file_content, file_path):
    tree = ast.parse(file_content)
    lines = file_content.splitlines()
    chunks = []

    covered_lines = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno

            chunk_text = "\n".join(lines[start-1:end])
            covered_lines.update(range(start, end+1))

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "file_path": file_path,
                    "type": type(node).__name__,
                    "name": node.name,
                    "start_line": start,
                    "end_line": end,
                    "language": "python"
                }
            })

    # module-level fallback
    remaining_lines = [
        lines[i-1] for i in range(1, len(lines)+1)
        if i not in covered_lines and lines[i-1].strip()
    ]

    if remaining_lines:
        chunks.append({
            "text": "\n".join(remaining_lines),
            "metadata": {
                "file_path": file_path,
                "type": "ModuleLevel",
                "name": "top_level_code",
                "language": "python"
            }
        })

    return chunks