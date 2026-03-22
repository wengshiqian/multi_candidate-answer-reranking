import argparse
import json
import math
from typing import List, Dict, Iterable, Tuple

import numpy as np


def _load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_answer(sample: Dict) -> str:
    convs = sample.get("conversations") or []
    if len(convs) >= 2:
        return str(convs[1].get("value", "")).strip()
    return ""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _build_chunks(samples: Iterable[Dict], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    for sample in samples:
        answer = _extract_answer(sample)
        if not answer:
            continue
        chunks.extend(_chunk_text(answer, chunk_size, overlap))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from PubMedVision RAG corpus.")
    parser.add_argument("--corpus", required=True, help="Filtered RAG corpus JSON")
    parser.add_argument("--output_index", required=True, help="Output FAISS index path")
    parser.add_argument("--output_chunks", required=True, help="Output chunks JSON path")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--chunk_size", type=int, default=800, help="Character chunk size")
    parser.add_argument("--overlap", type=int, default=120, help="Character overlap")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    samples = _load_json(args.corpus)
    chunks = _build_chunks(samples, args.chunk_size, args.overlap)

    if not chunks:
        raise ValueError("No chunks generated from corpus.")

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Please install: pip install sentence-transformers faiss-cpu"
        ) from exc

    model = SentenceTransformer(args.model)

    embeddings = []
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, args.output_index)
    with open(args.output_chunks, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks: {len(chunks)}")
    print(f"Saved FAISS index: {args.output_index}")
    print(f"Saved chunks file: {args.output_chunks}")


if __name__ == "__main__":
    main()
