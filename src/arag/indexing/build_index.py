from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")
_MIN_SENTENCE_LEN = 10  # characters


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_SENTENCE_LEN]


def build_faiss_index(
    chunks_file: str,
    output_dir: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """
    Build a FAISS flat inner-product index from a chunks.json file.

    Produces two files in *output_dir*:
      sentences.faiss   — FAISS index (one vector per sentence)
      sentence_map.json — maps each sentence index → chunk_id

    Sentence vectors are L2-normalised so inner product == cosine similarity.
    """
    from sentence_transformers import SentenceTransformer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    with open(chunks_file) as f:
        raw: list[str] = json.load(f)

    chunks: dict[int, str] = {}
    for entry in raw:
        idx_str, _, text = entry.partition(":")
        chunks[int(idx_str)] = text

    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # ------------------------------------------------ collect sentences
    all_sentences: list[str] = []
    sentence_map: list[int] = []  # sentence_map[i] = chunk_id

    for chunk_id in sorted(chunks.keys()):
        text = chunks[chunk_id]
        sentences = _split_sentences(text) or [text[:512]]
        for sent in sentences:
            all_sentences.append(sent)
            sentence_map.append(chunk_id)

    print(f"Embedding {len(all_sentences)} sentences with '{embedding_model}'…")

    # ------------------------------------------------------ embed
    model = SentenceTransformer(embedding_model, device=device)

    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(all_sentences), batch_size), desc="Embedding"):
        batch = all_sentences[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)

    matrix = np.vstack(all_embeddings).astype(np.float32)
    dim = matrix.shape[1]

    # ------------------------------------------------ build & save FAISS
    print(f"Building FAISS IndexFlatIP (dim={dim}, n={len(all_sentences)})…")
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, str(output_path / "sentences.faiss"))
    with open(output_path / "sentence_map.json", "w") as f:
        json.dump(sentence_map, f)

    print(f"Index saved to {output_dir}/")
    print(f"  sentences.faiss : {index.ntotal} vectors")
    print(f"  sentence_map.json : {len(sentence_map)} entries")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build FAISS index from chunks.json")
    p.add_argument("--chunks", required=True, help="Path to chunks.json")
    p.add_argument("--output", required=True, help="Output directory for index files")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence embedding model")
    p.add_argument("--device", default="cpu", help="Device: cpu | cuda:0 | mps")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    build_faiss_index(
        chunks_file=args.chunks,
        output_dir=args.output,
        embedding_model=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
