"""Build the search index from PDFs in data/docs.

Run from the project root:  python -m src.ingest
"""
import pickle

import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from src.config import (CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, EMBED_MODEL,
                        INDEX_DIR)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def load_chunks(docs_dir=DOCS_DIR):
    """Return a list of {source, page, text} dicts, one per chunk."""
    chunks = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        with fitz.open(pdf_path) as doc:
            for page_no, page in enumerate(doc, start=1):
                for piece in chunk_text(page.get_text()):
                    if len(piece.strip()) > 50:  # skip near-empty chunks
                        chunks.append({"source": pdf_path.name,
                                       "page": page_no,
                                       "text": piece})
    return chunks


def build_index():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}. Add some and retry.")

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunks],
                           normalize_embeddings=True,
                           show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine sim (vectors normalized)
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Indexed {len(chunks)} chunks from {DOCS_DIR}")


if __name__ == "__main__":
    build_index()
