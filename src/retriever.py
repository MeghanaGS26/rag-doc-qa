import pickle
import re

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import EMBED_MODEL, INDEX_DIR, RERANK_MODEL, TOP_K

MODES = ("dense", "bm25", "hybrid", "hybrid+rerank")


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._reranker = None  # loaded lazily

    def _dense(self, query, n):
        vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, ids = self.index.search(vec, n)
        return [int(i) for i in ids[0] if i != -1]

    def _sparse(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    @staticmethod
    def _rrf(rankings, k=60):
        """Reciprocal Rank Fusion: merge several ranked lists of ids."""
        fused = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def search(self, query, k=TOP_K, mode="hybrid+rerank"):
        """Return the top-k chunks (dicts) for a query."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        pool = max(k * 4, 20)  # candidate pool before final cut

        if mode == "dense":
            ids = self._dense(query, k)
        elif mode == "bm25":
            ids = self._sparse(query, k)
        else:
            ids = self._rrf([self._dense(query, pool), self._sparse(query, pool)])
            if mode == "hybrid+rerank":
                if self._reranker is None:
                    self._reranker = CrossEncoder(RERANK_MODEL)
                cands = ids[:pool]
                scores = self._reranker.predict(
                    [(query, self.chunks[i]["text"]) for i in cands])
                ids = [i for _, i in sorted(zip(scores, cands), reverse=True)]
            ids = ids[:k]

        return [self.chunks[i] for i in ids]
