import pytest

from src.ingest import chunk_text
from src.retriever import Retriever, tokenize


def test_chunks_overlap_and_cover_text():
    text = "word " * 1000
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0][-20:] == chunks[1][:20]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = Retriever._rrf([[1, 2, 3], [3, 1, 4]])
    assert fused[0] == 1  # 1 is high in both lists


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]
