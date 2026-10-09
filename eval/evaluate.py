"""Compare retrieval modes on your own question set.

Run from the project root:  python -m eval.evaluate

eval/questions.json is a list like:
  {"question": "...", "source": "file.pdf", "page": 12}
where source/page is where the answer actually is. A retrieved chunk counts as
a hit if it comes from that source and page.
"""
import csv
import json
from pathlib import Path

from src.config import TOP_K
from src.retriever import MODES, Retriever

QUESTIONS = Path(__file__).parent / "questions.json"
OUT = Path(__file__).parent / "results.csv"


def evaluate(retriever, questions, mode, k=TOP_K):
    hits, rr_total = 0, 0.0
    for q in questions:
        results = retriever.search(q["question"], k=k, mode=mode)
        rank = next((i for i, c in enumerate(results, start=1)
                     if c["source"] == q["source"] and c["page"] == q["page"]), None)
        if rank:
            hits += 1
            rr_total += 1.0 / rank
    n = len(questions)
    return {"mode": mode, f"hit@{k}": round(hits / n, 3), "MRR": round(rr_total / n, 3)}


def main():
    questions = json.loads(QUESTIONS.read_text())
    retriever = Retriever()
    rows = [evaluate(retriever, questions, m) for m in MODES]

    print(f"\nEvaluated on {len(questions)} questions\n")
    for r in rows:
        print("  ".join(f"{key}={val}" for key, val in r.items()))

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
