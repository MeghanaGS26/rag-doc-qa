"""Generate evaluation questions from random chunks of your PDFs.

Run from the project root:  python -m eval.generate_questions
Needs LLM_API_KEY, LLM_BASE_URL and LLM_MODEL set in the terminal.
"""
import json
import os
import pickle
import random
import re
import time
from pathlib import Path

from openai import OpenAI

from src.config import INDEX_DIR

N_QUESTIONS = 40
OUT = Path(__file__).parent / "questions.json"

random.seed(42)  # same chunks every run

with open(INDEX_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = OpenAI(base_url=os.getenv("LLM_BASE_URL") or None,
                api_key=os.getenv("LLM_API_KEY"))
model = os.getenv("LLM_MODEL")

PROMPT = (
    "Below is a passage from a study document.\n"
    "Write ONE question that this passage clearly answers.\n"
    "Rules:\n"
    "- The question must make sense on its own. Never say 'the passage', "
    "'the text' or 'the document'.\n"
    "- Use your own words; do not copy phrases from the passage.\n"
    "- If the passage is only a table of contents, references, or has no "
    "clear fact, reply exactly: SKIP\n"
    "- Reply with only the question.\n\n"
    "Passage:\n"
)

candidates = [c for c in chunks if len(c["text"]) > 500]
random.shuffle(candidates)

questions = []
for c in candidates:
    if len(questions) >= N_QUESTIONS:
        break
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[{"role": "user", "content": PROMPT + c["text"]}],
        )
    except Exception as e:
        print("API error, waiting 10s:", e)
        time.sleep(10)
        continue

    q = resp.choices[0].message.content.strip()
    q = re.sub(r"<think>.*?</think>", "", q, flags=re.DOTALL).strip()
    if q.upper().startswith("SKIP") or "?" not in q:
        continue

    questions.append({"question": q, "source": c["source"], "page": c["page"]})
    print(len(questions), q)
    time.sleep(1)

OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved {len(questions)} questions to {OUT}")