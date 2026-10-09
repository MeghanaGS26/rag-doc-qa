"""Answer generation. Works with any OpenAI-compatible API.

Set environment variables (examples):
  Groq:    LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=llama-3.1-8b-instant
  OpenAI:  (leave LLM_BASE_URL unset)                   LLM_MODEL=gpt-4o-mini
  Ollama:  LLM_BASE_URL=http://localhost:11434/v1       LLM_MODEL=llama3.1  LLM_API_KEY=ollama
and always set LLM_API_KEY (except where noted).
"""
import os

from openai import OpenAI

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you used like [1] or [2]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\" Do not use outside knowledge."
)


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{i}] ({c['source']}, p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def answer(question, chunks):
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL") or None,
                    api_key=os.getenv("LLM_API_KEY"))
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_tokens=500,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(question, chunks)}],
    )
    return resp.choices[0].message.content.strip()
