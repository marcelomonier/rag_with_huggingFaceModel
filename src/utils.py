"""Helper functions for file handling and prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write a list of dictionaries to JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL rows into a list."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(context_chunks: Sequence[str], question: str) -> str:
    """Construct the final prompt for the LLM."""
    context_text = "\n\n".join(context_chunks)
    return (
        "Você é um assistente especializado. Use APENAS as informações abaixo para responder. "
        "Se a resposta não estiver nos dados, diga que não encontrou evidências.\n\n"
        f"Contexto:\n{context_text}\n\nPergunta: {question}\nResposta:"
    )
