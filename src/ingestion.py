"""Document ingestion and chunking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


@dataclass
class Document:
    """Representation of a document loaded from disk."""

    source: str
    text: str


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(input_dir: Path) -> List[Document]:
    """Load all supported files from a directory."""
    docs: List[Document] = []
    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.suffix.lower() == ".pdf":
            raw_text = _read_pdf(file_path)
        else:
            raw_text = _read_text_file(file_path)
        cleaned = clean_text(raw_text)
        docs.append(Document(source=str(file_path), text=cleaned))
    if not docs:
        raise ValueError(f"Nenhum documento suportado encontrado em {input_dir}")
    return docs


def clean_text(text: str) -> str:
    """Normalize text to improve downstream chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 750, overlap: int = 150) -> List[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size deve ser positivo e overlap não-negativo.")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(docs: Sequence[Document], chunk_size: int, overlap: int) -> List[dict]:
    """Chunk documents and attach metadata."""
    all_chunks: List[dict] = []
    for doc in docs:
        for idx, chunk in enumerate(chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)):
            all_chunks.append(
                {"source": doc.source, "chunk_id": idx, "text": chunk}
            )
    return all_chunks
