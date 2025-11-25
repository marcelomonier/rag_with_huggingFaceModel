"""Configuration utilities for the RAG system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    """Convert environment strings to booleans."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    hf_token: Optional[str]
    hf_model_name: str
    embedding_model: str
    device: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_inference_api: bool
    hf_api_url: Optional[str]
    data_raw_dir: Path
    data_processed_dir: Path
    model_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment variables and .env file."""
        load_dotenv()
        cwd = Path(__file__).resolve().parent.parent
        return cls(
            hf_token=os.getenv("HF_TOKEN"),
            hf_model_name=os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            device=os.getenv("DEVICE", "cpu"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "750")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "4")),
            use_inference_api=_str_to_bool(os.getenv("USE_INFERENCE_API"), default=True),
            hf_api_url=os.getenv("HF_API_URL"),
            data_raw_dir=cwd / "data" / "raw",
            data_processed_dir=cwd / "data" / "processed",
            model_dir=cwd / "models",
        )
