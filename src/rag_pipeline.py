"""End-to-end RAG pipeline: ingestion, embeddings, vector store, and generation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import Settings
from ingestion import Document, chunk_documents, load_documents
from utils import build_prompt, ensure_dir


class RAGPipeline:
    """RAG pipeline with simple vector store (numpy) and Hugging Face LLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        ensure_dir(self.settings.data_processed_dir)
        ensure_dir(self.settings.model_dir)
        self.embedding_model = SentenceTransformer(
            self.settings.embedding_model, device=self.settings.device
        )
        self.embeddings: np.ndarray | None = None
        self.metadata: List[Dict] = []
        self.generation_client = None
        self.local_pipeline = None

    def ingest(self, input_dir: Path | None = None) -> List[Document]:
        """Load and clean raw documents."""
        dir_to_use = input_dir or self.settings.data_raw_dir
        return load_documents(dir_to_use)

    def chunk(self, docs: Sequence[Document]) -> List[Dict]:
        """Chunk documents into overlapping pieces."""
        return chunk_documents(
            docs, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Create embeddings for a list of strings."""
        vectors = self.embedding_model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.astype("float32")

    def build_vector_store(self, chunks: Sequence[Dict]) -> None:
        """Build in-memory index from text chunks."""
        self.embeddings = self.embed_texts([c["text"] for c in chunks])
        self.metadata = list(chunks)

    def save_index(self) -> None:
        """Persist embeddings and metadata."""
        if self.embeddings is None:
            raise ValueError("Índice não inicializado.")
        index_path = self.settings.model_dir / "embeddings.npy"
        meta_path = self.settings.model_dir / "metadata.pkl"
        np.save(index_path, self.embeddings)
        with meta_path.open("wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self) -> None:
        """Load embeddings and metadata from disk."""
        index_path = self.settings.model_dir / "embeddings.npy"
        meta_path = self.settings.model_dir / "metadata.pkl"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Index ou metadata não encontrados. Rode ingest e index primeiro.")
        self.embeddings = np.load(index_path)
        with meta_path.open("rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int | None = None) -> List[Dict]:
        """Retrieve relevant chunks given a question."""
        if self.embeddings is None:
            raise ValueError("Índice não carregado.")
        query_vec = self.embed_texts([query])
        k = top_k or self.settings.top_k
        # embeddings and query_vec are normalized; dot product == cosine similarity
        sims = np.dot(self.embeddings, query_vec[0])
        top_idx = np.argsort(sims)[::-1][:k]
        results: List[Dict] = []
        for idx in top_idx:
            meta = self.metadata[int(idx)]
            results.append(
                {
                    "score": float(sims[int(idx)]),
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                }
            )
        return results

    # --- Generation -----------------------------------------------------
    def _load_generation_client(self):
        if self.settings.use_inference_api:
            model_id = self.settings.hf_model_name
            if self.settings.hf_api_url:
                # If user provides a full endpoint (e.g., router URL), use it as the model id.
                model_id = self.settings.hf_api_url
            self.generation_client = InferenceClient(
                model=model_id,
                token=self.settings.hf_token,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.settings.hf_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.settings.hf_model_name, device_map="auto"
            )
            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
            )

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        """Call the Hugging Face model to produce text."""
        if self.generation_client is None and self.local_pipeline is None:
            self._load_generation_client()

        if self.settings.use_inference_api:
            assert self.generation_client is not None
            try:
                response = self.generation_client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    model=self.settings.hf_api_url or self.settings.hf_model_name,
                )
                if isinstance(response, str):
                    return response
                # TextGenerationOutput
                return response.generated_text
            except ValueError as err:
                # Some providers expose chat-only interfaces; fallback to chat_completion
                if "not supported for task text-generation" not in str(err):
                    raise
                chat_resp = self.generation_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "Você é um assistente que responde somente com base no contexto fornecido."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    model=self.settings.hf_api_url or self.settings.hf_model_name,
                )
                choice = chat_resp.choices[0]
                message = choice.message
                if isinstance(message, dict):
                    return message.get("content", "")
                return getattr(message, "content", "")
        assert self.local_pipeline is not None
        outputs = self.local_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return outputs[0]["generated_text"]

    # --- High level -----------------------------------------------------
    def answer_question(self, question: str, top_k: int | None = None) -> Tuple[str, List[Dict]]:
        """Full RAG flow: search and generate answer."""
        if self.embeddings is None:
            self.load_index()
        retrieved = self.search(question, top_k=top_k or self.settings.top_k)
        prompt = build_prompt([r["text"] for r in retrieved], question)
        answer = self.generate(prompt)
        return answer, retrieved

    def full_ingest_and_index(self, input_dir: Path | None = None) -> None:
        """Convenience method to run ingestion + chunking + index build."""
        docs = self.ingest(input_dir)
        chunks = self.chunk(docs)
        self.build_vector_store(chunks)
        self.save_index()
