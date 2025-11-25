# RAG HF Demo – Retrieval-Augmented QA with Hugging Face

## Overview
This project is a professional-grade prototype of a Retrieval-Augmented Generation (RAG) pipeline in Python. It ingests local documents (PDF, TXT, MD, CSV), chunks them, generates embeddings, indexes them in a local vector store (numpy-based), and uses a Hugging Face model to answer questions grounded in those documents. It includes a CLI and a simple FastAPI for querying.

## System architecture
High-level flow:
```
Ingestion → Chunking → Embeddings → Vector Store (numpy) → Retrieval → Context Prompt → HF LLM → Answer
```
- **Ingestion:** `pypdf` for PDFs; direct reads for text files. Basic cleanup of newlines and spaces.
- **Chunking:** Configurable blocks (`CHUNK_SIZE`, `CHUNK_OVERLAP`) to preserve context.
- **Embeddings:** `sentence-transformers` (default `sentence-transformers/all-MiniLM-L6-v2`) with normalization for cosine similarity.
- **Vector store:** Embeddings matrix on disk (`embeddings.npy`) + cosine search via numpy.
- **Retrieval:** Fetch top_k most relevant chunks.
- **Prompt:** Concatenate context + instruction to restrict the answer to retrieved text.
- **HF LLM:** Model defined in `HF_MODEL_NAME`, accessed via the Hugging Face Inference API (default) or loaded locally via `transformers`.

## Technologies
- Python 3.10+
- Hugging Face Transformers / Inference Client
- Sentence Transformers (embedding model)
- Numpy (lightweight vector store)
- FastAPI + Uvicorn (optional HTTP API)
- pypdf, python-dotenv, tqdm

## Repository structure
- `src/main.py`: CLI and FastAPI server.
- `src/rag_pipeline.py`: Full pipeline (ingestion, embeddings, vector store, HF generation).
- `src/ingestion.py`: Reading, cleaning, and chunking documents.
- `src/config.py`: Environment/config loading.
- `src/utils.py`: Utilities (prompt, JSONL, directories).
- `data/raw/`: Put input files here.
- `data/processed/`: Reserved for intermediate outputs (not required in this prototype).
- `models/`: Stores `embeddings.npy` and `metadata.pkl`.
- `requirements.txt`: Project dependencies.
- `.env.example`: Example configuration (copy to `.env`).

## Running the project
### 1) Prerequisites
- Python 3.10+ and `pip`
- Hugging Face token (for Inference API or to download private models)

### 2) Install
```bash
pip install -r requirements.txt
```

### 3) Configure `.env`
Copy the example and fill it out:
```bash
cp .env.example .env
```
Key variables:
- `HF_TOKEN`: your Hugging Face token (https://huggingface.co/settings/tokens).
- `HF_MODEL_NAME`: language model (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
- `EMBEDDING_MODEL`: embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- `USE_INFERENCE_API`: `true` to use the Inference API (default, via router.huggingface.co); `false` to load locally with `transformers`.
- `HF_API_URL`: leave empty to use the default router; set only if you need a custom endpoint or alternative model id.
- `DEVICE`: `cpu`, `cuda`, or `mps` (Apple Silicon).
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`: chunking and retrieval params.
Note: some providers expose certain models only as chat (`chat_completion`). If you see “task not supported”, switch to a model supporting `text-generation` (e.g., `HuggingFaceH4/zephyr-7b-beta`, `meta-llama/Meta-Llama-3-8B-Instruct`) or rely on the fallback to `chat_completion`.

### 4) Ingest and index
Place documents in `data/raw/` and run:
```bash
python src/main.py ingest
```
Optional: point to another input directory:
```bash
python src/main.py ingest --input /path/to/docs
```

### 5) Ask questions (CLI)
```bash
python src/main.py query --question "What is the main topic of the documents?"
```
The output shows the answer and the retrieved chunks.

### 6) Run the API (FastAPI)
```bash
python src/main.py api --host 0.0.0.0 --port 8000
```
Example `curl` call:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main content?", "top_k": 3}'
```
Returns JSON with `answer` and `context`.

## Adding new documents to RAG
1. Drop files in `data/raw/` (or another directory).
2. Run `python src/main.py ingest` to rebuild embeddings and index.
3. Query via CLI/API. For large corpora, consider smaller chunks, smaller `top_k`, and watch memory use.

## Switching LLM and embedding models
- Update env vars:
  - `HF_MODEL_NAME`: change to another instruction model (smaller = faster; larger = better quality, higher RAM/VRAM).
  - `EMBEDDING_MODEL`: set another `sentence-transformers` model (larger models → richer embeddings, slower/heavier).
- Considerations:
  - **Latency/cost:** Inference API is convenient but depends on network and may incur costs for paid models.
  - **Hardware:** Local loading requires sufficient GPU/VRAM; CPU works but can be slow.

## Good practices and limitations
- The model can hallucinate if context is weak; the prompt mitigates but doesn’t eliminate this.
- Quality depends on cleaning, chunking, and document coverage.
- Useful tuning:
  - `CHUNK_SIZE`/`CHUNK_OVERLAP`: smaller chunks improve recall but may over-fragment.
  - `TOP_K`: higher can bring more relevant context but also noise.
  - Generation params: `temperature`, `max_new_tokens`.
- Monitor logs; consider storing queries/responses for debugging.

## Next steps / improvements
- Authentication/authorization on the API.
- More robust vector store persistence (versions, multi-tenant).
- Namespace datasets for multiple collections.
- Quality evaluation (user feedback, simple metrics).
- Prompt/response caching and latency/cost monitoring.

## License
MIT License. Feel free to use and adapt with proper attribution.
