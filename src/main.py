"""CLI and optional API entrypoint for the RAG demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from config import Settings
from rag_pipeline import RAGPipeline


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


def create_api(pipeline: RAGPipeline) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(title="RAG HF Demo", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        answer, retrieved = pipeline.answer_question(req.question, top_k=req.top_k)
        return {"answer": answer, "context": retrieved}

    return app


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline powered by Hugging Face models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents and build the vector store.")
    ingest_parser.add_argument("--input", type=str, default=None, help="Diretório com arquivos para ingerir.")

    query_parser = subparsers.add_parser("query", help="Responder uma pergunta usando o índice.")
    query_parser.add_argument("--question", required=True, help="Pergunta do usuário.")
    query_parser.add_argument("--top_k", type=int, default=None, help="Número de chunks a recuperar.")

    api_parser = subparsers.add_parser("api", help="Subir API FastAPI para consultas.")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host do servidor.")
    api_parser.add_argument("--port", type=int, default=8000, help="Porta do servidor.")

    args = parser.parse_args()
    settings = Settings.load()
    pipeline = RAGPipeline(settings)

    if args.command == "ingest":
        pipeline.full_ingest_and_index(Path(args.input) if args.input else None)
        print("Ingestão e indexação concluídas.")
    elif args.command == "query":
        answer, retrieved = pipeline.answer_question(args.question)
        print("----- Resposta -----")
        print(answer)
        print("\n----- Contexto usado -----")
        for item in retrieved:
            print(f"[{item['score']:.3f}] {item['source']} (chunk {item['chunk_id']})")
    elif args.command == "api":
        app = create_api(pipeline)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
