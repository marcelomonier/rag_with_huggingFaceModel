# RAG HF Demo – Sistema de Perguntas e Respostas com Hugging Face e RAG

## Descrição geral
Este projeto é um protótipo profissional de um pipeline de Retrieval-Augmented Generation (RAG) em Python. Ele ingere documentos locais (PDF, TXT, MD, CSV), cria chunks, gera embeddings, indexa em um vector store FAISS e utiliza um modelo de linguagem hospedado no Hugging Face para responder perguntas com base nesses documentos. Inclui uma CLI e uma API FastAPI simples para consultas.

## Arquitetura do sistema
Fluxo em alto nível:
```
Ingestão → Chunking → Embeddings → Vector Store (numpy) → Retrieval → Prompt de Contexto → LLM (HF) → Resposta
```
- **Ingestão:** `pypdf` para PDFs, leitura direta para textos. Limpeza básica de quebras de linha e espaços.
- **Chunking:** Divisão em blocos configuráveis (`CHUNK_SIZE`, `CHUNK_OVERLAP`) para preservar contexto.
- **Embeddings:** `sentence-transformers` (por padrão `sentence-transformers/all-MiniLM-L6-v2`) com normalização para similaridade de cosseno.
- **Vector store:** matriz de embeddings em disco (`embeddings.npy`) + busca por produto interno (cosine) com `numpy`.
- **Retrieval:** Busca dos `top_k` chunks mais relevantes.
- **Prompt:** Contexto concatenado + instrução para restringir a resposta ao conteúdo encontrado.
- **LLM HF:** Modelo definido em `HF_MODEL_NAME`, acessado via Inference API (padrão) ou carregado localmente via `transformers`.

## Tecnologias utilizadas
- Python 3.10+
- Hugging Face Transformers / Inference Client
- Sentence Transformers (modelo de embeddings)
- Numpy (vector store local simplificado)
- FastAPI + Uvicorn (API HTTP opcional)
- pypdf, python-dotenv, tqdm

## Estrutura do repositório
- `src/main.py`: CLI e servidor FastAPI.
- `src/rag_pipeline.py`: Pipeline completo (ingestão, embeddings, FAISS, geração com HF).
- `src/ingestion.py`: Leitura, limpeza e chunking de documentos.
- `src/config.py`: Carregamento de variáveis de ambiente/configuração.
- `src/utils.py`: Utilidades (prompt, JSONL, diretórios).
- `data/raw/`: Coloque aqui os arquivos de entrada.
- `data/processed/`: Espaço reservado para saídas intermediárias (não obrigatório neste protótipo).
- `models/`: Armazena `embeddings.npy` e `metadata.pkl`.
- `requirements.txt`: Dependências do projeto.
- `.env.example`: Exemplo de configuração (copie para `.env`).

## Como rodar o projeto
### 1) Pré-requisitos
- Python 3.10+ e `pip`
- Token do Hugging Face (para Inference API ou para baixar modelos privados)

### 2) Instalação
```bash
pip install -r requirements.txt
```

### 3) Configuração do `.env`
Copie o arquivo de exemplo e preencha:
```bash
cp .env.example .env
```
Variáveis principais:
- `HF_TOKEN`: seu token do Hugging Face (https://huggingface.co/settings/tokens).
- `HF_MODEL_NAME`: modelo de linguagem (ex.: `mistralai/Mistral-7B-Instruct-v0.2`).
- `EMBEDDING_MODEL`: modelo de embedding (ex.: `sentence-transformers/all-MiniLM-L6-v2`).
- `USE_INFERENCE_API`: `true` para usar a Inference API (padrão, via router.huggingface.co), `false` para carregar o modelo localmente via `transformers`.
- `HF_API_URL`: deixe vazio para usar o roteador padrão; se precisar, forneça um endpoint completo ou id de modelo alternativo.
- `DEVICE`: `cpu`, `cuda` ou `mps` (Apple Silicon).
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`: parâmetros de chunking e busca.
Observação: alguns provedores expõem certos modelos apenas como chat (`chat_completion`). Se encontrar erro de task não suportada, troque o modelo por um com `text-generation` (ex.: `HuggingFaceH4/zephyr-7b-beta`, `meta-llama/Meta-Llama-3-8B-Instruct`) ou mantenha e deixe o fallback para `chat_completion`.

### 4) Ingestão e indexação
Coloque documentos em `data/raw/` e execute:
```bash
python src/main.py ingest
```
Opcional: apontar outro diretório de entrada:
```bash
python src/main.py ingest --input /caminho/para/docs
```

### 5) Fazer perguntas (CLI)
```bash
python src/main.py query --question "Qual é o tema principal dos documentos?"
```
Saída exibirá a resposta e os chunks utilizados.

### 6) Subir a API (FastAPI)
```bash
python src/main.py api --host 0.0.0.0 --port 8000
```
Chamada exemplo com `curl`:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é o conteúdo principal?", "top_k": 3}'
```
Retorno: JSON com `answer` e `context`.

## Como adicionar novos documentos ao RAG
1. Adicione os arquivos em `data/raw/` (ou outro diretório).
2. Rode `python src/main.py ingest` para regerar embeddings e índice.
3. Consulte via CLI/API. Para grandes volumes, considere chunk menor, `top_k` menor e monitorar memória.

## Como trocar o modelo de linguagem e o modelo de embedding
- Ajuste as variáveis no `.env` ou no ambiente:
  - `HF_MODEL_NAME`: troque para outro modelo de instrução (menores são mais rápidos; maiores têm melhor qualidade, mas exigem mais RAM/VRAM).
  - `EMBEDDING_MODEL`: altere para outro modelo do `sentence-transformers`. Modelos maiores geram embeddings mais ricos, porém mais lentos/pesados.
- Considerações:
  - **Latência/custo:** Inference API é prática, mas depende de rede e pode ter custo em modelos pagos.
  - **Hardware:** Para carregar localmente, é preciso GPU/VRAM proporcional ao modelo. Em CPU a latência pode ser alta.

## Boas práticas e limitações
- O modelo pode alucinar se o contexto não contiver a resposta; o prompt limita, mas não elimina o risco.
- Qualidade depende da limpeza, chunking e cobertura dos documentos.
- Ajustes úteis:
  - `CHUNK_SIZE`/`CHUNK_OVERLAP`: chunks menores aumentam recall, mas podem fragmentar demais.
  - `TOP_K`: aumentar pode trazer mais contexto relevante, porém com risco de ruído.
  - Parâmetros do gerador: `temperature`, `max_new_tokens`.
- Monitore logs e considere armazenar requisições/respostas para depuração.

## Próximos passos / melhorias sugeridas
- Autenticação/autorização na API.
- Persistência do vector store mais robusta (versões, multi-tenant).
- Separar coleções por namespace/dataset.
- Avaliação de qualidade (feedback do usuário, métricas simples).
- Cache de prompts/respostas e monitoramento de latência/custos.

## Licença
MIT License. Utilize e adapte livremente mantendo os avisos de licença.
