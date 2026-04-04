# TrustLayer

TrustLayer is a trust-aware research assistant for a local research-paper corpus. It combines:

- PDF ingestion
- metadata enrichment
- chunking
- vector retrieval
- BM25 sparse retrieval
- cross-encoder reranking
- corrective retrieval
- grounded answer generation
- verification-based abstention
- an interactive Streamlit interface

The goal is not just to answer questions, but to answer them with visible evidence and a clear trust signal.

## What This Project Does

At a high level, TrustLayer lets you ask questions about a set of research papers stored locally in `data/`. The app:

1. loads PDF pages into documents
2. enriches metadata where possible
3. splits documents into chunks
4. indexes those chunks in Chroma
5. builds a BM25 sparse retriever
6. retrieves candidates using dense + sparse retrieval
7. fuses and reranks those candidates
8. generates an answer strictly from retrieved evidence
9. verifies whether the answer is sufficiently supported
10. either returns the answer or abstains with `Insufficient evidence`

This means the system is designed to prefer caution over unsupported confidence.

## End-to-End Pipeline

### 1. Data Ingestion

Source files live under `data/`, organized by topic:

- `data/llm`
- `data/nlp`
- `data/rag`
- `data/transformers`

Each PDF is loaded page by page using `PyMuPDFLoader` in [loader.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/loader.py).

Important behavior:

- every PDF page becomes a LangChain `Document`
- metadata is derived from filename and page information
- optional Semantic Scholar enrichment can improve title/author metadata
- loaded pages are cached in `artifacts/loaded_pages.pkl`
- the manifest of PDFs is cached in `artifacts/manifest.json`

This cache avoids rebuilding the document list every time the app starts.

### 2. Metadata Enrichment

TrustLayer can enrich paper metadata with:

- filename fallback
- first-page title/author heuristics
- Semantic Scholar lookups

This logic lives in [loader.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/loader.py).

If API enrichment is enabled, the loader tries to resolve better paper metadata and caches it in:

- `artifacts/paper_metadata_cache.json`

### 3. Chunking

Documents are split into retrieval chunks in [chunking.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/chunking.py).

Default chunk parameters:

- `chunk_size = 800`
- `chunk_overlap = 150`

Each chunk gets a stable metadata identity, including:

- `doc_id`
- `page_id`
- `chunk_index`
- `chunk_id`

Chunks are cached to:

- `artifacts/chunks.pkl`
- `artifacts/chunk_config.pkl`

### 4. Embeddings and Vector Store

Embeddings are built via Hugging Face sentence-transformer embeddings in [embeddings.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/embeddings.py).

The vector database is managed in [vector_db.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/vector_db.py) using Chroma.

Behavior:

- if no vector DB exists, TrustLayer builds it from chunks
- if the embedding model or device changes, TrustLayer rebuilds it
- otherwise it reuses the cached database

Artifacts:

- `artifacts/chroma_papers/`
- `artifacts/vectordb_config.json`

### 5. Sparse Retrieval

TrustLayer also builds a BM25 index over chunk text in [main.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/main.py).

This gives a lexical retrieval signal that complements vector similarity.

### 6. Hybrid Retrieval + Reranking

Core retrieval logic lives in [corrective_Rag_pipeline.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/corrective_Rag_pipeline.py).

The retrieval flow is:

1. dense retrieval from Chroma
2. sparse retrieval from BM25
3. reciprocal rank fusion (RRF)
4. cross-encoder reranking

The reranker uses:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

This stage produces a small set of high-priority evidence chunks.

### 7. Corrective Retrieval

If initial retrieval confidence is low, the pipeline does not immediately trust the first retrieval result.

Instead, it performs a corrective step:

- computes retrieval confidence
- expands the query when appropriate
- re-retrieves candidates
- merges candidates
- reranks again

This is how the project tries to recover from weak first-pass retrieval.

### 8. Grounded Answer Generation

Answer generation lives in [llm_generate.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/llm_generate.py).

The model is prompted to:

- use only provided evidence
- avoid outside knowledge
- say `Insufficient evidence` if support is weak
- produce both an answer and justification

The prompt context is built from top retrieved evidence chunks and includes:

- title
- authors
- file
- page
- chunk ID
- chunk content

### 9. Verification and Abstention

TrustLayer does not blindly trust the first generated answer.

It verifies the answer using:

- retrieval confidence
- reranker confidence
- evidence similarity
- evidence coverage
- NLI entailment
- contradiction checks
- combined confidence thresholds

If the answer fails verification, the system abstains and returns:

- `Insufficient evidence`

This logic is also handled in [corrective_Rag_pipeline.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/corrective_Rag_pipeline.py).

### 10. Streamlit Interface

The user-facing application is [app.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/app.py).

The UI provides:

- login gate
- corpus overview
- query input
- answer + justification
- pipeline summary
- verification metrics
- evidence cards
- generator context viewer
- manual precision/recall evaluation
- user-specific chat history

## Main Files and Their Roles

### Application and Pipeline

- [app.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/app.py): Streamlit app and UI
- [main.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/main.py): assembles the retrieval pipeline
- [corrective_Rag_pipeline.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/corrective_Rag_pipeline.py): retrieval, correction, verification, abstention
- [llm_generate.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/llm_generate.py): OpenAI-backed answer generation

### Data Preparation

- [loader.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/loader.py): PDF loading and metadata enrichment
- [chunking.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/chunking.py): chunk creation and chunk caching
- [embeddings.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/embeddings.py): embedding wrapper
- [vector_db.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/vector_db.py): Chroma persistence and loading

### Evaluation / Dataset Generation

- [evaluate_retrieval_metrics.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/evaluate_retrieval_metrics.py)
- [generate_questions.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/generate_questions.py)
- [100_questions.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/100_questions.py)
- [knowledge_base_creation.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/knowledge_base_creation.py)
- [verification.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/verification.py)

## Repository Layout

```text
TrustLayer/
├── data/                   # local paper corpus
├── src/                    # application and pipeline code
├── .streamlit/             # Streamlit config
├── requirements.txt        # Python dependencies
├── .env.example            # environment variable template
└── README.md               # project documentation
```

Generated local outputs are written to `artifacts/` and are intentionally ignored in Git.

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv trustlayer_env
source trustlayer_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then fill in:

- `OPENAI_API_KEY`
- `TRUSTLAYER_APP_USERNAME`
- `TRUSTLAYER_APP_PASSWORD`

### 4. Confirm Your Local Corpus Exists

The app expects PDFs inside the `data/` directory.

If you clone this repo without the original paper corpus, the app may still start but retrieval will not be useful until you add PDFs.

## Running the App

```bash
source trustlayer_env/bin/activate
streamlit run src/app.py
```

## First Run Expectations

On the first real run, TrustLayer may take time to:

- load PDFs
- build document caches
- create chunks
- build the Chroma vector DB
- download local ML model weights if not already cached

Subsequent runs should be much faster because caches and vector DB artifacts are reused.

## How Caching Works

TrustLayer caches several expensive steps locally:

- loaded pages
- file manifest
- metadata cache
- chunks
- chunk configuration
- Chroma vector store
- vector DB configuration

If you change:

- the paper corpus
- chunk size / overlap
- embedding model
- device

you may need to rebuild relevant artifacts.

## Common Local Commands

### Run the Streamlit App

```bash
streamlit run src/app.py
```

### Compile-Check Python Files

```bash
python -m py_compile src/app.py
```

### Rebuild From Scratch

If you want to force regeneration inside code, update the `prepare_pipeline(...)` call in [main.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/main.py) or [app.py](/Users/nikhiljuluri/Desktop/TrustLayer/src/app.py) by toggling:

- `force_rebuild_documents=True`
- `force_rebuild_chunks=True`
- `force_rebuild_vectordb=True`

## Troubleshooting

### The App Always Says `Insufficient evidence`

Possible causes:

- retrieval confidence is genuinely low
- verification thresholds are strict
- the paper corpus does not contain enough relevant evidence
- a required model has not been downloaded yet

Check:

- the `Verification` tab
- the `Evidence` tab
- the `Context Sent to Generator` tab

### Imports Show Warnings in VS Code

Make sure VS Code is using:

- `trustlayer_env/bin/python`

and not the system Python interpreter.

### Streamlit Watcher Warnings

This repo includes a local Streamlit config in:

- [.streamlit/config.toml](/Users/nikhiljuluri/Desktop/TrustLayer/.streamlit/config.toml)

to reduce noisy watcher behavior around large ML libraries.

### Hugging Face Models Need Network Access on First Use

Some retrieval / verification models load lazily. If a model is not already cached locally, the first use may require internet access.

## GitHub / Repo Hygiene

This repo is prepared so that local-only files are not committed:

- `.env`
- virtual environments
- local caches
- generated `artifacts/`

Important:

- never commit `.env`
- rotate any credentials that were ever exposed locally before publishing

## Recommended Commit Contents

Safe, normal contents to publish:

- `src/`
- `data/` if you want to publish the corpus
- `.streamlit/config.toml`
- `requirements.txt`
- `.env.example`
- `README.md`

Usually do not publish:

- `artifacts/`
- `.env`
- `.venv/`
- `trustlayer_env/`

## If You Want to Extend TrustLayer

Good next steps:

- add source filtering by paper or domain
- add citation export
- add evaluation dashboards for retrieval quality
- tune abstention thresholds
- separate production dependencies from notebook/evaluation dependencies
- add GitHub Actions for linting and startup checks

## Summary

TrustLayer is a full local RAG stack for research-paper QA with:

- evidence-first retrieval
- correction when retrieval is weak
- grounded answer generation
- explicit verification and abstention
- an interactive UI for inspecting the full decision path

The project is most useful when you want not only an answer, but a visible reason to trust or distrust that answer.

