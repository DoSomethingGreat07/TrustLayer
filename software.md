# TrustLayer Software Guide

## 1. Purpose

This document provides a complete software guide for running, evaluating, packaging, and deploying TrustLayer.

TrustLayer is a verification-aware RAG system for local research-paper question answering. The system supports:

- local PDF ingestion and metadata enrichment
- chunking and vector indexing
- hybrid retrieval (dense + BM25 + fusion)
- cross-encoder reranking
- grounded answer generation
- verification and abstention
- Streamlit and React/FastAPI interfaces

## 2. Repository

GitHub repository:

- https://github.com/DoSomethingGreat07/TrustLayer

Main folders:

- `src/` - core backend pipeline and evaluation scripts
- `frontend/` - React TypeScript UI
- `assets/` - static assets used in docs/README
- `data/` - local PDF corpus (not committed)
- `artifacts/` - generated caches/results (not committed)

## 3. System Requirements

- OS: macOS/Linux (Windows works with equivalent commands)
- Python: 3.11 recommended
- Node.js: 20+ (or 22+)
- npm: compatible with your Node version
- OpenAI API key

## 4. Initial Setup

From repository root:

```bash
python3 -m venv trustlayer_env
source trustlayer_env/bin/activate
pip install -r requirements.txt
```

Frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## 5. Environment Variables

Create local environment file:

```bash
cp .env.example .env
```

Set required values in `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TRUSTLAYER_APP_USERNAME=admin
TRUSTLAYER_APP_PASSWORD=change_me
```

Security note:

- Never commit `.env`.

## 6. Data Preparation

TrustLayer expects PDFs inside `data/` (domain subfolders are supported). Example:

```text
data/
  llm/
  nlp/
  rag/
  transformers/
```

Optional helper for starter corpus generation:

```bash
source trustlayer_env/bin/activate
python src/knowledge_base_creation.py
```

## 7. Run Applications

### 7.1 Streamlit UI

```bash
source trustlayer_env/bin/activate
streamlit run src/app.py
```

Default URL is usually:

- http://localhost:8501

### 7.2 React + FastAPI UI

Run backend and frontend in separate terminals.

Backend:

```bash
source trustlayer_env/bin/activate
uvicorn src.react_api:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Frontend default URL:

- http://localhost:5173

Important runtime note:

- Avoid `--reload` for FastAPI when virtualenv is inside repo; file watching can cause repeated reload during model loading.

## 8. Evaluation Workflow

### 8.1 Generate evaluation questions

```bash
source trustlayer_env/bin/activate
trustlayer_env/bin/python src/generate_eval_questions.py \
  --target-questions 150 \
  --candidate-questions 600 \
  --target-papers 95 \
  --chunks-per-paper 12 \
  --output-csv artifacts/eval_questions_150_v5_chunks1200.csv
```

### 8.2 Run retrieval + verification metrics

Example (MPNet):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 trustlayer_env/bin/python src/evaluate_retrieval_metrics.py \
  --questions-csv artifacts/eval_questions_150_v5_chunks1200.csv \
  --output-csv artifacts/report_metrics_150_v5_chunks1200_mpnet_k10_with_verification.csv \
  --ks 1 3 5 10 \
  --dense-k 50 \
  --sparse-k 50 \
  --fusion-k 100 \
  --final-k 10 \
  --chunk-size 1200 \
  --chunk-overlap 250 \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --device cpu \
  --include-answer-metrics
```

Example (BGE-base):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 trustlayer_env/bin/python src/evaluate_retrieval_metrics.py \
  --questions-csv artifacts/eval_questions_150_v5_chunks1200.csv \
  --output-csv artifacts/report_metrics_150_v5_chunks1200_bge_k10_with_verification.csv \
  --ks 1 3 5 10 \
  --dense-k 50 \
  --sparse-k 50 \
  --fusion-k 100 \
  --final-k 10 \
  --chunk-size 1200 \
  --chunk-overlap 250 \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --device cpu \
  --include-answer-metrics
```

### 8.3 Generate report artifacts

```bash
source trustlayer_env/bin/activate
trustlayer_env/bin/python src/generate_report_artifacts.py
```

## 9. GitHub Deployment Checklist

### 9.1 Files safe to commit

- `src/`
- `frontend/`
- `assets/`
- `.streamlit/config.toml`
- `.github/workflows/ci.yml`
- `requirements.txt`
- `.env.example`
- `README.md`
- `GITHUB_DEPLOYMENT.md`
- `software.md`

### 9.2 Files/folders to exclude

- `.env`
- `data/`
- `artifacts/`
- `.venv/`
- `trustlayer_env/`
- `frontend/node_modules/`
- `frontend/dist/`
- local logs

### 9.3 Typical deployment commands

```bash
git status
git add src frontend README.md GITHUB_DEPLOYMENT.md requirements.txt .env.example .gitignore assets software.md

git commit -m "Prepare repository for deployment"
git push origin main
```

## 10. Course Software Submission Package

Create a clean zip bundle:

```bash
zip -r TrustLayer_software.zip \
  src frontend assets .streamlit .github \
  README.md GITHUB_DEPLOYMENT.md requirements.txt .env.example software.md \
  -x "trustlayer_env/*" \
     ".venv/*" \
     "frontend/node_modules/*" \
     "frontend/dist/*" \
     "data/*" \
     "artifacts/*" \
     "*.log"
```

## 11. Troubleshooting

### 11.1 First query is slow

Expected behavior. Models and retrieval artifacts are loaded/warmed on first run.

### 11.2 FastAPI restarts repeatedly

Run without `--reload`.

### 11.3 Weak answers or abstentions

`Insufficient evidence` is expected when verification gates fail. Check corpus coverage, chunking/retrieval settings, and evidence quality.

## 12. Reproducibility Notes

- Runtime defaults and evaluation settings are documented in `README.md`.
- Generated outputs under `artifacts/` are intentionally local-only and should be reproduced via scripts.
- Keep the backend process alive for stable latency and cache reuse.
