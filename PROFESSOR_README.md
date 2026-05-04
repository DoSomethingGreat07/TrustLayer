# TrustLayer Software Submission Notes

This zip contains the TrustLayer software implementation, selected evaluation outputs, and a detailed instruction PDF in `instructions/Final.pdf`.

## What Is Included

- `src/`: Python backend, retrieval, verification, evaluation, and report-generation code.
- `frontend/`: React/FastAPI frontend source.
- `assets/`: project visual assets.
- `artifacts/`: final evaluation question set and selected metrics CSV files.
- `README.md`: project overview and setup instructions.
- `requirements.txt`: Python dependencies.
- `.env.example`: environment variable template.
- `instructions/Final.pdf`: detailed beginner-oriented guide.

## What Is Not Included

The submission does not include:

- private `.env` file or API keys
- local virtual environment
- `node_modules`
- full research-paper PDF corpus under `data/`
- prebuilt Chroma vector database/cache files

To fully rebuild the knowledge base, place PDF files under `data/` before running the pipeline.

## Correct Final Experimental Setup

The final reported setup used:

- Corpus: 94 unique research papers / 95 PDFs
- Chunk size / overlap: `1200 / 250`
- Dense embedding model: `BAAI/bge-base-en-v1.5`
- Vector database: Chroma with HNSW cosine search
- Sparse retrieval: BM25
- Fusion: Reciprocal Rank Fusion
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator: `gpt-4o-mini`, temperature `0`
- Evaluation set: 150 generated questions, 132 answerable and 18 unanswerable

## Setup

```bash
python3 -m venv trustlayer_env
source trustlayer_env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TRUSTLAYER_APP_USERNAME=admin
TRUSTLAYER_APP_PASSWORD=change_me
```

## Run Streamlit Interface

```bash
source trustlayer_env/bin/activate
streamlit run src/app.py
```

## Run React + FastAPI Interface

Terminal 1:

```bash
source trustlayer_env/bin/activate
uvicorn src.react_api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Rebuild / Evaluate

If PDFs are available under `data/`, rebuild and evaluate with:

```bash
source trustlayer_env/bin/activate
python src/evaluate_retrieval_metrics.py \
  --questions-csv artifacts/eval_questions_150_v5_chunks1200.csv \
  --output-csv artifacts/report_metrics_150_v5_chunks1200_bge_k10.csv \
  --ks 1 3 5 10 \
  --dense-k 50 \
  --sparse-k 50 \
  --fusion-k 100 \
  --final-k 10 \
  --chunk-size 1200 \
  --chunk-overlap 250 \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --device cpu
```

## Notes

The included CSV files allow evaluation results to be inspected without rebuilding the full vector database. Full execution requires the local PDF corpus and an OpenAI API key.
