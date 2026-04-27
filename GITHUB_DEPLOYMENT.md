# GitHub Deployment Checklist

## Safe to Commit

- `src/`
- `frontend/`
- `assets/`
- `.streamlit/config.toml`
- `.github/workflows/ci.yml`
- `requirements.txt`
- `.env.example`
- `README.md`
- `GITHUB_DEPLOYMENT.md`

## Do Not Commit

- `.env`
- `data/`
- `artifacts/`
- `.venv/`
- `trustlayer_env/`
- `frontend/node_modules/`
- `frontend/dist/`

These are already ignored by `.gitignore`.

## Local Run Commands

Streamlit:

```bash
python3 -m venv trustlayer_env
source trustlayer_env/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

React API backend:

```bash
source trustlayer_env/bin/activate
uvicorn src.react_api:app --host 127.0.0.1 --port 8000
```

React frontend:

```bash
cd frontend
npm install
npm run dev
```

## Required Environment Variables

Copy `.env.example` to `.env` and set:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TRUSTLAYER_APP_USERNAME=admin
TRUSTLAYER_APP_PASSWORD=change_me
```

## Corpus Note

The paper corpus is intentionally not committed. Add PDFs under `data/` locally,
or run `src/knowledge_base_creation.py` to build a corpus before using retrieval.
