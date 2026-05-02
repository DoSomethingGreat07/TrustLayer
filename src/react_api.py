import os
import sys
import json
import re
import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

BASE_DIR = CURRENT_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
METADATA_CACHE_PATH = ARTIFACTS_DIR / "paper_metadata_cache.json"
ANSWER_CACHE_PATH = ARTIFACTS_DIR / "react_answer_cache.json"
ANSWER_CACHE_VERSION = "trustlayer-react-answer-cache-v3-display-metadata"
ANSWER_CACHE_MAX_ENTRIES = 100

load_dotenv()

app = FastAPI(title="TrustLayer React API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2)
    use_api_enrichment: bool = False
    device: str = "cpu"


def _expected_credentials() -> tuple[str, str]:
    return (
        os.getenv("TRUSTLAYER_APP_USERNAME", "admin"),
        os.getenv("TRUSTLAYER_APP_PASSWORD", "change_me"),
    )


def _validate_credentials(username: str | None, password: str | None) -> None:
    expected_username, expected_password = _expected_credentials()
    if username != expected_username or password != expected_password:
        raise HTTPException(status_code=401, detail="Invalid TrustLayer credentials")


@lru_cache(maxsize=1)
def _main_helpers():
    from llm_generate import llm_generate_fn
    from main import get_paper_metadata, prepare_pipeline
    from light_agentic_pipeline import light_agentic_corrective_rag_pipeline

    return (
        light_agentic_corrective_rag_pipeline,
        get_paper_metadata,
        prepare_pipeline,
        llm_generate_fn,
    )


@lru_cache(maxsize=4)
def _get_pipeline(use_api_enrichment: bool, device: str) -> dict[str, Any]:
    _, _, prepare_pipeline, _ = _main_helpers()
    return prepare_pipeline(
        force_rebuild_documents=False,
        force_rebuild_chunks=False,
        force_rebuild_vectordb=False,
        use_api_enrichment=use_api_enrichment,
        chunk_size=1200,
        chunk_overlap=250,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        device=device,
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _stable_json_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _corpus_fingerprint() -> str:
    manifest = _read_json(MANIFEST_PATH, [])
    return _stable_json_hash(manifest)


def _answer_cache_key(payload: AskRequest) -> str:
    key_data = {
        "version": ANSWER_CACHE_VERSION,
        "query": _normalize_query(payload.query),
        "use_api_enrichment": payload.use_api_enrichment,
        "device": payload.device,
        "corpus_fingerprint": _corpus_fingerprint(),
    }
    return _stable_json_hash(key_data)


def _load_answer_cache() -> dict[str, Any]:
    cache = _read_json(ANSWER_CACHE_PATH, {})
    if isinstance(cache, dict):
        return cache
    return {}


def _save_answer_cache(cache: dict[str, Any]) -> None:
    entries = sorted(
        cache.items(),
        key=lambda item: item[1].get("created_at", 0) if isinstance(item[1], dict) else 0,
        reverse=True,
    )[:ANSWER_CACHE_MAX_ENTRIES]
    _write_json(ANSWER_CACHE_PATH, dict(entries))


def _get_cached_answer(cache_key: str) -> dict[str, Any] | None:
    cached = _load_answer_cache().get(cache_key)
    if not isinstance(cached, dict):
        return None
    response = cached.get("response")
    if not isinstance(response, dict):
        return None
    return {
        **response,
        "cache_hit": True,
        "cache_key": cache_key,
    }


def _store_cached_answer(cache_key: str, response: dict[str, Any]) -> None:
    cache = _load_answer_cache()
    cache[cache_key] = {
        "created_at": time.time(),
        "response": {
            **response,
            "cache_hit": False,
            "cache_key": cache_key,
        },
    }
    _save_answer_cache(cache)


def _filename_to_title(file_name: str) -> str:
    title = Path(file_name).stem
    title = re.sub(r"^\d+[_\-\s]*", "", title)
    title = title.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", title).strip() or file_name


def _looks_like_bad_display_title(title: str) -> bool:
    normalized = title.strip().lower().strip(".")
    if not normalized:
        return True

    generic_titles = {
        "preprint",
        "unknown",
        "untitled",
        "abstract",
        "introduction",
        "references",
    }
    if normalized in generic_titles:
        return True

    if re.fullmatch(r"[a-z]+ \d{1,2}, \d{4}", normalized):
        return True

    if re.fullmatch(r"\d{4}", normalized):
        return True

    return False


def _clean_display_title(title: Any, file_name: str) -> str:
    fallback = _filename_to_title(file_name)
    if not isinstance(title, str):
        return fallback

    cleaned = re.sub(r"\s+", " ", title).strip()
    if _looks_like_bad_display_title(cleaned):
        return fallback

    return cleaned


def _authors_to_text(authors: Any) -> str:
    if isinstance(authors, list):
        cleaned = [str(author).strip() for author in authors if str(author).strip()]
        return ", ".join(cleaned) if cleaned else "Unknown"
    if isinstance(authors, str) and authors.strip():
        return authors.strip()
    return "Unknown"


def _serialize_evidence(item: dict[str, Any], idx: int) -> dict[str, Any]:
    _, get_paper_metadata, _, _ = _main_helpers()
    doc = item.get("doc")
    if doc is None:
        return {
            "rank": idx,
            "score": float(item.get("score", 0.0) or 0.0),
            "content": "",
            "metadata": {},
        }

    metadata = get_paper_metadata(doc)
    metadata["paper_title"] = _clean_display_title(
        metadata.get("paper_title"),
        metadata.get("file_name", ""),
    )
    return {
        "rank": idx,
        "score": float(item.get("score", 0.0) or 0.0),
        "content": doc.page_content,
        "metadata": metadata,
    }


def _corpus_summary_from_manifest() -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH, [])
    metadata_cache = _read_json(METADATA_CACHE_PATH, {})
    papers: dict[str, dict[str, Any]] = {}
    domains = set()

    for item in manifest:
        relative_path = item.get("relative_path", "")
        if not relative_path:
            continue

        path = Path(relative_path)
        file_name = path.name
        domain = path.parts[0] if len(path.parts) > 1 else "Unknown"
        cache_key = str(BASE_DIR / "data" / relative_path)
        cached_metadata = metadata_cache.get(cache_key, {})

        raw_title = (
            cached_metadata.get("resolved_title")
            or cached_metadata.get("title")
            or _filename_to_title(file_name)
        )
        title = _clean_display_title(raw_title, file_name)
        authors = _authors_to_text(
            cached_metadata.get("resolved_authors")
            or cached_metadata.get("authors")
        )

        domains.add(domain)
        papers[file_name] = {
            "title": title,
            "authors": authors,
            "domain": domain,
            "file_name": file_name,
            "metadata_source": cached_metadata.get("metadata_source", "manifest"),
        }

    return {
        "document_count": 0,
        "chunk_count": 0,
        "paper_count": len(papers),
        "domains": sorted(domain for domain in domains if domain and domain != "Unknown"),
        "papers": sorted(papers.values(), key=lambda item: (item["domain"], item["title"])),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/login")
def login(payload: LoginRequest) -> dict[str, bool]:
    _validate_credentials(payload.username, payload.password)
    return {"ok": True}


@app.get("/api/corpus")
def corpus(
    x_trustlayer_username: str | None = Header(default=None),
    x_trustlayer_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_credentials(x_trustlayer_username, x_trustlayer_password)
    return _corpus_summary_from_manifest()


@app.post("/api/ask")
def ask(
    payload: AskRequest,
    x_trustlayer_username: str | None = Header(default=None),
    x_trustlayer_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_credentials(x_trustlayer_username, x_trustlayer_password)
    cache_key = _answer_cache_key(payload)
    cached_response = _get_cached_answer(cache_key)
    if cached_response is not None:
        return cached_response

    agentic_pipeline, _, _, llm_generate_fn = _main_helpers()
    pipeline = _get_pipeline(payload.use_api_enrichment, payload.device)
    result = agentic_pipeline(
        query=payload.query,
        vectorstore=pipeline["vectorstore"],
        bm25=pipeline["bm25"],
        chunks=pipeline["chunks"],
        reranker=pipeline["reranker"],
        llm_generate_fn=llm_generate_fn,
    )

    response = {
        "answer": result.get("answer", ""),
        "justification": result.get("justification", ""),
        "corrected": bool(result.get("corrected", False)),
        "used_queries": result.get("used_queries", []),
        "retrieval_confidence": float(result.get("retrieval_confidence", 0.0) or 0.0),
        "verification": result.get("verification", {}),
        "verification_params": result.get("verification_params", {}),
        "evidence": [
            _serialize_evidence(item, idx)
            for idx, item in enumerate(result.get("final_docs", []), start=1)
        ],
        "context": result.get("context", ""),
        "abstained": bool(result.get("abstained", False)),
        "retries_used": int(result.get("retries_used", 0) or 0),
        "pipeline_mode": result.get("pipeline_mode", "light_agentic_corrective_rag"),
        "retrieval_strategy": result.get("retrieval_strategy", {}),
        "cache_hit": False,
        "cache_key": cache_key,
    }
    _store_cached_answer(cache_key, response)
    return response
