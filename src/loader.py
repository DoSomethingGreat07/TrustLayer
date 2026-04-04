from pathlib import Path
from collections import Counter
import pickle
import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_community.document_loaders import PyMuPDFLoader


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LOADED_PAGES_PATH = ARTIFACTS_DIR / "loaded_pages.pkl"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
PAPER_METADATA_CACHE_PATH = ARTIFACTS_DIR / "paper_metadata_cache.json"

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


# =========================
# SMALL HELPERS
# =========================
def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return load_json(path)
    except Exception:
        return default


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def filename_to_title(filename: str) -> str:
    """
    Example:
    01_Language_Models_are_Few-Shot_Learners.pdf
    -> Language Models are Few-Shot Learners
    """
    name = filename.replace(".pdf", "")
    name = re.sub(r"^\d+[_\-\s]*", "", name)
    name = name.replace("_", " ").replace("-", " ")
    name = normalize_whitespace(name)
    return name


def build_pdf_manifest(folder_path: str) -> List[Dict[str, Any]]:
    folder = Path(folder_path)
    pdf_files = sorted(folder.rglob("*.pdf"))

    manifest = []
    for pdf_path in pdf_files:
        stat = pdf_path.stat()
        manifest.append({
            "relative_path": str(pdf_path.relative_to(folder)),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return manifest


def manifests_match(folder_path: str) -> bool:
    if not MANIFEST_PATH.exists():
        return False
    old_manifest = load_json(MANIFEST_PATH)
    new_manifest = build_pdf_manifest(folder_path)
    return old_manifest == new_manifest


# =========================
# TITLE / AUTHOR EXTRACTION
# =========================
def extract_title_author_from_first_page(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Simple heuristic fallback:
    - title = first non-empty reasonable line
    - author = next non-empty line if it looks like author text
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [normalize_whitespace(line) for line in lines if len(line.strip()) > 2]

    if not lines:
        return None, None

    title = None
    author = None

    for line in lines[:10]:
        if len(line) > 8 and not line.lower().startswith(("abstract", "arxiv", "conference", "proceedings")):
            title = line
            break

    if title:
        title_idx = lines.index(title)
        candidate_lines = lines[title_idx + 1:title_idx + 5]

        for line in candidate_lines:
            lowered = line.lower()
            if "@" in line:
                continue
            if any(token in lowered for token in ["university", "institute", "department", "school", "laboratory", "lab"]):
                continue
            if len(line) < 3:
                continue
            author = line
            break

    return title, author


def fetch_metadata_from_semantic_scholar(
    title_query: str,
    timeout: int = 12,
    sleep_seconds: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """
    No API key required for basic usage.
    Returns best-match paper metadata if available.
    """
    try:
        params = {
            "query": title_query,
            "limit": 1,
            "fields": "title,authors,year,venue"
        }
        response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        papers = data.get("data", [])
        if not papers:
            return None

        paper = papers[0]
        authors = [a.get("name", "").strip() for a in paper.get("authors", []) if a.get("name")]

        time.sleep(sleep_seconds)

        return {
            "title": paper.get("title", "").strip(),
            "authors": authors,
            "year": paper.get("year"),
            "venue": paper.get("venue", "")
        }
    except Exception:
        return None


def get_enriched_paper_metadata(
    pdf_path: Path,
    first_page_text: str,
    use_api_enrichment: bool,
    metadata_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Priority:
    1. cached API result
    2. Semantic Scholar result
    3. first-page heuristic
    4. filename fallback
    """
    cache_key = str(pdf_path)

    if cache_key in metadata_cache:
        return metadata_cache[cache_key]

    title_from_filename = filename_to_title(pdf_path.name)
    title_from_page, author_from_page = extract_title_author_from_first_page(first_page_text)

    result = {
        "resolved_title": title_from_filename,
        "resolved_authors": ["Unknown"],
        "year": None,
        "venue": "",
        "metadata_source": "filename_fallback",
    }

    if title_from_page:
        result["resolved_title"] = title_from_page
        result["metadata_source"] = "first_page_heuristic"

    if author_from_page:
        result["resolved_authors"] = [author_from_page]

    if use_api_enrichment:
        query = title_from_page or title_from_filename
        api_result = fetch_metadata_from_semantic_scholar(query)
        if api_result and api_result.get("title"):
            result = {
                "resolved_title": api_result["title"],
                "resolved_authors": api_result["authors"] if api_result["authors"] else ["Unknown"],
                "year": api_result.get("year"),
                "venue": api_result.get("venue", ""),
                "metadata_source": "semantic_scholar",
            }

    metadata_cache[cache_key] = result
    return result


# =========================
# MAIN LOADER
# =========================
def load_documents_from_pdfs(
    folder_path: str,
    skip_empty_pages: bool = True,
    min_words: int = 5,
    debug_samples: int = 3,
    use_api_enrichment: bool = False,
) -> List:
    """
    Build page-level documents from PDFs.
    Each page becomes one LangChain Document.
    """
    documents = []
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    pdf_files = sorted(folder.rglob("*.pdf"))
    metadata_cache = safe_load_json(PAPER_METADATA_CACHE_PATH, default={})

    total_files = 0
    total_kept_pages = 0
    total_skipped_pages = 0

    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
            total_files += 1

            if not pages:
                print(f"[WARN] No pages found in {pdf_path.name}")
                continue

            doc_id = str(pdf_path.relative_to(folder)).replace("/", "_").replace(".pdf", "")
            domain = pdf_path.parent.name
            relative_path = str(pdf_path.relative_to(folder))

            first_page_text = pages[0].page_content if pages else ""
            enriched = get_enriched_paper_metadata(
                pdf_path=pdf_path,
                first_page_text=first_page_text,
                use_api_enrichment=use_api_enrichment,
                metadata_cache=metadata_cache,
            )

            total_pages = pages[0].metadata.get("total_pages", len(pages))

            for page_idx, page in enumerate(pages):
                content = page.page_content.strip()

                if skip_empty_pages and (not content or len(content.split()) < min_words):
                    total_skipped_pages += 1
                    continue

                page.metadata = {
                    "doc_id": doc_id,
                    "domain": domain,
                    "pdf_path": str(pdf_path),
                    "relative_path": relative_path,
                    "file_name": pdf_path.name,
                    "page_id": page_idx,
                    "page_number": page_idx + 1,
                    "total_pages": total_pages,
                    "source": str(pdf_path),
                    "title": enriched["resolved_title"],
                    "authors": enriched["resolved_authors"],
                    "year": enriched["year"],
                    "venue": enriched["venue"],
                    "metadata_source": enriched["metadata_source"],
                }

                documents.append(page)
                total_kept_pages += 1

            print(f"[OK] Loaded {pdf_path.name} ({len(pages)} pages)")

        except Exception as e:
            print(f"[ERROR] Could not load {pdf_path}: {e}")

    save_json(metadata_cache, PAPER_METADATA_CACHE_PATH)

    print("\n===== LOAD SUMMARY =====")
    print(f"PDF files processed: {total_files}")
    print(f"Pages kept: {total_kept_pages}")
    print(f"Pages skipped: {total_skipped_pages}")

    domains = [doc.metadata["domain"] for doc in documents]
    domain_counts = Counter(domains)

    print("\n===== DOMAIN DISTRIBUTION =====")
    for d, c in sorted(domain_counts.items()):
        print(f"{d}: {c} pages")

    if documents:
        print("\n===== SAMPLE DOCUMENTS =====")
        samples = random.sample(documents, min(debug_samples, len(documents)))
        for i, doc in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print("Text preview:")
            print(doc.page_content[:200].replace("\n", " "))
            print("\nMetadata:")
            for k, v in doc.metadata.items():
                print(f"{k}: {v}")

    return documents


def get_or_build_documents(
    folder_path: str,
    force_rebuild: bool = False,
    skip_empty_pages: bool = True,
    min_words: int = 5,
    debug_samples: int = 3,
    use_api_enrichment: bool = False,
):
    """
    Load saved page-level artifact if available and valid.
    Otherwise rebuild from PDFs and save artifacts.
    """
    need_rebuild = force_rebuild

    if not LOADED_PAGES_PATH.exists():
        need_rebuild = True

    if not manifests_match(folder_path):
        need_rebuild = True

    if need_rebuild:
        print("\n[INFO] Building documents from PDFs...")
        documents = load_documents_from_pdfs(
            folder_path=folder_path,
            skip_empty_pages=skip_empty_pages,
            min_words=min_words,
            debug_samples=debug_samples,
            use_api_enrichment=use_api_enrichment,
        )

        save_pickle(documents, LOADED_PAGES_PATH)
        save_json(build_pdf_manifest(folder_path), MANIFEST_PATH)

        print(f"\n[INFO] Saved page-level artifact to: {LOADED_PAGES_PATH}")
        print(f"[INFO] Saved manifest to: {MANIFEST_PATH}")
        print(f"[INFO] Saved paper metadata cache to: {PAPER_METADATA_CACHE_PATH}")
    else:
        print("\n[INFO] Loading page-level documents from saved artifact...")
        documents = load_pickle(LOADED_PAGES_PATH)
        print(f"[INFO] Loaded {len(documents)} documents from cache")

    return documents


# =========================
# DEBUG HELPERS
# =========================
def check_metadata_samples(documents, n: int = 3) -> None:
    print("\n===== RANDOM METADATA SAMPLES =====")
    samples = random.sample(documents, min(n, len(documents)))
    for i, doc in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        for k, v in doc.metadata.items():
            print(f"{k}: {v}")
        print("\nText preview:")
        print(doc.page_content[:150].replace("\n", " "))


def check_missing_metadata(documents) -> None:
    print("\n===== CHECKING MISSING METADATA =====")
    required_fields = [
        "doc_id",
        "domain",
        "pdf_path",
        "relative_path",
        "file_name",
        "page_id",
        "page_number",
        "total_pages",
        "source",
        "title",
        "authors",
        "metadata_source",
    ]

    missing = 0
    for doc in documents:
        for field in required_fields:
            if field not in doc.metadata:
                missing += 1
                print(f"Missing field: {field}")

    print(f"Total missing fields: {missing}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    documents = get_or_build_documents(
        folder_path=str(DATA_DIR),
        force_rebuild=False,
        skip_empty_pages=True,
        min_words=5,
        debug_samples=2,
        use_api_enrichment=True,   
    )

    print("\n===== FINAL METADATA CHECK =====")
    if documents:
        print(documents[88].metadata)

    check_missing_metadata(documents)
    check_metadata_samples(documents, n=2)