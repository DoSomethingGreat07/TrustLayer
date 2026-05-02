from pathlib import Path
import json
import shutil
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from embeddings import build_sentence_transformer_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CHROMA_DIR = ARTIFACTS_DIR / "chroma_papers"
VECTORDB_CONFIG_PATH = ARTIFACTS_DIR / "vectordb_config.json"
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def vectordb_config_matches(model_name: str, device: str) -> bool:
    if not VECTORDB_CONFIG_PATH.exists():
        return False

    old_config = load_json(VECTORDB_CONFIG_PATH)
    new_config = {
        "embedding_model": model_name,
        "device": device,
        "collection_metadata": CHROMA_COLLECTION_METADATA,
    }
    return old_config == new_config


def reset_chroma_dir() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def get_or_build_vectorstore(
    chunks: List,
    force_rebuild: bool = False,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
) -> Tuple[Chroma, Any]:
    embeddings = build_sentence_transformer_embeddings(
        model_name=embedding_model,
        device=device,
    )

    need_rebuild = force_rebuild

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        need_rebuild = True

    if not vectordb_config_matches(embedding_model, device):
        need_rebuild = True

    if need_rebuild:
        print("\n[INFO] Building Chroma DB...")

        if CHROMA_DIR.exists():
            reset_chroma_dir()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )

        save_json(
            {
                "embedding_model": embedding_model,
                "device": device,
                "collection_metadata": CHROMA_COLLECTION_METADATA,
            },
            VECTORDB_CONFIG_PATH,
        )

        print(f"[INFO] Saved Chroma DB to: {CHROMA_DIR}")
    else:
        print("\n[INFO] Loading existing Chroma DB from cache...")

        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )

    return vectorstore, embeddings


def debug_retrieval(vectorstore: Chroma, query: str, k: int = 5) -> None:
    print("\n===== RETRIEVAL DEBUG =====")

    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print("Title:", doc.metadata.get("title"))
        print("File:", doc.metadata.get("file_name"))
        print("Page:", doc.metadata.get("page_number"))
        print("Chunk ID:", doc.metadata.get("chunk_id"))
        print("Text preview:")
        print(doc.page_content[:300].replace("\n", " "))
