from pathlib import Path
import pickle
from typing import Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.pkl"
CHUNK_CONFIG_PATH = ARTIFACTS_DIR / "chunk_config.pkl"


def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def split_documents(
    documents: List,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        doc_id = chunk.metadata["doc_id"]
        page_id = chunk.metadata["page_id"]

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{doc_id}_page_{page_id}_chunk_{idx}"

    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks


def chunk_config_matches(
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    if not CHUNK_CONFIG_PATH.exists():
        return False

    old_config = load_pickle(CHUNK_CONFIG_PATH)
    new_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    return old_config == new_config


def get_or_build_chunks(
    documents: List,
    force_rebuild: bool = False,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List:
    need_rebuild = force_rebuild

    if not CHUNKS_PATH.exists():
        need_rebuild = True

    if not chunk_config_matches(chunk_size, chunk_overlap):
        need_rebuild = True

    if need_rebuild:
        print("\n[INFO] Building chunks...")
        chunks = split_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        save_pickle(chunks, CHUNKS_PATH)
        save_pickle(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            CHUNK_CONFIG_PATH,
        )

        print(f"[INFO] Saved chunks to: {CHUNKS_PATH}")
    else:
        print("\n[INFO] Loading chunks from cache...")
        chunks = load_pickle(CHUNKS_PATH)
        print(f"[INFO] Loaded {len(chunks)} chunks from cache")

    return chunks


def debug_chunk_samples(chunks: List, n: int = 3) -> None:
    print("\n===== CHUNK SAMPLES =====")

    for i, chunk in enumerate(chunks[:n]):
        print(f"\n--- Chunk {i+1} ---")
        print("Text preview:")
        print(chunk.page_content[:250].replace("\n", " "))
        print("\nMetadata:")
        for k, v in chunk.metadata.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    from loader import get_or_build_documents, DATA_DIR

    documents = get_or_build_documents(
        folder_path=str(DATA_DIR),
        force_rebuild=False,
        use_api_enrichment=False,
    )

    chunks = get_or_build_chunks(
        documents=documents,
        force_rebuild=False,
        chunk_size=1200,
        chunk_overlap=250,
    )

    debug_chunk_samples(chunks, n=2)
