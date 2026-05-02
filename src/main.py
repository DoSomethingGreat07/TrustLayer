# main.py

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from loader import get_or_build_documents, DATA_DIR
from chunking import get_or_build_chunks
from vector_db import get_or_build_vectorstore
from corrective_Rag_pipeline import corrective_rag_pipeline_v2
from llm_generate import llm_generate_fn





# ============================================================
# SPARSE RETRIEVAL SETUP
# ============================================================
def build_bm25_index(chunks):
    tokenized_docs = [
        chunk.page_content.split()
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25


# ============================================================
# RERANKER SETUP
# ============================================================
def build_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)


# ============================================================
# METADATA HELPERS
# ============================================================
def format_authors(authors):
    if authors is None:
        return "Unknown"

    if isinstance(authors, list):
        cleaned = [str(a).strip() for a in authors if str(a).strip()]
        if not cleaned:
            return "Unknown"
        return ", ".join(cleaned)

    if isinstance(authors, str):
        authors = authors.strip()
        return authors if authors else "Unknown"

    return str(authors)


def get_paper_metadata(doc):
    meta = doc.metadata

    paper_title = (
        meta.get("title")
        or meta.get("file_name")
        or meta.get("doc_id")
        or "Unknown"
    )

    authors = format_authors(meta.get("authors"))
    metadata_source = meta.get("metadata_source", "unknown")

    return {
        "paper_title": paper_title,
        "authors": authors,
        "metadata_source": metadata_source,
        "file_name": meta.get("file_name", "Unknown"),
        "page_number": meta.get("page_number", "Unknown"),
        "chunk_id": meta.get("chunk_id", "Unknown"),
        "domain": meta.get("domain", "Unknown"),
    }


# ============================================================
# DEBUG / DISPLAY HELPERS
# ============================================================
def print_result(result: dict):
    print("\n" + "=" * 90)
    print("FINAL ANSWER")
    print("=" * 90)
    print(result["answer"])

    print("\n" + "=" * 90)
    print("PIPELINE STATS")
    print("=" * 90)
    print(f"Corrected: {result['corrected']}")
    print(f"Used Queries: {result['used_queries']}")
    print(f"Retrieval Confidence: {result['retrieval_confidence']:.4f}")
    print(f"Verification: {result['verification']}")

    print("\n" + "=" * 90)
    print("TOP EVIDENCE WITH PAPER METADATA")
    print("=" * 90)

    for idx, item in enumerate(result["final_docs"], start=1):
        doc = item["doc"]
        info = get_paper_metadata(doc)

        print(f"\n--- Evidence {idx} ---")
        print(f"Paper Title      : {info['paper_title']}")
        print(f"Authors          : {info['authors']}")
        print(f"Metadata Source  : {info['metadata_source']}")
        print(f"Domain           : {info['domain']}")
        print(f"File Name        : {info['file_name']}")
        print(f"Page Number      : {info['page_number']}")
        print(f"Chunk ID         : {info['chunk_id']}")
        print(f"Rerank Score     : {item.get('score', 0.0):.4f}")
        print("Text Preview     :")
        print(doc.page_content[:500].replace("\n", " "))

    print("\n" + "=" * 90)
    print("CONTEXT SENT TO GENERATOR")
    print("=" * 90)
    print(result["context"][:2500])
    print("\nFINAL ANSWER:")
    print(result["answer"])

    print("\nJUSTIFICATION:")
    print(result["justification"])


# ============================================================
# PIPELINE PREPARATION
# ============================================================
def prepare_pipeline(
    force_rebuild_documents: bool = False,
    force_rebuild_chunks: bool = False,
    force_rebuild_vectordb: bool = False,
    use_api_enrichment: bool = False,
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
):
    print("\n[INFO] Loading/building documents...")
    documents = get_or_build_documents(
        folder_path=str(DATA_DIR),
        force_rebuild=force_rebuild_documents,
        use_api_enrichment=use_api_enrichment,
    )

    print("\n[INFO] Loading/building chunks...")
    chunks = get_or_build_chunks(
        documents=documents,
        force_rebuild=force_rebuild_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("\n[INFO] Loading/building vector DB...")
    vectorstore, embeddings = get_or_build_vectorstore(
        chunks=chunks,
        force_rebuild=force_rebuild_vectordb,
        embedding_model=embedding_model,
        device=device,
    )

    print("\n[INFO] Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    print("\n[INFO] Loading reranker...")
    reranker = build_reranker()

    return {
        "documents": documents,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "embeddings": embeddings,
        "bm25": bm25,
        "reranker": reranker,
    }


# ============================================================
# ASK QUESTION
# ============================================================
def ask_question(
    query: str,
    vectorstore,
    bm25,
    chunks,
    reranker,
):
    result = corrective_rag_pipeline_v2(
    query=query,
    vectorstore=vectorstore,
    bm25=bm25,
    chunks=chunks,
    reranker=reranker,
    llm_generate_fn=llm_generate_fn,
    dense_k=20,
    sparse_k=20,
    fusion_k=50,
    final_k=5,
    max_retries=2,
)
    return result


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pipeline = prepare_pipeline(
        force_rebuild_documents=False,
        force_rebuild_chunks=False,
        force_rebuild_vectordb=False,
        use_api_enrichment=True,   # set True if you want API-enriched paper title/authors
        chunk_size=1200,
        chunk_overlap=250,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        device="cpu",
    )

    query = "What is corrective RAG?"

    result = ask_question(
        query=query,
        vectorstore=pipeline["vectorstore"],
        bm25=pipeline["bm25"],
        chunks=pipeline["chunks"],
        reranker=pipeline["reranker"],
    )

    print_result(result)
