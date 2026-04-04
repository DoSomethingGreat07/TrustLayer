# retrieval_pipeline.py

def hybrid_retrieve(
    query,
    vectorstore,
    bm25,
    chunks,
    reranker
):

    # Dense
    dense_docs = dense_retrieve(
        vectorstore,
        query,
        k=20
    )

    # Sparse
    sparse_docs = sparse_retrieve(
        bm25,
        chunks,
        query,
        k=20
    )

    # Fusion
    fused_docs = rrf_fusion(
        dense_docs,
        sparse_docs,
        top_k=50
    )

    # Reranking
    final_docs = rerank_documents(
        reranker,
        query,
        fused_docs,
        top_k=5
    )

    return final_docs