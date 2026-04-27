from typing import Any, Dict

from corrective_Rag_pipeline import corrective_rag_pipeline_v2


def plan_retrieval_strategy(query: str) -> Dict[str, Any]:
    """
    Lightweight agentic router.

    This intentionally avoids a multi-agent framework. It only chooses retrieval
    knobs before handing control to the existing corrective RAG pipeline.
    """
    q = query.lower().strip()
    tokens = q.split()

    strategy = {
        "name": "balanced_hybrid",
        "reason": "Default hybrid search for general research questions.",
        "dense_k": 20,
        "sparse_k": 20,
        "fusion_k": 50,
        "final_k": 5,
        "max_retries": 2,
    }

    definition_cues = ("what is", "define", "meaning of", "overview", "explain")
    comparison_cues = ("compare", "difference", "versus", " vs ", "tradeoff", "better than")
    evidence_cues = ("evidence", "support", "cite", "citation", "which paper", "according to")
    method_cues = ("how", "why", "improve", "work", "architecture", "pipeline")

    if any(cue in q for cue in definition_cues):
        strategy.update({
            "name": "definition_broad_context",
            "reason": "Definition-style questions benefit from broader semantic recall.",
            "dense_k": 28,
            "sparse_k": 16,
            "fusion_k": 60,
            "final_k": 6,
        })
    elif any(cue in q for cue in comparison_cues):
        strategy.update({
            "name": "comparison_more_evidence",
            "reason": "Comparison questions need more final evidence to cover both sides.",
            "dense_k": 28,
            "sparse_k": 24,
            "fusion_k": 70,
            "final_k": 7,
            "max_retries": 2,
        })
    elif any(cue in q for cue in evidence_cues):
        strategy.update({
            "name": "citation_precision",
            "reason": "Evidence-seeking questions favor precise lexical matches and citations.",
            "dense_k": 18,
            "sparse_k": 30,
            "fusion_k": 60,
            "final_k": 6,
        })
    elif any(cue in q for cue in method_cues):
        strategy.update({
            "name": "mechanism_explanation",
            "reason": "Mechanism questions need enough context for processes and causality.",
            "dense_k": 24,
            "sparse_k": 22,
            "fusion_k": 60,
            "final_k": 6,
        })

    if len(tokens) <= 4:
        strategy["dense_k"] = max(strategy["dense_k"], 28)
        strategy["fusion_k"] = max(strategy["fusion_k"], 60)
        strategy["reason"] += " Short queries get wider recall."

    return strategy


def light_agentic_corrective_rag_pipeline(
    query: str,
    vectorstore,
    bm25,
    chunks,
    reranker,
    llm_generate_fn,
) -> Dict:
    strategy = plan_retrieval_strategy(query)

    result = corrective_rag_pipeline_v2(
        query=query,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        llm_generate_fn=llm_generate_fn,
        dense_k=strategy["dense_k"],
        sparse_k=strategy["sparse_k"],
        fusion_k=strategy["fusion_k"],
        final_k=strategy["final_k"],
        max_retries=strategy["max_retries"],
    )
    result["retrieval_strategy"] = strategy
    result["pipeline_mode"] = "light_agentic_corrective_rag"
    return result
