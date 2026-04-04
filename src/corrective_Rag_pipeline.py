# corrective_rag_v2.py

from typing import Dict, List, Tuple
import math

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ============================================================
# GLOBAL VERIFICATION MODELS
# Load once
# ============================================================
SIM_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"

similarity_model = None
nli_model = None


def get_similarity_model():
    global similarity_model
    if similarity_model is None:
        similarity_model = SentenceTransformer(SIM_MODEL_NAME)
    return similarity_model


def get_nli_model():
    global nli_model
    if nli_model is None:
        nli_model = pipeline(
            "text-classification",
            model=NLI_MODEL_NAME,
            top_k=None,
        )
    return nli_model


# ============================================================
# BASIC HELPERS
# ============================================================
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def cosine_similarity(a, b) -> float:
    # embeddings are normalized if requested by ST encode
    return float((a * b).sum())


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def compute_retrieval_confidence(reranked_docs: List[dict]) -> dict:
    if not reranked_docs:
        return {
            "confidence": 0.0,
            "top_score": 0.0,
            "avg_top3": 0.0,
            "needs_correction": True,
        }

    raw_scores = [float(item["score"]) for item in reranked_docs]
    probs = [sigmoid(s) for s in raw_scores]

    top_score = probs[0]
    avg_top3 = sum(probs[:3]) / min(3, len(probs))
    confidence = 0.6 * top_score + 0.4 * avg_top3

    return {
        "confidence": confidence,
        "top_score": top_score,
        "avg_top3": avg_top3,
        "needs_correction": confidence < 0.45,
    }


def expand_query(query: str) -> List[str]:
    expansions = [query]
    q = query.lower().strip()

    keyword_map = {
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "transformer": "attention transformer architecture",
        "few shot": "in context learning few shot prompting",
        "hallucination": "unsupported generation factual inconsistency",
        "retrieval": "dense retrieval sparse retrieval bm25 reranking",
        "reranking": "cross encoder reranker relevance scoring",
    }

    expanded = query
    for short_term, full_term in keyword_map.items():
        if short_term in q and full_term not in q:
            expanded += " " + full_term

    if expanded != query:
        expansions.append(expanded)

    return list(dict.fromkeys(expansions))


def dense_retrieve(vectorstore, query: str, k: int = 20) -> List[dict]:
    results = vectorstore.similarity_search_with_score(query, k=k)

    dense_docs = []
    for doc, score in results:
        dense_docs.append({
            "doc": doc,
            "score": float(score),
            "source": "dense",
        })
    return dense_docs


def sparse_retrieve(bm25, chunks, query: str, k: int = 20) -> List[dict]:
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    sparse_docs = []
    for idx in top_indices:
        sparse_docs.append({
            "doc": chunks[idx],
            "score": float(scores[idx]),
            "source": "sparse",
        })

    return sparse_docs


def rrf_fusion(dense_docs: List[dict], sparse_docs: List[dict], k: int = 60, top_k: int = 50) -> List[dict]:
    combined = {}

    def add_scores(docs: List[dict]):
        for rank, item in enumerate(docs):
            chunk_id = item["doc"].metadata["chunk_id"]
            score = 1.0 / (k + rank)

            if chunk_id not in combined:
                combined[chunk_id] = {
                    "doc": item["doc"],
                    "score": score,
                    "sources": [item["source"]],
                }
            else:
                combined[chunk_id]["score"] += score
                combined[chunk_id]["sources"].append(item["source"])

    add_scores(dense_docs)
    add_scores(sparse_docs)

    fused_docs = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    return fused_docs


def rerank_documents(reranker, query: str, fused_docs: List[dict], top_k: int = 5) -> List[dict]:
    if not fused_docs:
        return []

    pairs = [(query, item["doc"].page_content) for item in fused_docs]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(fused_docs, scores):
        reranked.append({
            "doc": item["doc"],
            "score": float(score),
            "sources": item.get("sources", []),
        })

    reranked = sorted(
        reranked,
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    return reranked


def corrective_retrieve(
    query: str,
    vectorstore,
    bm25,
    chunks,
    reranker,
    dense_k: int = 20,
    sparse_k: int = 20,
    fusion_k: int = 50,
    final_k: int = 5,
) -> Dict:
    dense_docs = dense_retrieve(vectorstore, query, k=dense_k)
    sparse_docs = sparse_retrieve(bm25, chunks, query, k=sparse_k)
    fused_docs = rrf_fusion(dense_docs, sparse_docs, top_k=fusion_k)
    reranked_docs = rerank_documents(reranker, query, fused_docs, top_k=final_k)

    initial_stats = compute_retrieval_confidence(reranked_docs)

    if not initial_stats["needs_correction"]:
        return {
            "final_docs": reranked_docs,
            "retrieval_stats": initial_stats,
            "corrected": False,
            "used_queries": [query],
        }

    expanded_queries = expand_query(query)
    combined_candidates = {}
    used_queries = []

    for q in expanded_queries:
        used_queries.append(q)

        d_docs = dense_retrieve(vectorstore, q, k=dense_k)
        s_docs = sparse_retrieve(bm25, chunks, q, k=sparse_k)
        f_docs = rrf_fusion(d_docs, s_docs, top_k=fusion_k)

        for item in f_docs:
            chunk_id = item["doc"].metadata["chunk_id"]

            if chunk_id not in combined_candidates:
                combined_candidates[chunk_id] = {
                    "doc": item["doc"],
                    "score": item["score"],
                    "sources": item.get("sources", []),
                }
            else:
                combined_candidates[chunk_id]["score"] += item["score"]
                combined_candidates[chunk_id]["sources"].extend(item.get("sources", []))

    merged_candidates = sorted(
        combined_candidates.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:fusion_k]

    corrected_reranked = rerank_documents(
        reranker=reranker,
        query=query,
        fused_docs=merged_candidates,
        top_k=final_k,
    )

    corrected_stats = compute_retrieval_confidence(corrected_reranked)

    if corrected_stats["confidence"] >= initial_stats["confidence"]:
        return {
            "final_docs": corrected_reranked,
            "retrieval_stats": corrected_stats,
            "corrected": True,
            "used_queries": used_queries,
        }

    return {
        "final_docs": reranked_docs,
        "retrieval_stats": initial_stats,
        "corrected": False,
        "used_queries": [query],
    }


# ============================================================
# CONTEXT + PROMPT
# ============================================================
def build_context(final_docs: List[dict], max_docs: int = 5) -> str:
    context_parts = []

    for rank, item in enumerate(final_docs[:max_docs], start=1):
        doc = item["doc"]
        meta = doc.metadata

        block = (
            f"[Evidence {rank}]\n"
            f"Title: {meta.get('title', 'Unknown')}\n"
            f"Authors: {meta.get('authors', ['Unknown'])}\n"
            f"File: {meta.get('file_name', 'Unknown')}\n"
            f"Page: {meta.get('page_number', 'Unknown')}\n"
            f"Chunk ID: {meta.get('chunk_id', 'Unknown')}\n"
            f"Content:\n{doc.page_content}\n"
        )
        context_parts.append(block)

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context: str) -> str:
    return f"""
You are a grounded research assistant.

Answer the question using ONLY the provided evidence.
Do not use outside knowledge.
If the evidence is insufficient, say exactly: Insufficient evidence.

Question:
{query}

Evidence:
{context}

Return:
1. A concise final answer
2. A short evidence-based justification
""".strip()


def _normalize_generation_output(generation_output):
    if isinstance(generation_output, dict):
        answer = generation_output.get("answer", "Insufficient evidence")
        justification = generation_output.get("justification", "")
        return answer.strip(), justification.strip()

    if isinstance(generation_output, str):
        text = generation_output.strip()
        if not text:
            return "Insufficient evidence", ""

        if text.lower() == "insufficient evidence":
            return "Insufficient evidence", ""

        marker = "Justification:"
        if marker in text:
            answer, justification = text.split(marker, 1)
            return answer.strip(), justification.strip()

        return text, ""

    return "Insufficient evidence", ""


# ============================================================
# VERIFICATION PARAMS
# ============================================================
def evidence_similarity_scoring(answer: str, final_docs: List[dict]) -> dict:
    """
    Semantic similarity between answer and evidence chunks.
    """
    if not answer or not final_docs:
        return {
            "max_similarity": 0.0,
            "avg_top3_similarity": 0.0,
            "per_chunk_similarities": [],
        }

    texts = [answer] + [item["doc"].page_content[:1200] for item in final_docs]
    embeddings = get_similarity_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    answer_emb = embeddings[0]
    evidence_embs = embeddings[1:]

    similarities = []
    for emb in evidence_embs:
        sim = float((answer_emb * emb).sum())
        similarities.append(sim)

    similarities_sorted = sorted(similarities, reverse=True)
    avg_top3 = sum(similarities_sorted[:3]) / min(3, len(similarities_sorted))

    return {
        "max_similarity": similarities_sorted[0] if similarities_sorted else 0.0,
        "avg_top3_similarity": avg_top3 if similarities_sorted else 0.0,
        "per_chunk_similarities": similarities,
    }


def nli_entailment_check(answer: str, final_docs: List[dict]) -> dict:
    """
    Checks whether evidence entails the generated answer.
    """
    if not answer or answer.lower().strip() == "insufficient evidence" or not final_docs:
        return {
            "max_entailment": 0.0,
            "avg_top3_entailment": 0.0,
            "contradiction_max": 0.0,
            "per_chunk_nli": [],
        }

    current_nli_model = get_nli_model()

    per_chunk = []

    for item in final_docs:
        evidence = item["doc"].page_content[:1200]
        combined_input = f"{evidence} </s> {answer}"

        outputs = current_nli_model(combined_input)[0]
        label_scores = {entry["label"].upper(): float(entry["score"]) for entry in outputs}

        entailment = label_scores.get("ENTAILMENT", 0.0)
        contradiction = label_scores.get("CONTRADICTION", 0.0)

        per_chunk.append({
            "entailment": entailment,
            "contradiction": contradiction,
        })

    entailments = sorted([x["entailment"] for x in per_chunk], reverse=True)
    contradictions = [x["contradiction"] for x in per_chunk]

    avg_top3_ent = sum(entailments[:3]) / min(3, len(entailments))

    return {
        "max_entailment": entailments[0] if entailments else 0.0,
        "avg_top3_entailment": avg_top3_ent if entailments else 0.0,
        "contradiction_max": max(contradictions) if contradictions else 0.0,
        "per_chunk_nli": per_chunk,
    }


def evidence_coverage(answer: str, final_docs: List[dict]) -> float:
    """
    Simple lexical support score.
    """
    if not answer or not final_docs:
        return 0.0

    answer_words = set(answer.lower().split())
    context_words = set()

    for item in final_docs:
        context_words.update(item["doc"].page_content.lower().split())

    if not answer_words:
        return 0.0

    return len(answer_words & context_words) / len(answer_words)


def compute_verification_params(answer: str, final_docs: List[dict], retrieval_stats: dict) -> dict:
    """
    Four important parameters:
    1. retrieval_confidence
    2. reranker_confidence
    3. evidence_similarity
    4. nli_entailment
    Also includes coverage + contradiction as extra signals.
    """
    reranker_conf = retrieval_stats.get("top_score", 0.0)

    sim_stats = evidence_similarity_scoring(answer, final_docs)
    nli_stats = nli_entailment_check(answer, final_docs)
    coverage = evidence_coverage(answer, final_docs)

    retrieval_conf = retrieval_stats.get("confidence", 0.0)
    evidence_similarity = 0.6 * sim_stats["max_similarity"] + 0.4 * sim_stats["avg_top3_similarity"]
    grounding_score = 0.6 * nli_stats["max_entailment"] + 0.4 * nli_stats["avg_top3_entailment"]

    # combined confidence
    combined_confidence = (
        0.25 * retrieval_conf +
        0.20 * reranker_conf +
        0.20 * coverage +
        0.15 * evidence_similarity +
        0.20 * grounding_score
    )

    verified = (
        retrieval_conf >= 0.35 and
        reranker_conf >= 0.35 and
        evidence_similarity >= 0.30 and
        grounding_score >= 0.45 and
        nli_stats["contradiction_max"] < 0.40
    )

    return {
        "retrieval_confidence": retrieval_conf,
        "reranker_confidence": reranker_conf,
        "evidence_coverage": coverage,
        "evidence_similarity": evidence_similarity,
        "grounding_score": grounding_score,
        "nli_max_entailment": nli_stats["max_entailment"],
        "nli_avg_top3_entailment": nli_stats["avg_top3_entailment"],
        "nli_contradiction_max": nli_stats["contradiction_max"],
        "combined_confidence": combined_confidence,
        "verified": verified,
        "similarity_debug": sim_stats,
        "nli_debug": nli_stats,
    }


# ============================================================
# ANSWER CORRECTION LOOP
# ============================================================
def corrective_generation_loop(
    query: str,
    vectorstore,
    bm25,
    chunks,
    reranker,
    llm_generate_fn,
    max_retries: int = 2,
    dense_k: int = 20,
    sparse_k: int = 20,
    fusion_k: int = 50,
    final_k: int = 5,
) -> Dict:
    retrieval_output = corrective_retrieve(
        query=query,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        dense_k=dense_k,
        sparse_k=sparse_k,
        fusion_k=fusion_k,
        final_k=final_k,
    )

    final_docs = retrieval_output["final_docs"]
    retrieval_stats = retrieval_output["retrieval_stats"]
    corrected = retrieval_output["corrected"]
    used_queries = list(retrieval_output["used_queries"])

    if retrieval_stats["confidence"] < 0.20:
        return {
            "answer": "Insufficient evidence",
            "justification": "Retrieved evidence was too weak before generation.",
            "corrected": corrected,
            "used_queries": used_queries,
            "retrieval_confidence": retrieval_stats["confidence"],
            "verification": {
                "verified": False,
                "reason": "retrieval_too_weak_pre_generation",
            },
            "verification_params": {},
            "final_docs": final_docs,
            "context": "",
            "abstained": True,
            "retries_used": 0,
        }

    context = build_context(final_docs, max_docs=final_k)
    prompt = build_grounded_prompt(query, context)

    generation_output = llm_generate_fn(prompt)
    answer, justification = _normalize_generation_output(generation_output)

    verification_params = compute_verification_params(
        answer=answer,
        final_docs=final_docs,
        retrieval_stats=retrieval_stats,
    )

    retries_used = 0

    while (not verification_params["verified"]) and retries_used < max_retries:
        retries_used += 1

        # stronger correction: expand query + re-retrieve
        retry_queries = expand_query(query)
        if len(retry_queries) > 0:
            retry_query = retry_queries[min(retries_used - 1, len(retry_queries) - 1)]
            if retry_query not in used_queries:
                used_queries.append(retry_query)
        else:
            retry_query = query

        retry_output = corrective_retrieve(
            query=retry_query,
            vectorstore=vectorstore,
            bm25=bm25,
            chunks=chunks,
            reranker=reranker,
            dense_k=dense_k,
            sparse_k=sparse_k,
            fusion_k=fusion_k,
            final_k=final_k,
        )

        if retry_output["retrieval_stats"]["confidence"] >= retrieval_stats["confidence"]:
            final_docs = retry_output["final_docs"]
            retrieval_stats = retry_output["retrieval_stats"]
            corrected = True

        context = build_context(final_docs, max_docs=final_k)
        prompt = build_grounded_prompt(query, context)
        generation_output = llm_generate_fn(prompt)
        answer, justification = _normalize_generation_output(generation_output)

        verification_params = compute_verification_params(
            answer=answer,
            final_docs=final_docs,
            retrieval_stats=retrieval_stats,
        )

    abstained = False
    if (
        not verification_params["verified"] or
        verification_params["combined_confidence"] < 0.40 or
        verification_params["grounding_score"] < 0.45
    ):
        answer = "Insufficient evidence"
        justification = "The answer could not be sufficiently verified against the retrieved evidence."
        abstained = True

    return {
        "answer": answer,
        "justification": justification,
        "corrected": corrected,
        "used_queries": used_queries,
        "retrieval_confidence": retrieval_stats["confidence"],
        "verification": {
            "verified": verification_params["verified"],
            "reason": "verification_gate_passed" if verification_params["verified"] else "verification_gate_failed",
        },
        "verification_params": verification_params,
        "final_docs": final_docs,
        "context": context,
        "abstained": abstained,
        "retries_used": retries_used,
    }


def corrective_rag_pipeline_v2(
    query: str,
    vectorstore,
    bm25,
    chunks,
    reranker,
    llm_generate_fn,
    dense_k: int = 20,
    sparse_k: int = 20,
    fusion_k: int = 50,
    final_k: int = 5,
    max_retries: int = 2,
) -> Dict:
    return corrective_generation_loop(
        query=query,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        llm_generate_fn=llm_generate_fn,
        max_retries=max_retries,
        dense_k=dense_k,
        sparse_k=sparse_k,
        fusion_k=fusion_k,
        final_k=final_k,
    )


# ============================================================
# ECE EVALUATION
# ============================================================
def expected_calibration_error(
    confidences: List[float],
    correctness: List[int],
    n_bins: int = 10,
) -> Dict:
    """
    ECE over predicted confidence vs actual correctness.
    correctness: 1 if correct, 0 if incorrect
    """
    assert len(confidences) == len(correctness), "confidences and correctness must have same length"
    if len(confidences) == 0:
        return {
            "ece": 0.0,
            "bin_details": [],
        }

    bin_details = []
    ece = 0.0
    total = len(confidences)

    for b in range(n_bins):
        lower = b / n_bins
        upper = (b + 1) / n_bins

        indices = [
            i for i, c in enumerate(confidences)
            if (lower <= c < upper) or (b == n_bins - 1 and lower <= c <= upper)
        ]

        if not indices:
            continue

        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        bin_acc = sum(correctness[i] for i in indices) / len(indices)
        bin_weight = len(indices) / total

        ece += abs(bin_acc - bin_conf) * bin_weight

        bin_details.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "count": len(indices),
            "avg_confidence": bin_conf,
            "accuracy": bin_acc,
            "gap": abs(bin_acc - bin_conf),
        })

    return {
        "ece": ece,
        "bin_details": bin_details,
    }


# ============================================================
# DUMMY LLM
# ============================================================
def llm_generate_fn(prompt: str):
    return {
        "answer": "Dummy grounded answer",
        "justification": "Replace llm_generate_fn with your actual GPT-4o-mini call."
    }


if __name__ == "__main__":
    print("Import and call corrective_rag_pipeline_v2(...) from main.py")
