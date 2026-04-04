import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_openai import ChatOpenAI


# ==========================================================
# GLOBAL MODELS
# ==========================================================

SIM_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
NLI_MODEL.eval()

ID2LABEL = {int(k): v.lower() for k, v in NLI_MODEL.config.id2label.items()}


# ==========================================================
# HELPERS
# ==========================================================

def split_into_sentences(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _nli_scores(premise, hypothesis, max_length=512):
    inputs = NLI_TOKENIZER(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = NLI_MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    score_map = {}
    for i, p in enumerate(probs):
        label = ID2LABEL[i]
        score_map[label] = float(p)

    return {
        "entailment": score_map.get("entailment", 0.0),
        "contradiction": score_map.get("contradiction", 0.0),
        "neutral": score_map.get("neutral", 0.0),
    }


# ==========================================================
# 1. EVIDENCE SIMILARITY
# ==========================================================

def compute_evidence_similarity(answer, top_results):
    answer_emb = SIM_MODEL.encode([answer], normalize_embeddings=True)

    chunk_texts = [doc.page_content for doc, _ in top_results]
    chunk_embs = SIM_MODEL.encode(chunk_texts, normalize_embeddings=True)

    sims = cosine_similarity(answer_emb, chunk_embs)[0]

    return {
        "chunk_similarities": sims.tolist(),
        "max_similarity": float(np.max(sims)),
        "mean_similarity": float(np.mean(sims)),
    }


# ==========================================================
# 2. NLI ENTAILMENT (UPDATED: sentence-level)
# ==========================================================

def compute_nli_entailment(answer, top_results):
    entailment_scores = []
    contradiction_scores = []
    neutral_scores = []

    best_supporting_sentences = []

    for doc, _ in top_results:
        sentences = split_into_sentences(doc.page_content)

        local_scores = []
        for sent in sentences:
            score_map = _nli_scores(sent, answer)
            local_scores.append((sent, score_map))

        if local_scores:
            best_entail = max(local_scores, key=lambda x: x[1]["entailment"])
            best_contra = max(local_scores, key=lambda x: x[1]["contradiction"])
            best_neutral = max(local_scores, key=lambda x: x[1]["neutral"])

            entailment_scores.append(best_entail[1]["entailment"])
            contradiction_scores.append(best_contra[1]["contradiction"])
            neutral_scores.append(best_neutral[1]["neutral"])

            best_supporting_sentences.append({
                "sentence": best_entail[0],
                "entailment": best_entail[1]["entailment"],
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
            })
        else:
            entailment_scores.append(0.0)
            contradiction_scores.append(0.0)
            neutral_scores.append(0.0)

            best_supporting_sentences.append({
                "sentence": "",
                "entailment": 0.0,
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
            })

    return {
        "entailment_scores": entailment_scores,
        "contradiction_scores": contradiction_scores,
        "neutral_scores": neutral_scores,
        "max_entailment": float(np.max(entailment_scores)),
        "mean_entailment": float(np.mean(entailment_scores)),
        "max_contradiction": float(np.max(contradiction_scores)),
        "mean_contradiction": float(np.mean(contradiction_scores)),
        "best_supporting_sentences": best_supporting_sentences,
    }


# ==========================================================
# 3. SELF-CONSISTENCY
# ==========================================================

def generate_multiple_answers(query, context, n=3, model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0.3)

    prompt = f"""
You are answering a research question using only the provided context.

Rules:
1. Use only the context below.
2. If unsupported, say "Insufficient evidence in retrieved context."
3. Be concise and factual.

Question:
{query}

Context:
{context}

Answer:
""".strip()

    answers = []
    for _ in range(n):
        resp = llm.invoke(prompt)
        answers.append(resp.content.strip())

    return answers


def compute_self_consistency(answers):
    if len(answers) < 2:
        return {
            "answers": answers,
            "pairwise_similarities": [],
            "mean_consistency": 1.0,
        }

    embs = SIM_MODEL.encode(answers, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embs)

    pairwise = []
    n = len(answers)
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(float(sim_matrix[i, j]))

    return {
        "answers": answers,
        "pairwise_similarities": pairwise,
        "mean_consistency": float(np.mean(pairwise)),
    }


# ==========================================================
# 4. COMBINE VERIFICATION SCORES
# ==========================================================

def combine_verification_scores(similarity_result, nli_result, consistency_result):
    sim_score = similarity_result["max_similarity"]
    entail_score = nli_result["max_entailment"]
    contradiction_penalty = nli_result["max_contradiction"]
    consistency_score = consistency_result["mean_consistency"]

    verification_score = (
    0.15 * sim_score
    + 0.55 * entail_score
    + 0.25 * consistency_score
    - 0.15 * contradiction_penalty
)
   

    return {
        "similarity_score": float(sim_score),
        "entailment_score": float(entail_score),
        "consistency_score": float(consistency_score),
        "contradiction_penalty": float(contradiction_penalty),
        "verification_score": float(verification_score),
    }


# ==========================================================
# 5. FULL VERIFICATION PIPELINE
# ==========================================================

def verify_answer(query, answer, context, top_results):
    similarity_result = compute_evidence_similarity(answer, top_results)

    nli_result = compute_nli_entailment(answer, top_results)

    answers = generate_multiple_answers(query, context, n=3)

    consistency_result = compute_self_consistency(answers)

    final_score = combine_verification_scores(
        similarity_result,
        nli_result,
        consistency_result,
    )
    if "insufficient evidence in retrieved context" in answer.lower():
     final_score["verification_score"] = min(final_score["verification_score"], 0.25)

    return {
        "similarity": similarity_result,
        "nli": nli_result,
        "self_consistency": consistency_result,
        "final_verification": final_score,
    }