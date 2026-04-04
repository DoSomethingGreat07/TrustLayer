import os
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from loader import load_documents
from chunking import chunk_documents
from retrieval import build_vector_store, build_bm25, hybrid_retrieve


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data")
OUTPUT_CSV = "artifacts/generated_questions.csv"

MODEL_NAME = "gpt-4o-mini"

CATEGORY_COUNTS = {
    "factual": 4,
    "process": 3,
    "comparison": 2,
    "why_purpose": 2,
    "limitation": 2,
    "multihop": 2,
    "unanswerable": 5,
}

RETRIEVABILITY_THRESHOLD = 0.45
CHUNKS_PER_PAPER = 6
CHUNK_CONTEXT_CHARS = 10000


def get_llm(model_name=MODEL_NAME):
    return ChatOpenAI(model=model_name, temperature=0.7)


def group_chunks_by_doc_id(chunks):
    grouped = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id", "unknown_doc")
        grouped[doc_id].append(chunk)
    return grouped


def sample_chunks_for_paper(chunks, n=CHUNKS_PER_PAPER):
    if len(chunks) <= n:
        return chunks

    indices = [round(i * (len(chunks) - 1) / (n - 1)) for i in range(n)]
    return [chunks[i] for i in indices]


def build_chunk_context(chunks, max_chars=CHUNK_CONTEXT_CHARS):
    parts = []
    total = 0

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id", "?")
        page = chunk.metadata.get("page_id", "?")
        text = chunk.page_content.strip()

        block = f"[chunk_id: {chunk_id} | page: {page}]\n{text}\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 300:
                parts.append(block[:remaining] + "\n[TRUNCATED]")
            break

        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)


def build_question_prompt(doc_id, chunk_context, category_counts):
    return f"""
You are creating an evaluation dataset for a Trust-Aware RAG system over research papers.

Your task is to generate questions that can be answered using the EXACT chunk text provided.
This is critical — the retrieval system will need to surface these same chunks to answer
the questions. Do NOT generate questions that require information outside the chunks below.

Research Paper ID: {doc_id}

Generate questions in the following categories and counts:
- factual:      {category_counts["factual"]}
- process:      {category_counts["process"]}
- comparison:   {category_counts["comparison"]}
- why_purpose:  {category_counts["why_purpose"]}
- limitation:   {category_counts["limitation"]}
- multihop:     {category_counts["multihop"]}
- unanswerable: {category_counts["unanswerable"]}

Rules:
1. Answerable questions MUST be directly supported by the chunk text below.
2. The answer must be findable in one or two chunks — not require reading the whole paper.
3. Factual: ask for explicit facts, numbers, names, or definitions from the chunks.
4. Process: ask how something works based on what is described in the chunks.
5. Comparison: compare two things that are both mentioned in the chunks.
6. Why_purpose: ask about motivations or goals explicitly stated in the chunks.
7. Limitation: ask about weaknesses or constraints explicitly mentioned in the chunks.
8. Multihop: require combining information from exactly TWO chunks.
9. Unanswerable: sound plausible and related to the paper topic, but the answer must NOT
   appear anywhere in the chunks provided.
10. Do not duplicate questions or ask the same thing in different wording.
11. Output ONLY valid JSON — no preamble, no markdown fences.

Required JSON format:
[
  {{
    "question": "...",
    "category": "factual|process|comparison|why_purpose|limitation|multihop|unanswerable",
    "answerable": 1,
    "source_chunk_ids": ["chunk_id_1"]
  }},
  {{
    "question": "...",
    "category": "multihop",
    "answerable": 1,
    "source_chunk_ids": ["chunk_id_1", "chunk_id_2"]
  }},
  {{
    "question": "...",
    "category": "unanswerable",
    "answerable": 0,
    "source_chunk_ids": []
  }}
]

Research Paper Chunks:
{chunk_context}
""".strip()


def clean_json_response(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from model response.")


def generate_questions_for_paper(llm, doc_id, chunks, category_counts):
    sampled = sample_chunks_for_paper(chunks, n=CHUNKS_PER_PAPER)
    chunk_context = build_chunk_context(sampled)
    prompt = build_question_prompt(doc_id, chunk_context, category_counts)

    response = llm.invoke(prompt)
    items = clean_json_response(response.content)

    cleaned = []
    for item in items:
        question = str(item.get("question", "")).strip()
        category = str(item.get("category", "")).strip()
        answerable = int(item.get("answerable", 1))
        source_chunk_ids = item.get("source_chunk_ids", [])

        if not question:
            continue

        cleaned.append({
            "doc_id": doc_id,
            "question": question,
            "category": category,
            "answerable": answerable,
            "source_chunk_ids": "|".join(str(c) for c in source_chunk_ids),
        })

    return cleaned


def validate_retrievability(
    rows,
    vector_store,
    bm25,
    filtered_chunks,
    threshold=RETRIEVABILITY_THRESHOLD,
):
    print("\nValidating retrievability of answerable questions...")

    for row in rows:
        if row["answerable"] == 0:
            row["retrieval_validated"] = True
            row["retrieval_max_score"] = None
            continue

        try:
            result = hybrid_retrieve(
                query=row["question"],
                vector_store=vector_store,
                bm25=bm25,
                filtered_chunks=filtered_chunks,
                alpha=0.7,
                dense_k=12,
                sparse_k=12,
                fusion_k=10,
                final_k=5,
                use_reranker=False,
            )

            fused_scores = [score for _, score in result["fused_results"]]
            max_score = max(fused_scores) if fused_scores else 0.0

            row["retrieval_validated"] = max_score >= threshold
            row["retrieval_max_score"] = round(float(max_score), 4)

        except Exception as e:
            print(f"  Validation error for '{row['question'][:60]}': {e}")
            row["retrieval_validated"] = False
            row["retrieval_max_score"] = 0.0

    failed = [r for r in rows if r["answerable"] == 1 and not r["retrieval_validated"]]
    total_answerable = len([r for r in rows if r["answerable"] == 1])

    print(f"  Answerable questions failing retrieval threshold: {len(failed)}/{total_answerable}")
    if failed:
        print("  These questions will behave like unanswerable questions at inference time.")

    return rows


def normalize_question(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_questions(rows):
    seen = set()
    deduped = []

    for row in rows:
        key = (row["doc_id"], row["category"], normalize_question(row["question"]))
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def save_questions_to_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        "doc_id", "question", "category", "answerable",
        "source_chunk_ids", "retrieval_validated", "retrieval_max_score",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nQUESTION GENERATION SUMMARY")
    print("-" * 50)
    print(f"Total questions: {len(rows)}")

    by_doc = defaultdict(int)
    by_cat = defaultdict(int)
    by_answerable = defaultdict(int)

    failed_retrieval = 0
    for row in rows:
        by_doc[row["doc_id"]] += 1
        by_cat[row["category"]] += 1
        by_answerable["Answerable" if row["answerable"] == 1 else "Unanswerable"] += 1
        if row["answerable"] == 1 and not row.get("retrieval_validated", True):
            failed_retrieval += 1

    print("\nBy category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")

    print("\nAnswerable vs Unanswerable:")
    for label, count in sorted(by_answerable.items()):
        print(f"  {label}: {count} ({count/len(rows)*100:.1f}%)")

    if failed_retrieval > 0:
        print(f"\n⚠️  Answerable questions failing retrieval: {failed_retrieval}")
        print("   Consider re-generating or lowering RETRIEVABILITY_THRESHOLD")

    print(f"\nBy document:")
    for doc_id, count in sorted(by_doc.items()):
        print(f"  {doc_id}: {count}")


def main():
    print("Loading documents...")
    documents = load_documents(DATA_PATH)

    print("Chunking documents...")
    chunks = chunk_documents(
        documents,
        chunk_size=1800,
        chunk_overlap=350,
        min_chunk_length=200,
    )

    print("Building vector store...")
    try:
        vector_store, filtered_chunks = build_vector_store(
            chunks=chunks,
            persist_dir="artifacts/chroma_papers",
            collection_name="research_papers",
            model_name="sentence-transformers/all-mpnet-base-v2",
            min_text_length=200,
        )
        bm25 = build_bm25(filtered_chunks)
        validate = True
    except Exception as e:
        print(f"  ⚠️ Could not build vector store: {e}. Skipping retrievability validation.")
        vector_store = bm25 = None
        filtered_chunks = [c for c in chunks if len(c.page_content.strip()) >= 200]
        validate = False

    print("Grouping chunks by paper...")
    grouped = group_chunks_by_doc_id(filtered_chunks)

    llm = get_llm()
    all_rows = []

    for i, (doc_id, doc_chunks) in enumerate(sorted(grouped.items()), start=1):
        print(f"\n[{i}/{len(grouped)}] Generating questions for: {doc_id} ({len(doc_chunks)} chunks)")

        try:
            rows = generate_questions_for_paper(
                llm=llm,
                doc_id=doc_id,
                chunks=doc_chunks,
                category_counts=CATEGORY_COUNTS,
            )
            all_rows.extend(rows)
            print(f"  Generated: {len(rows)} questions")

        except Exception as e:
            print(f"  Failed for {doc_id}: {e}")

    all_rows = deduplicate_questions(all_rows)

    if validate and vector_store and bm25:
        all_rows = validate_retrievability(
            all_rows,
            vector_store=vector_store,
            bm25=bm25,
            filtered_chunks=filtered_chunks,
        )

        all_rows = [
            row for row in all_rows
            if row["answerable"] == 0 or row.get("retrieval_validated", False)
        ]
    else:
        for row in all_rows:
            row["retrieval_validated"] = None
            row["retrieval_max_score"] = None

    save_questions_to_csv(all_rows, OUTPUT_CSV)
    print_summary(all_rows)
    print(f"\nSaved questions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
