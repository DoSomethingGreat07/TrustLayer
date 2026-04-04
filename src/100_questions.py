import os
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from loader import load_documents


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data")
OUTPUT_CSV = "artifacts/generated_questions_100.csv"

MODEL_NAME = "gpt-4o-mini"
MAX_PAPERS = 5   # 5 papers × 20 questions each = ~100 questions

# Per-paper balanced distribution
CATEGORY_COUNTS = {
    "factual": 5,
    "process": 4,
    "comparison": 3,
    "why_purpose": 3,
    "limitation": 2,
    "multihop": 2,
    "unanswerable": 1,
}


def get_llm(model_name=MODEL_NAME):
    return ChatOpenAI(model=model_name, temperature=0.7)


def group_documents_by_doc_id(documents):
    grouped = defaultdict(list)

    for doc in documents:
        doc_id = doc.metadata.get("doc_id", "unknown_doc")
        grouped[doc_id].append(doc)

    return grouped


def build_paper_context(docs, max_chars=12000):
    """
    Combine chunks/pages for one paper into a single context string.
    Keeps order by page_id if available.
    """
    def page_sort_key(d):
        page_id = d.metadata.get("page_id", 0)
        try:
            return int(page_id)
        except Exception:
            return 0

    docs = sorted(docs, key=page_sort_key)

    parts = []
    total = 0

    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue

        block = f"[page: {d.metadata.get('page_id', '?')}]\n{text}\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 300:
                parts.append(block[:remaining] + "\n[TRUNCATED]")
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def build_question_prompt(doc_id, context, category_counts):
    return f"""
You are creating an evaluation dataset for a Trust-Aware RAG system over research papers.

Your task is to generate diversified questions from the research paper context below.

Research Paper ID:
{doc_id}

Generate questions in the following categories and counts:
- factual: {category_counts["factual"]}
- process: {category_counts["process"]}
- comparison: {category_counts["comparison"]}
- why_purpose: {category_counts["why_purpose"]}
- limitation: {category_counts["limitation"]}
- multihop: {category_counts["multihop"]}
- unanswerable: {category_counts["unanswerable"]}

Rules:
1. Questions must cover important concepts from the paper.
2. Questions must be specific, not generic.
3. Questions must be diverse and non-duplicate.
4. Factual questions should ask for explicit information.
5. Process questions should ask how something works.
6. Comparison questions should compare methods, concepts, or stages.
7. Why_purpose questions should ask about motivations or goals.
8. Limitation questions should ask about weaknesses, caveats, or constraints.
9. Multihop questions should require combining multiple pieces of information from the paper.
10. Unanswerable questions should sound plausible, but the answer must NOT be directly supported by the provided context.
11. Do not ask the same thing in different wording.
12. Output ONLY valid JSON.

Required JSON format:
[
  {{
    "question": "...",
    "category": "factual|process|comparison|why_purpose|limitation|multihop|unanswerable",
    "answerable": 1
  }},
  ...
]

For unanswerable questions, set "answerable": 0.
For all others, set "answerable": 1.

Research Paper Context:
{context}
""".strip()


def clean_json_response(text):
    """
    Extract JSON array from model response.
    """
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


def generate_questions_for_paper(llm, doc_id, docs, category_counts):
    context = build_paper_context(docs)
    prompt = build_question_prompt(doc_id, context, category_counts)

    response = llm.invoke(prompt)
    items = clean_json_response(response.content)

    cleaned = []
    for item in items:
        question = str(item.get("question", "")).strip()
        category = str(item.get("category", "")).strip()
        answerable = int(item.get("answerable", 1))

        if not question:
            continue

        cleaned.append({
            "doc_id": doc_id,
            "question": question,
            "category": category,
            "answerable": answerable,
        })

    return cleaned


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

    fieldnames = ["doc_id", "question", "category", "answerable"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nQUESTION GENERATION SUMMARY")
    print("-" * 50)
    print(f"Total questions: {len(rows)}")

    by_doc = defaultdict(int)
    by_cat = defaultdict(int)

    for row in rows:
        by_doc[row["doc_id"]] += 1
        by_cat[row["category"]] += 1

    print("\nBy document:")
    for doc_id, count in sorted(by_doc.items()):
        print(f"{doc_id}: {count}")

    print("\nBy category:")
    for cat, count in sorted(by_cat.items()):
        print(f"{cat}: {count}")


def main():
    print("Loading documents...")
    documents = load_documents(DATA_PATH)

    print("Grouping documents by paper...")
    grouped = group_documents_by_doc_id(documents)

    # Limit to first MAX_PAPERS papers for testing
    grouped_items = sorted(grouped.items())[:MAX_PAPERS]

    llm = get_llm()

    all_rows = []

    for i, (doc_id, docs) in enumerate(grouped_items, start=1):
        print(f"\n[{i}/{len(grouped_items)}] Generating questions for: {doc_id}")

        try:
            rows = generate_questions_for_paper(
                llm=llm,
                doc_id=doc_id,
                docs=docs,
                category_counts=CATEGORY_COUNTS,
            )
            all_rows.extend(rows)
            print(f"Generated: {len(rows)} questions")

        except Exception as e:
            print(f"Failed for {doc_id}: {e}")

    all_rows = deduplicate_questions(all_rows)

    save_questions_to_csv(all_rows, OUTPUT_CSV)
    print_summary(all_rows)

    print(f"\nSaved questions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
