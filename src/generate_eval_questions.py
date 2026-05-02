import argparse
import csv
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


DEFAULT_CHUNKS_PATH = Path("artifacts/chunks.pkl")
DEFAULT_OUTPUT_CSV = Path("artifacts/eval_questions_150.csv")

CATEGORIES = {
    "factual": "explicit facts, definitions, model names, datasets, or numeric details",
    "conceptual": "core ideas, mechanisms, or design choices explained by the text",
    "methodology": "how the method, experiment, training, or evaluation works",
    "comparison": "differences or relationships between two items both present in the text",
    "limitation": "constraints, weaknesses, caveats, or failure cases stated in the text",
    "multihop": "requires combining evidence from exactly two provided chunks",
    "unanswerable": "plausible but not answered by the provided chunks",
}

COMMON_ENGLISH_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "model", "models", "method", "results", "training", "data", "task", "using",
    "learning", "network", "paper", "approach", "performance", "evaluation",
    "experiments", "proposed", "based", "between", "shown", "used", "can",
}

GENERIC_QUESTION_PATTERNS = (
    "discussed in the paper",
    "mentioned in the context",
    "mentioned in the paper",
    "described in the paper",
    "according to the paper",
    "in the paper",
    "this paper",
    "the research paper",
    "this work",
    "the work",
    "this research",
    "the research",
    "the study",
    "this study",
    "the context",
    "this context",
    "the method",
    "this method",
    "the model",
    "this model",
    "the approach",
    "this approach",
    "the authors",
    "the experiments",
    "the evaluation",
    "what is the model name",
    "what dataset is mentioned",
    "what is the title",
    "who are the authors",
    "main contributions of this work",
)

ANCHOR_STOPWORDS = {
    "paper", "research", "study", "method", "model", "models", "approach",
    "using", "with", "from", "this", "that", "their", "there", "which",
    "what", "does", "about", "into", "over", "under", "between", "towards",
    "toward", "based", "learning", "neural", "deep", "networks", "network",
    "systems", "system", "efficient", "effective", "large", "language",
}


def load_chunks(path: Path) -> list:
    with path.open("rb") as handle:
        return pickle.load(handle)


def group_chunks_by_doc(chunks: list) -> dict[str, list]:
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.metadata.get("doc_id", "unknown")].append(chunk)
    return dict(grouped)


def representative_papers(grouped: dict[str, list], target_papers: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for doc_id, chunks in grouped.items():
        domain = chunks[0].metadata.get("domain", "Unknown") if chunks else "Unknown"
        by_domain[domain].append(doc_id)

    selected = []
    domains = sorted(by_domain)
    while len(selected) < target_papers and domains:
        for domain in domains:
            remaining = [doc_id for doc_id in by_domain[domain] if doc_id not in selected]
            if remaining and len(selected) < target_papers:
                selected.append(rng.choice(remaining))
        domains = [domain for domain in domains if any(doc_id not in selected for doc_id in by_domain[domain])]

    return selected


def sample_chunks(chunks: list, count: int, seed: int) -> list:
    readable_chunks = [chunk for chunk in chunks if is_readable_chunk(chunk.page_content)]
    if len(readable_chunks) >= max(3, count // 2):
        chunks = readable_chunks

    if len(chunks) <= count:
        return chunks

    rng = random.Random(seed)
    sorted_chunks = sorted(chunks, key=lambda item: item.metadata.get("chunk_index", 0))
    anchors = [
        sorted_chunks[round(i * (len(sorted_chunks) - 1) / max(1, count - 1))]
        for i in range(max(1, count // 2))
    ]
    remaining = [chunk for chunk in sorted_chunks if chunk not in anchors]
    extra = rng.sample(remaining, min(count - len(anchors), len(remaining)))
    return sorted(anchors + extra, key=lambda item: item.metadata.get("chunk_index", 0))


def is_readable_chunk(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if len(words) < 40:
        return False

    common_hits = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
    common_ratio = common_hits / len(words)
    odd_chars = sum(1 for char in text if not (char.isalnum() or char.isspace() or char in ".,;:!?%-()/[]'\""))
    odd_ratio = odd_chars / max(1, len(text))

    return common_ratio >= 0.035 and odd_ratio <= 0.08


def build_context(chunks: list, max_chars: int) -> str:
    parts = []
    total = 0
    for chunk in chunks:
        meta = chunk.metadata
        text = re.sub(r"\s+", " ", chunk.page_content).strip()
        block = (
            f"[chunk_id: {meta.get('chunk_id')} | page: {meta.get('page_number')} | "
            f"title: {meta.get('title')}]\n{text}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)


def filename_to_clean_title(filename: str) -> str:
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"^\d+[_\-\s]*", "", name)
    name = re.sub(r"\s+-\s+\d{4}\.\d+v\d+$", "", name)
    name = name.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


def looks_like_clean_title(title: str) -> bool:
    if not title:
        return False
    words = re.findall(r"[A-Za-z]{3,}", title)
    if len(words) < 3:
        return False
    odd_chars = sum(1 for char in title if not (char.isalnum() or char.isspace() or char in ".,;:!?%-()/[]'\""))
    letters = re.findall(r"[A-Za-z]", title)
    vowels = re.findall(r"[AEIOUaeiou]", title)
    vowel_ratio = len(vowels) / max(1, len(letters))
    common_hits = sum(1 for word in words if word.lower() in COMMON_ENGLISH_WORDS)
    common_ratio = common_hits / len(words)
    return (
        odd_chars / max(1, len(title)) <= 0.04
        and vowel_ratio >= 0.28
        and common_ratio >= 0.08
    )


def paper_label(chunks: list) -> str:
    meta = chunks[0].metadata if chunks else {}
    title = str(meta.get("title", "")).strip()
    if looks_like_clean_title(title):
        return title
    return filename_to_clean_title(str(meta.get("file_name", meta.get("doc_id", "research paper"))))


def anchor_terms(label: str, doc_id: str) -> set[str]:
    text = f"{label} {doc_id}".replace("_", " ").replace("-", " ")
    terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", text)
        if word.lower() not in ANCHOR_STOPWORDS
    }
    return terms


def has_usable_label(label: str, doc_id: str) -> bool:
    anchors = anchor_terms(label, doc_id)
    alpha_chars = re.findall(r"[A-Za-z]", label)
    return len(anchors) >= 2 and len(alpha_chars) >= 12


def questions_per_paper(total_questions: int, paper_count: int) -> list[int]:
    base = total_questions // paper_count
    remainder = total_questions % paper_count
    return [base + (1 if i < remainder else 0) for i in range(paper_count)]


def category_plan(count: int) -> dict[str, int]:
    weights = [
        ("factual", 0.20),
        ("conceptual", 0.18),
        ("methodology", 0.16),
        ("comparison", 0.12),
        ("limitation", 0.12),
        ("multihop", 0.12),
        ("unanswerable", 0.10),
    ]
    plan = {category: int(count * weight) for category, weight in weights}
    while sum(plan.values()) < count:
        category = min(plan, key=plan.get)
        plan[category] += 1
    return plan


def prompt_for_questions(doc_id: str, label: str, context: str, plan: dict[str, int]) -> str:
    category_lines = "\n".join(
        f"- {category}: {count} ({CATEGORIES[category]})"
        for category, count in plan.items()
        if count > 0
    )
    return f"""
You are creating a high-quality evaluation dataset for a research-paper RAG system.

Use ONLY the provided chunks for answerable questions. Each answerable question must have
an answer that is directly supported by one or two listed chunk IDs.

Paper ID: {doc_id}
Paper/method label to use naturally in questions: {label}

Generate exactly this many questions:
{category_lines}

Quality rules:
1. Questions must be clear, specific, and useful for evaluating retrieval and grounded answer generation.
2. Do not ask vague questions such as "What is discussed in the paper?"
3. Every question must be self-contained: include the paper/method/model/dataset name or the specific technical topic.
4. Do not use phrases like "the paper", "this paper", "this work", "the method", "the model", "the approach", or "the context".
5. Do not ask author, title, publication year, venue, or generic contribution-summary questions.
6. Prefer natural wording such as "In Faster R-CNN, how does the Region Proposal Network share convolutional features?"
7. For answerable questions, include a concise ground_truth_answer and one or two source_chunk_ids.
8. For multihop questions, use exactly two source_chunk_ids.
9. For unanswerable questions, set answerable to 0, source_chunk_ids to [], and ground_truth_answer to "Insufficient evidence".
10. Unanswerable questions should be plausible and related to the named method/topic, but not answered by the chunks.
11. Output only valid JSON. No markdown.

JSON schema:
[
  {{
    "question": "...",
    "category": "factual|conceptual|methodology|comparison|limitation|multihop|unanswerable",
    "answerable": 1,
    "ground_truth_answer": "...",
    "source_chunk_ids": ["..."],
    "evidence_quote": "short supporting phrase from the chunk"
  }}
]

Chunks:
{context}
""".strip()


def clean_json(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_question(question: str) -> str:
    question = question.lower().strip()
    question = re.sub(r"[^\w\s]", "", question)
    return re.sub(r"\s+", " ", question)


def is_generic_question(question: str) -> bool:
    normalized = normalize_question(question)
    return any(pattern in normalized for pattern in GENERIC_QUESTION_PATTERNS)


def has_anchor(question: str, anchors: set[str]) -> bool:
    if not anchors:
        return True
    question_terms = set(re.findall(r"[a-z][a-z0-9]{3,}", question.lower()))
    return bool(question_terms & anchors)


def generate_for_paper(llm, doc_id: str, chunks: list, count: int, args: argparse.Namespace) -> list[dict]:
    sampled_chunks = sample_chunks(chunks, args.chunks_per_paper, args.seed + len(doc_id))
    valid_chunk_ids = {chunk.metadata.get("chunk_id") for chunk in sampled_chunks}
    label = paper_label(chunks)
    if not has_usable_label(label, doc_id):
        print(f"  Skipping weak/corrupted paper label: {doc_id} -> {label}")
        return []
    anchors = anchor_terms(label, doc_id)
    context = build_context(sampled_chunks, args.max_context_chars)
    response = llm.invoke(prompt_for_questions(doc_id, label, context, category_plan(count)))
    items = clean_json(response.content)

    rows = []
    for item in items:
        question = str(item.get("question", "")).strip()
        category = str(item.get("category", "")).strip()
        answerable = int(item.get("answerable", 1))
        source_ids = [str(chunk_id).strip() for chunk_id in item.get("source_chunk_ids", []) if str(chunk_id).strip()]

        if not question or category not in CATEGORIES:
            continue
        if is_generic_question(question):
            continue
        if not has_anchor(question, anchors):
            continue
        if answerable == 1 and not set(source_ids).issubset(valid_chunk_ids):
            continue
        if category == "multihop" and answerable == 1 and len(source_ids) != 2:
            continue
        if category == "unanswerable":
            answerable = 0
            source_ids = []

        first_chunk = next((chunk for chunk in sampled_chunks if chunk.metadata.get("chunk_id") in source_ids), None)
        first_meta = first_chunk.metadata if first_chunk else chunks[0].metadata

        rows.append({
            "doc_id": doc_id,
            "domain": first_meta.get("domain", ""),
            "paper_title": label,
            "file_name": first_meta.get("file_name", ""),
            "question": question,
            "category": category,
            "answerable": answerable,
            "ground_truth_answer": str(item.get("ground_truth_answer", "")).strip(),
            "source_chunk_ids": "|".join(source_ids),
            "evidence_quote": str(item.get("evidence_quote", "")).strip(),
        })
    return rows


def dedupe_and_limit(rows: list[dict], target: int) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = normalize_question(row["question"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    answerable = [row for row in deduped if int(row["answerable"]) == 1]
    unanswerable = [row for row in deduped if int(row["answerable"]) == 0]
    max_unanswerable = max(1, round(target * 0.12))
    final_rows = answerable[: target - max_unanswerable] + unanswerable[:max_unanswerable]
    return final_rows[:target]


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id",
        "domain",
        "paper_title",
        "file_name",
        "question",
        "category",
        "answerable",
        "ground_truth_answer",
        "source_chunk_ids",
        "evidence_quote",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\nQuestion dataset summary")
    print("-" * 40)
    print(f"Total questions: {len(rows)}")
    print(f"Answerable: {sum(int(row['answerable']) == 1 for row in rows)}")
    print(f"Unanswerable: {sum(int(row['answerable']) == 0 for row in rows)}")
    print("\nBy category:")
    for category, count in sorted(Counter(row["category"] for row in rows).items()):
        print(f"  {category}: {count}")
    print("\nBy domain:")
    for domain, count in sorted(Counter(row["domain"] for row in rows).items()):
        print(f"  {domain}: {count}")


def scan_question_quality(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    vague_rows = []
    for line_no, row in enumerate(rows, start=2):
        question = row.get("question", "")
        if is_generic_question(question):
            vague_rows.append((line_no, row))

    print(f"Scanned: {csv_path}")
    print(f"Total questions: {len(rows)}")
    print(f"Vague/generic questions: {len(vague_rows)}")
    for line_no, row in vague_rows[:25]:
        print(f"  line {line_no}: {row.get('question', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 100-150 grounded RAG evaluation questions.")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--quality-scan-csv", type=Path)
    parser.add_argument("--target-questions", type=int, default=150)
    parser.add_argument(
        "--candidate-questions",
        type=int,
        default=0,
        help="Generate this many raw candidates before filtering down to target-questions. Default: 3x target.",
    )
    parser.add_argument("--target-papers", type=int, default=20)
    parser.add_argument("--chunks-per-paper", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=14000)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.quality_scan_csv:
        scan_question_quality(args.quality_scan_csv)
        return

    if not 100 <= args.target_questions <= 150:
        raise ValueError("--target-questions should be between 100 and 150 for this evaluation setup.")

    chunks = load_chunks(args.chunks_path)
    grouped = group_chunks_by_doc(chunks)
    selected_doc_ids = representative_papers(grouped, args.target_papers, args.seed)
    candidate_questions = args.candidate_questions or args.target_questions * 3
    candidate_questions = max(candidate_questions, args.target_questions)
    per_paper_counts = questions_per_paper(candidate_questions, len(selected_doc_ids))

    llm = ChatOpenAI(model=args.model, temperature=0.2)
    rows = []
    for index, (doc_id, count) in enumerate(zip(selected_doc_ids, per_paper_counts), start=1):
        print(f"[{index}/{len(selected_doc_ids)}] Generating {count} questions for {doc_id}")
        try:
            rows.extend(generate_for_paper(llm, doc_id, grouped[doc_id], count, args))
        except Exception as exc:
            print(f"  Failed: {exc}")

    final_rows = dedupe_and_limit(rows, args.target_questions)
    save_csv(final_rows, args.output_csv)
    print_summary(final_rows)
    print(f"\nSaved questions to: {args.output_csv}")


if __name__ == "__main__":
    main()
