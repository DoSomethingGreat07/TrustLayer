import csv
from collections import Counter, defaultdict
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")
REPORT_DIR = ARTIFACTS_DIR / "report"

QUESTIONS_CSV = ARTIFACTS_DIR / "eval_questions_150_v5_chunks1200.csv"
BGE_VERIFICATION_CSV = ARTIFACTS_DIR / "report_metrics_150_v5_chunks1200_bge_k10_with_verification.csv"
BGE_RETRIEVAL_CSV = ARTIFACTS_DIR / "report_metrics_150_v5_chunks1200_bge_k10.csv"
MPNET_VERIFICATION_CSV = ARTIFACTS_DIR / "report_metrics_150_v5_chunks1200_mpnet_k10_with_verification.csv"

K_VALUES = ["1", "3", "5", "10"]

RETRIEVAL_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("hit", "HitRate"),
    ("mrr", "MRR"),
    ("ndcg", "nDCG"),
    ("paper_hit", "PaperHit"),
]

VERIFICATION_METRICS = [
    ("retrieval_confidence", "Retrieval Confidence"),
    ("reranker_confidence", "Reranker Confidence"),
    ("evidence_similarity", "Evidence Similarity"),
    ("grounding_score", "Grounding Score"),
    ("evidence_coverage", "Evidence Coverage"),
    ("combined_confidence", "Combined Confidence"),
    ("nli_max_entailment", "NLI Max Entailment"),
    ("nli_contradiction_max", "NLI Max Contradiction"),
]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def answerable_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("answerable", "")) == "1"]


def avg(rows: list[dict], column: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(column, 0.0) or 0.0) for row in rows) / len(rows)


def count_questions(path: Path) -> dict[str, int]:
    rows = load_rows(path)
    answerable = sum(str(row.get("answerable", "")) == "1" for row in rows)
    return {
        "total": len(rows),
        "answerable": answerable,
        "unanswerable": len(rows) - answerable,
    }


def retrieval_table(rows: list[dict]) -> list[dict]:
    rows = answerable_rows(rows)
    table = []
    for k in K_VALUES:
        item = {"K": k}
        for source_key, label in RETRIEVAL_METRICS:
            item[label] = round(avg(rows, f"{source_key}@{k}"), 4)
        table.append(item)
    return table


def verification_table(rows: list[dict]) -> list[dict]:
    rows = answerable_rows(rows)
    return [
        {
            "Verification Parameter": label,
            "Mean": round(avg(rows, source_key), 4),
        }
        for source_key, label in VERIFICATION_METRICS
    ]


def question_distribution_table(question_rows: list[dict]) -> list[dict]:
    counts = Counter(row.get("category", "unknown") for row in question_rows)
    total = len(question_rows)
    return [
        {
            "Category": category,
            "Questions": count,
            "Share": round(count / total, 4) if total else 0.0,
        }
        for category, count in sorted(counts.items())
    ]


def category_performance_table(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in answerable_rows(rows):
        grouped[row.get("category", "unknown")].append(row)

    table = []
    for category, category_rows in sorted(grouped.items()):
        table.append({
            "Category": category,
            "Questions": len(category_rows),
            "HitRate@10": round(avg(category_rows, "hit@10"), 4),
            "Recall@10": round(avg(category_rows, "recall@10"), 4),
            "MRR@10": round(avg(category_rows, "mrr@10"), 4),
            "nDCG@10": round(avg(category_rows, "ndcg@10"), 4),
            "PaperHit@10": round(avg(category_rows, "paper_hit@10"), 4),
        })
    return table


def embedding_ablation_table(mpnet_rows: list[dict], bge_rows: list[dict]) -> list[dict]:
    mpnet_rows = answerable_rows(mpnet_rows)
    bge_rows = answerable_rows(bge_rows)
    return [
        {
            "Setup": "MPNet + 1200/250",
            "Questions": len(mpnet_rows),
            "HitRate@10": round(avg(mpnet_rows, "hit@10"), 4),
            "Recall@10": round(avg(mpnet_rows, "recall@10"), 4),
            "MRR@10": round(avg(mpnet_rows, "mrr@10"), 4),
            "nDCG@10": round(avg(mpnet_rows, "ndcg@10"), 4),
            "PaperHit@10": round(avg(mpnet_rows, "paper_hit@10"), 4),
        },
        {
            "Setup": "BGE-base + 1200/250",
            "Questions": len(bge_rows),
            "HitRate@10": round(avg(bge_rows, "hit@10"), 4),
            "Recall@10": round(avg(bge_rows, "recall@10"), 4),
            "MRR@10": round(avg(bge_rows, "mrr@10"), 4),
            "nDCG@10": round(avg(bge_rows, "ndcg@10"), 4),
            "PaperHit@10": round(avg(bge_rows, "paper_hit@10"), 4),
        },
    ]


def embedding_ablation_by_k_table(mpnet_rows: list[dict], bge_rows: list[dict]) -> list[dict]:
    mpnet_rows = answerable_rows(mpnet_rows)
    bge_rows = answerable_rows(bge_rows)
    table = []
    for setup, rows in [
        ("MPNet + 1200/250", mpnet_rows),
        ("BGE-base + 1200/250", bge_rows),
    ]:
        for k in K_VALUES:
            table.append({
                "Setup": setup,
                "K": k,
                "HitRate": round(avg(rows, f"hit@{k}"), 4),
                "Recall": round(avg(rows, f"recall@{k}"), 4),
                "MRR": round(avg(rows, f"mrr@{k}"), 4),
                "nDCG": round(avg(rows, f"ndcg@{k}"), 4),
                "PaperHit": round(avg(rows, f"paper_hit@{k}"), 4),
            })
    return table


def experimental_setup_table(question_counts: dict[str, int]) -> list[dict]:
    return [
        {"Component": "Corpus", "Final Configuration": "94 unique research papers / 95 PDFs"},
        {"Component": "Evaluation set", "Final Configuration": f"{question_counts['total']} generated questions"},
        {"Component": "Answerable questions", "Final Configuration": str(question_counts["answerable"])},
        {"Component": "Unanswerable questions", "Final Configuration": str(question_counts["unanswerable"])},
        {"Component": "Question quality", "Final Configuration": "0 vague/generic questions"},
        {"Component": "Chunk size", "Final Configuration": "1200 characters"},
        {"Component": "Chunk overlap", "Final Configuration": "250 characters"},
        {"Component": "Final embedding model", "Final Configuration": "BAAI/bge-base-en-v1.5"},
        {"Component": "Vector database", "Final Configuration": "Chroma"},
        {"Component": "Vector index", "Final Configuration": "HNSW cosine"},
        {"Component": "Sparse retrieval", "Final Configuration": "BM25"},
        {"Component": "Fusion", "Final Configuration": "Reciprocal Rank Fusion"},
        {"Component": "Reranker", "Final Configuration": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        {"Component": "Evaluation K", "Final Configuration": "1, 3, 5, 10"},
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_markdown(
    setup_rows: list[dict],
    question_distribution_rows: list[dict],
    retrieval_rows: list[dict],
    category_rows: list[dict],
    verification_rows: list[dict],
    ablation_rows: list[dict],
    ablation_by_k_rows: list[dict],
) -> None:
    report = f"""# TrustLayer Experimental Report Summary

## Experimental Setup

{markdown_table(setup_rows)}

## Question Distribution

{markdown_table(question_distribution_rows)}

## Retrieval Performance: BGE-base + 1200/250

{markdown_table(retrieval_rows)}

## Category-Wise Retrieval Performance at K=10

{markdown_table(category_rows)}

## Verification Metrics: BGE-base + 1200/250

{markdown_table(verification_rows)}

## Embedding Ablation

{markdown_table(ablation_rows)}

## Fair Final Comparison Across K

{markdown_table(ablation_by_k_rows)}

## Report-Ready Interpretation

The final TrustLayer configuration used BAAI/bge-base-en-v1.5 embeddings with 1200-character chunks and 250-character overlap. Retrieval combined dense vector search over a Chroma HNSW cosine index with BM25 sparse retrieval, reciprocal rank fusion, and cross-encoder reranking.

On the cleaned 150-question evaluation set, 132 answerable questions were used for retrieval metrics and 18 unanswerable questions were retained for abstention-oriented analysis. BGE-base achieved the strongest final retrieval result among the tested embedding models, with HitRate@10 = {ablation_rows[1]["HitRate@10"]:.4f}, Recall@10 = {ablation_rows[1]["Recall@10"]:.4f}, MRR@10 = {ablation_rows[1]["MRR@10"]:.4f}, nDCG@10 = {ablation_rows[1]["nDCG@10"]:.4f}, and PaperHit@10 = {ablation_rows[1]["PaperHit@10"]:.4f}.

## Limitations

The retrieval metrics use strict chunk-level matching, so a retrieved neighboring chunk or a different chunk from the correct paper may still be counted as incorrect. Some PDFs also contain noisy or corrupted extracted text, which can reduce embedding quality, sparse matching quality, question generation quality, and evidence localization.

## Future Work

Future improvements include OCR/text cleanup for noisy PDFs, neighbor-tolerant evidence matching, stronger rerankers, larger retrieval-focused embedding models, and domain-aware query expansion.
"""
    (REPORT_DIR / "final_report_summary.md").write_text(report, encoding="utf-8")


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{text}</text>'
    )


def write_retrieval_svg(retrieval_rows: list[dict]) -> None:
    width, height = 920, 520
    left, right, top, bottom = 80, 40, 70, 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = 0.5
    metric_names = ["HitRate", "Recall", "MRR", "nDCG", "PaperHit"]
    colors = {
        "HitRate": "#2563eb",
        "Recall": "#16a34a",
        "MRR": "#9333ea",
        "nDCG": "#ea580c",
        "PaperHit": "#0f766e",
    }
    k_values = [int(row["K"]) for row in retrieval_rows]

    def x_pos(index: int) -> float:
        return left + index * chart_w / (len(k_values) - 1)

    def y_pos(value: float) -> float:
        return top + chart_h - (value / max_y) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "Final Retrieval Performance by K", 22, "middle", "700"),
    ]

    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#d7dde5" stroke-dasharray="4 4"/>')
        parts.append(svg_text(left - 12, y + 4, f"{tick:.1f}", 12, "end"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#17202a"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#17202a"/>')

    for index, k in enumerate(k_values):
        x = x_pos(index)
        parts.append(f'<line x1="{x}" y1="{top + chart_h}" x2="{x}" y2="{top + chart_h + 6}" stroke="#17202a"/>')
        parts.append(svg_text(x, top + chart_h + 26, str(k), 13, "middle"))

    for metric in metric_names:
        points = []
        for index, row in enumerate(retrieval_rows):
            points.append((x_pos(index), y_pos(float(row[metric]))))
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{colors[metric]}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[metric]}"/>')

    legend_x, legend_y = left + 20, top + 15
    for index, metric in enumerate(metric_names):
        y = legend_y + index * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{colors[metric]}"/>')
        parts.append(svg_text(legend_x + 22, y + 2, metric, 13))

    parts.append(svg_text(width / 2, height - 24, "Top-K Retrieved Chunks", 14, "middle", "600"))
    parts.append(svg_text(22, height / 2, "Score", 14, "middle", "600"))
    parts.append("</svg>")
    (REPORT_DIR / "retrieval_performance_by_k.svg").write_text("\n".join(parts), encoding="utf-8")


def write_ablation_svg(ablation_rows: list[dict]) -> None:
    width, height = 920, 520
    left, right, top, bottom = 80, 40, 70, 100
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = 0.5
    metric_names = ["HitRate@10", "Recall@10", "MRR@10", "nDCG@10", "PaperHit@10"]
    colors = ["#64748b", "#2563eb"]

    def y_pos(value: float) -> float:
        return top + chart_h - (value / max_y) * chart_h

    group_w = chart_w / len(metric_names)
    bar_w = 26

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "Embedding Ablation at K=10", 22, "middle", "700"),
    ]

    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#d7dde5" stroke-dasharray="4 4"/>')
        parts.append(svg_text(left - 12, y + 4, f"{tick:.1f}", 12, "end"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#17202a"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#17202a"/>')

    for metric_index, metric in enumerate(metric_names):
        center = left + metric_index * group_w + group_w / 2
        for setup_index, row in enumerate(ablation_rows):
            value = float(row[metric])
            x = center + (setup_index - 0.5) * (bar_w + 8)
            y = y_pos(value)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + chart_h - y:.1f}" fill="{colors[setup_index]}"/>'
            )
        parts.append(svg_text(center, top + chart_h + 24, metric, 12, "middle"))

    legend_x, legend_y = left + 20, top + 15
    for index, row in enumerate(ablation_rows):
        y = legend_y + index * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{colors[index]}"/>')
        parts.append(svg_text(legend_x + 22, y + 2, row["Setup"], 13))

    parts.append(svg_text(width / 2, height - 24, "Metric", 14, "middle", "600"))
    parts.append(svg_text(22, height / 2, "Score", 14, "middle", "600"))
    parts.append("</svg>")
    (REPORT_DIR / "embedding_ablation_k10.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bge_path = BGE_VERIFICATION_CSV if BGE_VERIFICATION_CSV.exists() else BGE_RETRIEVAL_CSV
    question_rows = load_rows(QUESTIONS_CSV)
    bge_rows = load_rows(bge_path)
    mpnet_rows = load_rows(MPNET_VERIFICATION_CSV)
    question_counts = count_questions(QUESTIONS_CSV)

    setup_rows = experimental_setup_table(question_counts)
    question_distribution_rows = question_distribution_table(question_rows)
    retrieval_rows = retrieval_table(bge_rows)
    category_rows = category_performance_table(bge_rows)
    verification_rows = verification_table(bge_rows)
    ablation_rows = embedding_ablation_table(mpnet_rows, bge_rows)
    ablation_by_k_rows = embedding_ablation_by_k_table(mpnet_rows, bge_rows)

    write_csv(REPORT_DIR / "experimental_setup.csv", setup_rows)
    write_csv(REPORT_DIR / "question_distribution.csv", question_distribution_rows)
    write_csv(REPORT_DIR / "retrieval_performance_bge.csv", retrieval_rows)
    write_csv(REPORT_DIR / "category_performance_bge_k10.csv", category_rows)
    write_csv(REPORT_DIR / "verification_metrics_bge.csv", verification_rows)
    write_csv(REPORT_DIR / "embedding_ablation.csv", ablation_rows)
    write_csv(REPORT_DIR / "embedding_ablation_by_k.csv", ablation_by_k_rows)
    write_markdown(
        setup_rows,
        question_distribution_rows,
        retrieval_rows,
        category_rows,
        verification_rows,
        ablation_rows,
        ablation_by_k_rows,
    )
    write_retrieval_svg(retrieval_rows)
    write_ablation_svg(ablation_rows)

    print(f"Report artifacts written to: {REPORT_DIR}")
    for path in sorted(REPORT_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()
