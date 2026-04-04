import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_QUESTIONS_CSV = Path("artifacts/generated_questions.csv")
DEFAULT_OUTPUT_CSV = Path("artifacts/retrieval_metrics_per_query.csv")


def parse_chunk_ids(raw_value: str) -> set[str]:
    if not raw_value:
        return set()

    chunk_ids = set()
    for part in raw_value.split("|"):
        cleaned = part.strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("chunk_id:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        chunk_ids.add(cleaned)
    return chunk_ids


def load_question_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def get_top_chunk_ids(query: str, pipeline: dict, dense_k: int, sparse_k: int, fusion_k: int, final_k: int) -> list[str]:
    from corrective_Rag_pipeline import (
        dense_retrieve,
        sparse_retrieve,
        rrf_fusion,
        rerank_documents,
    )

    dense_docs = dense_retrieve(pipeline["vectorstore"], query, k=dense_k)
    sparse_docs = sparse_retrieve(pipeline["bm25"], pipeline["chunks"], query, k=sparse_k)
    fused_docs = rrf_fusion(dense_docs, sparse_docs, top_k=fusion_k)
    reranked_docs = rerank_documents(
        pipeline["reranker"],
        query,
        fused_docs,
        top_k=final_k,
    )
    return [str(item["doc"].metadata.get("chunk_id", "")).strip() for item in reranked_docs]


def compute_metrics_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    top_k_ids = retrieved_ids[:k]

    if not relevant_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit": 0,
            "relevant_retrieved": 0,
        }

    relevant_retrieved = sum(1 for chunk_id in top_k_ids if chunk_id in relevant_ids)

    return {
        "precision": relevant_retrieved / k if k > 0 else 0.0,
        "recall": relevant_retrieved / len(relevant_ids),
        "hit": int(relevant_retrieved > 0),
        "relevant_retrieved": relevant_retrieved,
    }


def evaluate_rows(
    rows: list[dict],
    pipeline: dict,
    ks: list[int],
    dense_k: int,
    sparse_k: int,
    fusion_k: int,
    final_k: int,
    answerable_only: bool,
) -> list[dict]:
    evaluated_rows = []

    for row_id, row in enumerate(rows, start=1):
        answerable = int(row.get("answerable", 0))
        relevant_ids = parse_chunk_ids(row.get("source_chunk_ids", ""))

        if answerable_only and answerable != 1:
            continue
        if answerable == 1 and not relevant_ids:
            continue

        retrieved_ids = get_top_chunk_ids(
            query=row["question"],
            pipeline=pipeline,
            dense_k=dense_k,
            sparse_k=sparse_k,
            fusion_k=fusion_k,
            final_k=final_k,
        )

        base_row = {
            "row_id": row_id,
            "doc_id": row.get("doc_id", ""),
            "question": row.get("question", ""),
            "category": row.get("category", ""),
            "answerable": answerable,
            "ground_truth_chunk_ids": "|".join(sorted(relevant_ids)),
            "retrieved_chunk_ids": "|".join(retrieved_ids),
        }

        for k in ks:
            metrics = compute_metrics_at_k(retrieved_ids, relevant_ids, k)
            base_row[f"precision@{k}"] = round(metrics["precision"], 4)
            base_row[f"recall@{k}"] = round(metrics["recall"], 4)
            base_row[f"hit@{k}"] = metrics["hit"]
            base_row[f"relevant_retrieved@{k}"] = metrics["relevant_retrieved"]

        evaluated_rows.append(base_row)

    return evaluated_rows


def summarize_results(rows: list[dict], ks: list[int]) -> dict:
    summary = {"num_questions": len(rows)}

    if not rows:
        for k in ks:
            summary[f"precision@{k}"] = 0.0
            summary[f"recall@{k}"] = 0.0
            summary[f"hit_rate@{k}"] = 0.0
        return summary

    for k in ks:
        summary[f"precision@{k}"] = round(mean(row[f"precision@{k}"] for row in rows), 4)
        summary[f"recall@{k}"] = round(mean(row[f"recall@{k}"] for row in rows), 4)
        summary[f"hit_rate@{k}"] = round(mean(row[f"hit@{k}"] for row in rows), 4)

    return summary


def summarize_by_category(rows: list[dict], ks: list[int]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    return {
        category: summarize_results(category_rows, ks)
        for category, category_rows in sorted(grouped.items())
    }


def save_results(rows: list[dict], output_csv: Path, ks: list[int]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_id",
        "doc_id",
        "question",
        "category",
        "answerable",
        "ground_truth_chunk_ids",
        "retrieved_chunk_ids",
    ]
    for k in ks:
        fieldnames.extend([
            f"precision@{k}",
            f"recall@{k}",
            f"hit@{k}",
            f"relevant_retrieved@{k}",
        ])

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval Precision@K and Recall@K using source_chunk_ids as pseudo ground truth."
    )
    parser.add_argument(
        "--questions-csv",
        type=Path,
        default=DEFAULT_QUESTIONS_CSV,
        help=f"Path to generated questions CSV. Default: {DEFAULT_QUESTIONS_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to write per-query retrieval metrics. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values to evaluate. Example: --ks 1 3 5",
    )
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--sparse-k", type=int, default=20)
    parser.add_argument("--fusion-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include unanswerable questions in the output CSV. They are excluded from aggregate metrics.",
    )
    return parser.parse_args()


def main() -> None:
    from main import prepare_pipeline

    args = parse_args()

    if not args.questions_csv.exists():
        raise FileNotFoundError(f"Questions CSV not found: {args.questions_csv}")

    max_requested_k = max(args.ks)
    if args.final_k < max_requested_k:
        raise ValueError(
            f"--final-k must be >= max(--ks). Received final_k={args.final_k}, max_k={max_requested_k}"
        )

    print("Preparing retrieval pipeline...")
    pipeline = prepare_pipeline(
        force_rebuild_documents=False,
        force_rebuild_chunks=False,
        force_rebuild_vectordb=False,
        use_api_enrichment=True,
        chunk_size=800,
        chunk_overlap=150,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        device="cpu",
    )

    print(f"Loading questions from: {args.questions_csv}")
    question_rows = load_question_rows(args.questions_csv)

    print("Running retrieval evaluation...")
    evaluated_rows = evaluate_rows(
        rows=question_rows,
        pipeline=pipeline,
        ks=sorted(set(args.ks)),
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        fusion_k=args.fusion_k,
        final_k=args.final_k,
        answerable_only=not args.include_unanswerable,
    )

    metric_rows = [row for row in evaluated_rows if int(row["answerable"]) == 1]
    overall_summary = summarize_results(metric_rows, sorted(set(args.ks)))
    category_summary = summarize_by_category(metric_rows, sorted(set(args.ks)))

    save_results(evaluated_rows, args.output_csv, sorted(set(args.ks)))

    print(f"\nSaved per-query metrics to: {args.output_csv}")
    print("\nOverall retrieval metrics")
    print("-" * 40)
    print(f"Questions evaluated: {overall_summary['num_questions']}")
    for k in sorted(set(args.ks)):
        print(
            f"Precision@{k}: {overall_summary[f'precision@{k}']:.4f} | "
            f"Recall@{k}: {overall_summary[f'recall@{k}']:.4f} | "
            f"HitRate@{k}: {overall_summary[f'hit_rate@{k}']:.4f}"
        )

    if category_summary:
        print("\nBy category")
        print("-" * 40)
        for category, summary in category_summary.items():
            print(f"{category} ({summary['num_questions']} questions)")
            for k in sorted(set(args.ks)):
                print(
                    f"  Precision@{k}: {summary[f'precision@{k}']:.4f} | "
                    f"Recall@{k}: {summary[f'recall@{k}']:.4f} | "
                    f"HitRate@{k}: {summary[f'hit_rate@{k}']:.4f}"
                )


if __name__ == "__main__":
    main()
