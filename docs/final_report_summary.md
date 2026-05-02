# TrustLayer Experimental Report Summary

## Experimental Setup

| Component | Final Configuration |
| --- | --- |
| Corpus | 94 unique research papers / 95 PDFs |
| Evaluation set | 150 generated questions |
| Answerable questions | 132 |
| Unanswerable questions | 18 |
| Question quality | 0 vague/generic questions |
| Chunk size | 1200 characters |
| Chunk overlap | 250 characters |
| Final embedding model | BAAI/bge-base-en-v1.5 |
| Vector database | Chroma |
| Vector index | HNSW cosine |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Evaluation K | 1, 3, 5, 10 |

## Question Distribution

| Category | Questions | Share |
| --- | --- | --- |
| comparison | 22 | 0.1467 |
| conceptual | 25 | 0.1667 |
| factual | 19 | 0.1267 |
| limitation | 25 | 0.1667 |
| methodology | 22 | 0.1467 |
| multihop | 19 | 0.1267 |
| unanswerable | 18 | 0.12 |

## Retrieval Performance: BGE-base + 1200/250

| K | Precision | Recall | HitRate | MRR | nDCG | PaperHit |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1364 | 0.1288 | 0.1364 | 0.1364 | 0.1364 | 0.3485 |
| 3 | 0.0783 | 0.2235 | 0.2348 | 0.1755 | 0.1829 | 0.4015 |
| 5 | 0.0561 | 0.2576 | 0.2652 | 0.1827 | 0.1981 | 0.4091 |
| 10 | 0.0326 | 0.2955 | 0.303 | 0.1874 | 0.2105 | 0.4318 |

## Category-Wise Retrieval Performance at K=10

| Category | Questions | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 | PaperHit@10 |
| --- | --- | --- | --- | --- | --- | --- |
| comparison | 22 | 0.3182 | 0.2955 | 0.2784 | 0.2723 | 0.4545 |
| conceptual | 25 | 0.32 | 0.32 | 0.2267 | 0.2505 | 0.48 |
| factual | 19 | 0.4211 | 0.4211 | 0.2707 | 0.307 | 0.5263 |
| limitation | 25 | 0.24 | 0.24 | 0.1217 | 0.1498 | 0.4 |
| methodology | 22 | 0.3182 | 0.3182 | 0.0913 | 0.1447 | 0.4091 |
| multihop | 19 | 0.2105 | 0.1842 | 0.1447 | 0.1455 | 0.3158 |

## Verification Metrics: BGE-base + 1200/250

| Verification Parameter | Mean |
| --- | --- |
| Retrieval Confidence | 0.4191 |
| Reranker Confidence | 0.4328 |
| Evidence Similarity | 0.2889 |
| Grounding Score | 0.3295 |
| Evidence Coverage | 0.3636 |
| Combined Confidence | 0.3733 |
| NLI Max Entailment | 0.3667 |
| NLI Max Contradiction | 0.213 |

## Embedding Ablation

| Setup | Questions | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 | PaperHit@10 |
| --- | --- | --- | --- | --- | --- | --- |
| MPNet + 1200/250 | 132 | 0.2879 | 0.2765 | 0.1839 | 0.2023 | 0.3939 |
| BGE-base + 1200/250 | 132 | 0.303 | 0.2955 | 0.1874 | 0.2105 | 0.4318 |

## Fair Final Comparison Across K

| Setup | K | HitRate | Recall | MRR | nDCG | PaperHit |
| --- | --- | --- | --- | --- | --- | --- |
| MPNet + 1200/250 | 1 | 0.1364 | 0.1288 | 0.1364 | 0.1364 | 0.3333 |
| MPNet + 1200/250 | 3 | 0.2273 | 0.2159 | 0.173 | 0.1791 | 0.3864 |
| MPNet + 1200/250 | 5 | 0.2576 | 0.2462 | 0.1802 | 0.1923 | 0.3939 |
| MPNet + 1200/250 | 10 | 0.2879 | 0.2765 | 0.1839 | 0.2023 | 0.3939 |
| BGE-base + 1200/250 | 1 | 0.1364 | 0.1288 | 0.1364 | 0.1364 | 0.3485 |
| BGE-base + 1200/250 | 3 | 0.2348 | 0.2235 | 0.1755 | 0.1829 | 0.4015 |
| BGE-base + 1200/250 | 5 | 0.2652 | 0.2576 | 0.1827 | 0.1981 | 0.4091 |
| BGE-base + 1200/250 | 10 | 0.303 | 0.2955 | 0.1874 | 0.2105 | 0.4318 |

## Report-Ready Interpretation

The final TrustLayer configuration used BAAI/bge-base-en-v1.5 embeddings with 1200-character chunks and 250-character overlap. Retrieval combined dense vector search over a Chroma HNSW cosine index with BM25 sparse retrieval, reciprocal rank fusion, and cross-encoder reranking.

On the cleaned 150-question evaluation set, 132 answerable questions were used for retrieval metrics and 18 unanswerable questions were retained for abstention-oriented analysis. BGE-base achieved the strongest final retrieval result among the tested embedding models, with HitRate@10 = 0.3030, Recall@10 = 0.2955, MRR@10 = 0.1874, nDCG@10 = 0.2105, and PaperHit@10 = 0.4318.

## Limitations

The retrieval metrics use strict chunk-level matching, so a retrieved neighboring chunk or a different chunk from the correct paper may still be counted as incorrect. Some PDFs also contain noisy or corrupted extracted text, which can reduce embedding quality, sparse matching quality, question generation quality, and evidence localization.

## Future Work

Future improvements include OCR/text cleanup for noisy PDFs, neighbor-tolerant evidence matching, stronger rerankers, larger retrieval-focused embedding models, and domain-aware query expansion.
