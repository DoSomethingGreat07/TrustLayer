import os
import re
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

# =========================
# CONFIG
# =========================
ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "TrustLayer-RAG-Project/1.0 (research use; contact: local-project)"
}

DOMAINS = {
    "nlp": [
        "Sequence to Sequence Learning with Neural Networks",
        "Neural Machine Translation by Jointly Learning to Align and Translate",
        "Effective Approaches to Attention-based Neural Machine Translation",
        "GloVe: Global Vectors for Word Representation",
        "Distributed Representations of Words and Phrases and their Compositionality",
        "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        "SpanBERT: Improving Pre-training by Representing and Predicting Spans",
        "DistilBERT: a distilled version of BERT",
        "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
        "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        "Universal Language Model Fine-tuning for Text Classification",
        "A Neural Probabilistic Language Model",
        "TextRank: Bringing Order into Text",
        "Question Answering Systems in the Deep Learning Era",
        "Named Entity Recognition with Bidirectional LSTM-CNNs",
        "Minimally Supervised Learning of Affective Events Using Discourse Relations"
    ],
    "llm": [
        "Language Models are Few-Shot Learners",
        "LLaMA: Open and Efficient Foundation Language Models",
        "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "Self-Instruct: Aligning Language Models with Self-Generated Instructions",
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "Training language models to follow instructions with human feedback",
        "Constitutional AI: Harmlessness from AI Feedback",
        "Alpaca: A Strong, Replicable Instruction-Following Model",
        "WizardLM: Empowering Large Language Models to Follow Complex Instructions",
        "Instruction Tuning for Large Language Models: A Survey"
    ],
    "transformers": [
        "Attention Is All You Need",
        "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
        "Longformer: The Long-Document Transformer",
        "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
        "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        "Reformer: The Efficient Transformer",
        "Performer: Linear Attention-Based Transformer Architectures",
        "Linformer: Self-Attention with Linear Complexity",
        "Big Bird: Transformers for Longer Sequences",
        "Vision Transformer"
    ],
    "rag": [
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "REALM: Retrieval-Augmented Language Model Pre-Training",
        "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "ATLAS: Few-shot Learning with Retrieval Augmented Language Models",
        "Fusion-in-Decoder for Knowledge Intensive Tasks",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "RETRO: Improving Language Models by Retrieving from Trillions of Tokens",
        "Precise Zero-Shot Dense Retrieval without Relevance Labels",
        "Hypothetical Document Embeddings"
    ]
}


# =========================
# HELPERS
# =========================
def safe_filename(text, max_len=140):
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


def build_url(search_query, start=0, max_results=3, sort_by="relevance", sort_order="descending"):
    return (
        f"{ARXIV_API}?search_query={quote_plus(search_query)}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy={sort_by}&sortOrder={sort_order}"
    )


def parse_feed(xml_text):
    root = ET.fromstring(xml_text)
    entries = []

    for entry in root.findall("atom:entry", NS):
        title = entry.find("atom:title", NS)
        summary = entry.find("atom:summary", NS)
        paper_id = entry.find("atom:id", NS)
        published = entry.find("atom:published", NS)

        authors = []
        for author in entry.findall("atom:author", NS):
            name = author.find("atom:name", NS)
            if name is not None:
                authors.append(name.text.strip())

        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", NS)]

        pdf_link = None
        for link in entry.findall("atom:link", NS):
            href = link.attrib.get("href", "")
            title_attr = link.attrib.get("title", "")
            if title_attr == "pdf":
                pdf_link = href if href.endswith(".pdf") else href + ".pdf"
                break

        if pdf_link is None and paper_id is not None:
            abs_url = paper_id.text.strip()
            if "/abs/" in abs_url:
                pdf_link = abs_url.replace("/abs/", "/pdf/") + ".pdf"

        entries.append({
            "id": paper_id.text.strip() if paper_id is not None else None,
            "title": title.text.strip().replace("\n", " ") if title is not None else "",
            "summary": summary.text.strip().replace("\n", " ") if summary is not None else "",
            "authors": authors,
            "published": published.text.strip() if published is not None else "",
            "categories": categories,
            "pdf_url": pdf_link,
        })

    return entries


def request_with_retry(url, timeout=30, retries=5):
    last_error = None

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=HEADERS)
            resp.raise_for_status()
            return resp

        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None

            if status == 429:
                wait_time = 5 * (attempt + 1)
                print(f"429 rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

        except requests.exceptions.RequestException as e:
            last_error = e
            wait_time = 3 * (attempt + 1)
            print(f"Request failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception(f"Request failed after {retries} retries: {last_error}")


def search_arxiv_by_title(title, max_results=3, retries=5):
    query = f'ti:"{title}"'
    url = build_url(query, max_results=max_results, sort_by="relevance", sort_order="descending")
    resp = request_with_retry(url, timeout=30, retries=retries)
    return parse_feed(resp.text)


def normalize_title(title):
    return re.sub(r"\s+", " ", title.strip().lower())


def pick_best_match(target_title, results):
    if not results:
        return None

    target_norm = normalize_title(target_title)

    for paper in results:
        cand = normalize_title(paper["title"])
        if target_norm == cand:
            return paper

    for paper in results:
        cand = normalize_title(paper["title"])
        if target_norm in cand or cand in target_norm:
            return paper

    return results[0]


def download_pdf(pdf_url, output_path, retries=5):
    last_error = None

    for attempt in range(retries):
        try:
            resp = requests.get(pdf_url, timeout=60, headers=HEADERS)
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True

        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None

            if status == 429:
                wait_time = 5 * (attempt + 1)
                print(f"429 during PDF download. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

        except requests.exceptions.RequestException as e:
            last_error = e
            wait_time = 3 * (attempt + 1)
            print(f"PDF download failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception(f"PDF download failed after {retries} retries: {last_error}")


def save_metadata(domain_dir, papers):
    metadata_path = os.path.join(domain_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


# =========================
# MAIN
# =========================
def main():
    all_papers = []

    for domain, titles in DOMAINS.items():
        print(f"\n=== Collecting {domain} papers ===")
        domain_dir = os.path.join(BASE_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)

        domain_papers = []

        for i, title in enumerate(titles, start=1):
            print(f"\nSearching for: {title}")

            try:
                results = search_arxiv_by_title(title, max_results=3, retries=5)
                best = pick_best_match(title, results)

                if best is None:
                    print("   -> No result found")
                    continue

                print(f"   -> Matched: {best['title']}")
                print(f"   -> Published: {best['published']}")
                print(f"   -> PDF: {best['pdf_url']}")

                filename = f"{i:02d}_{safe_filename(best['title'])}.pdf"
                filepath = os.path.join(domain_dir, filename)

                if best["pdf_url"]:
                    try:
                        download_pdf(best["pdf_url"], filepath, retries=5)
                        print(f"   -> Downloaded to {filepath}")
                    except Exception as e:
                        print(f"   -> Download failed: {e}")
                else:
                    print("   -> No PDF URL found, skipping download")

                paper_record = {
                    "domain": domain,
                    "requested_title": title,
                    **best,
                    "saved_filename": filename,
                    "saved_path": filepath,
                }

                domain_papers.append(paper_record)
                all_papers.append(paper_record)

            except Exception as e:
                print(f"   -> Search failed: {e}")
                continue

            # polite delay between papers
            time.sleep(4)

        save_metadata(domain_dir, domain_papers)

    with open(os.path.join(BASE_DIR, "all_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Total papers collected: {len(all_papers)}")


if __name__ == "__main__":
    main()
