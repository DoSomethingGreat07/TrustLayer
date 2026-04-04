# app.py

import hmac
import os

import streamlit as st
from dotenv import load_dotenv

from main import prepare_pipeline
from corrective_Rag_pipeline import corrective_rag_pipeline_v2
from llm_generate import llm_generate_fn


load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TrustLayer Research Assistant",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# HELPERS
# ============================================================
def inject_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --tl-bg: #10141b;
            --tl-surface: #171c24;
            --tl-surface-2: #1d2430;
            --tl-panel-border: rgba(255,255,255,0.09);
            --tl-accent: #f0c35b;
            --tl-accent-2: #7fd1dc;
            --tl-text: #f4f7fb;
            --tl-muted: #aeb8c7;
            --tl-good: #8fd66b;
            --tl-warn: #f2b36d;
        }

        .stApp {
            background: linear-gradient(180deg, #0d1117 0%, #121821 100%);
            color: var(--tl-text);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.25rem;
            max-width: 1240px;
        }

        .tl-login-mode .block-container {
            padding-top: 0.4rem;
        }

        .tl-hero {
            padding: 1.9rem 2rem;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            background:
                linear-gradient(90deg, rgba(240,195,91,0.08), transparent 28%),
                linear-gradient(180deg, #171d27 0%, #141a23 100%);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
            margin-bottom: 1.4rem;
        }

        .tl-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.78rem;
            color: var(--tl-accent);
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .tl-title {
            font-size: 2.5rem;
            line-height: 1.02;
            font-weight: 800;
            color: var(--tl-text);
            margin: 0;
            letter-spacing: -0.03em;
        }

        .tl-subtitle {
            margin-top: 0.85rem;
            color: var(--tl-muted);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 50rem;
        }

        .tl-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.15rem;
        }

        .tl-chip {
            padding: 0.46rem 0.82rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.09);
            color: var(--tl-text);
            font-size: 0.84rem;
        }

        .tl-section-card {
            padding: 1.05rem 1.15rem;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.07);
            background: var(--tl-surface);
            margin-bottom: 1rem;
        }

        .tl-answer {
            padding: 1.15rem 1.2rem;
            border-radius: 20px;
            border: 1px solid rgba(240,195,91,0.18);
            background: linear-gradient(180deg, rgba(240,195,91,0.05), rgba(255,255,255,0.01)), #1b222d;
            color: var(--tl-text);
        }

        .tl-justification {
            padding: 1.15rem 1.2rem;
            border-radius: 20px;
            border: 1px solid rgba(127,209,220,0.18);
            background: linear-gradient(180deg, rgba(127,209,220,0.05), rgba(255,255,255,0.01)), #1a212c;
            color: var(--tl-text);
        }

        .tl-mini-title {
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--tl-muted);
            margin-bottom: 0.55rem;
            font-weight: 700;
        }

        .tl-status {
            display: inline-block;
            padding: 0.36rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.45rem;
            margin-bottom: 0.6rem;
        }

        .tl-status-good {
            background: #1d2a1f;
            color: #c9efb4;
            border: 1px solid rgba(143,214,107,0.28);
        }

        .tl-status-warn {
            background: #2b221a;
            color: #ffdcb8;
            border: 1px solid rgba(242,179,109,0.28);
        }

        .tl-status-neutral {
            background: #202734;
            color: var(--tl-text);
            border: 1px solid rgba(255,255,255,0.10);
        }

        .tl-empty {
            padding: 1.3rem 1.35rem;
            border-radius: 20px;
            border: 1px dashed rgba(255,255,255,0.15);
            background: var(--tl-surface);
            color: var(--tl-muted);
            line-height: 1.7;
        }

        [data-testid="stMetric"] {
            background: var(--tl-surface-2);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        .stMarkdown,
        .stText,
        .stCaption,
        label,
        p,
        li,
        div {
            color: var(--tl-text);
        }

        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricValue"] > div,
        .stCaption {
            color: var(--tl-muted) !important;
        }

        [data-testid="stSidebar"] {
            background: #11161e;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .tl-login-mode [data-testid="stSidebar"] {
            display: none;
        }

        .tl-login-mode [data-testid="collapsedControl"] {
            display: none;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stExpander"] {
            background: var(--tl-surface);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
            margin-bottom: 0.75rem;
        }

        button[role="tab"] {
            background: #171d27 !important;
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 14px !important;
            color: var(--tl-muted) !important;
            padding: 0.35rem 0.9rem !important;
        }

        button[role="tab"][aria-selected="true"] {
            background: #202b3a !important;
            border-color: rgba(240,195,91,0.28) !important;
            color: var(--tl-text) !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        .stNumberInput input {
            background: #151b24 !important;
            color: var(--tl-text) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
        }

        .stButton button,
        .stFormSubmitButton button {
            background: #243044 !important;
            color: var(--tl-text) !important;
            border: 1px solid rgba(240,195,91,0.28) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
        }

        .stButton button:hover,
        .stFormSubmitButton button:hover {
            transform: translateY(-1px);
            background: #29364a !important;
            border-color: rgba(240,195,91,0.38) !important;
        }

        .tl-sidebar-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #161d27 0%, #121821 100%);
            border: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 0.85rem;
        }

        .tl-sidebar-title {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--tl-accent);
            margin-bottom: 0.55rem;
            font-weight: 800;
        }

        .tl-sidebar-stat {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--tl-text);
            line-height: 1.1;
        }

        .tl-sidebar-label {
            color: var(--tl-muted);
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }

        .tl-sidebar-divider {
            height: 1px;
            background: rgba(255,255,255,0.06);
            margin: 0.85rem 0;
        }

        .tl-evidence-card {
            padding: 1rem 1.05rem 1.05rem 1.05rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #171e28 0%, #141a23 100%);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .tl-evidence-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--tl-accent);
            margin-bottom: 0.45rem;
            font-weight: 800;
        }

        .tl-evidence-title {
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.35;
            color: var(--tl-text);
            margin-bottom: 0.55rem;
        }

        .tl-evidence-meta {
            color: var(--tl-muted);
            font-size: 0.88rem;
            line-height: 1.55;
            margin-bottom: 0.8rem;
        }

        .tl-evidence-snippet {
            padding: 0.85rem 0.95rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            color: var(--tl-text);
            line-height: 1.7;
            font-size: 0.95rem;
        }

        .tl-context-box {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            background: #0f141c;
            border: 1px solid rgba(127,209,220,0.16);
            color: #e8eef7;
            line-height: 1.72;
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .tl-login-shell {
            min-height: 86vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .tl-login-card {
            width: 100%;
            max-width: 720px;
            padding: 2rem 2rem 1.7rem 2rem;
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(240,195,91,0.10), transparent 32%),
                linear-gradient(180deg, #171e28 0%, #121821 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.20);
        }

        .tl-login-eyebrow {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--tl-accent);
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .tl-login-title {
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--tl-text);
            margin-bottom: 0.7rem;
        }

        .tl-login-copy {
            color: var(--tl-muted);
            line-height: 1.7;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .tl-login-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .tl-login-panel {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
        }

        .tl-login-panel strong {
            color: var(--tl-text);
        }

        .tl-login-list {
            margin: 0;
            padding-left: 1.1rem;
            color: var(--tl-muted);
            line-height: 1.8;
        }

        .tl-login-form-wrap {
            margin-top: 0.2rem;
            padding: 1rem 1rem 0.25rem 1rem;
            border-radius: 18px;
            background: rgba(11, 15, 21, 0.28);
            border: 1px solid rgba(255,255,255,0.07);
        }

        .tl-login-hint {
            margin-top: 0.85rem;
            color: var(--tl-muted);
            font-size: 0.9rem;
        }

        .tl-login-badge {
            display: inline-block;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--tl-text);
            font-size: 0.82rem;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        div[data-testid="stTextInput"] {
            margin-bottom: 0.65rem;
        }

        div[data-testid="stTextInput"] > label,
        div[data-testid="stTextInputRootElement"] + label {
            color: var(--tl-muted) !important;
        }

        .stTextInput input {
            min-height: 2.9rem;
        }

        @media (max-width: 900px) {
            .tl-login-grid {
                grid-template-columns: 1fr;
            }

            .tl-login-card {
                padding: 1.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_login_mode_styles():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="collapsedControl"] {
            display: none;
        }

        .block-container {
            padding-top: 0.35rem !important;
            max-width: 1080px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(documents_count: int, paper_count: int, domains: list[str]):
    domain_text = " | ".join(domains[:4]) if domains else "Research corpus"
    st.markdown(
        f"""
        <section class="tl-hero">
            <div class="tl-eyebrow">Trust-Aware Retrieval Workspace</div>
            <h1 class="tl-title">TrustLayer Research Assistant</h1>
            <p class="tl-subtitle">
                Ask grounded questions across your paper collection and inspect how retrieval,
                reranking, and verification shaped the final answer.
            </p>
            <div class="tl-chip-row">
                <span class="tl-chip">{paper_count} papers indexed</span>
                <span class="tl-chip">{documents_count} document pages loaded</span>
                <span class="tl-chip">{domain_text}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_chat_state():
    st.markdown(
        """
        <div class="tl-empty">
            Ask a question to start a grounded research conversation. The first response will show
            answer quality, verification signals, evidence chunks, and the exact context sent to the generator.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_overview(paper_count: int, domains: list[str], chat_count: int):
    domain_preview = ", ".join(domains[:3]) if domains else "No domains"
    st.sidebar.markdown(
        f"""
        <div class="tl-sidebar-card">
            <div class="tl-sidebar-title">Corpus Overview</div>
            <div class="tl-sidebar-stat">{paper_count}</div>
            <div class="tl-sidebar-label">Indexed papers</div>
            <div class="tl-sidebar-divider"></div>
            <div class="tl-sidebar-stat">{len(domains)}</div>
            <div class="tl-sidebar-label">Domains: {domain_preview}</div>
            <div class="tl-sidebar-divider"></div>
            <div class="tl-sidebar-stat">{chat_count}</div>
            <div class="tl-sidebar-label">Saved turns in your history</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_paper_browser(paper_catalog: dict):
    st.sidebar.markdown(
        """
        <div class="tl-sidebar-card">
            <div class="tl-sidebar-title">Paper Browser</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    papers_by_domain = {}
    for _, paper in paper_catalog.items():
        papers_by_domain.setdefault(paper["domain"], []).append(paper)

    for domain in sorted(papers_by_domain):
        papers = papers_by_domain[domain]
        with st.sidebar.expander(f"{domain.title()} ({len(papers)})", expanded=False):
            for paper in papers:
                with st.container():
                    st.markdown(f"**{paper['title']}**")
                    st.caption(paper["authors"])
                    st.write(f"File: {paper['file_name']}")
                    st.write(f"Source: {paper['metadata_source']}")
                    st.markdown("---")


def render_status_pills(result: dict):
    corrected = result.get("corrected", False)
    abstained = result.get("abstained", False)
    verified = result.get("verification", {}).get("verified", False)

    pill_specs = [
        ("Corrected", corrected, "tl-status-good" if corrected else "tl-status-neutral"),
        ("Verified", verified, "tl-status-good" if verified else "tl-status-warn"),
        ("Abstained", abstained, "tl-status-warn" if abstained else "tl-status-neutral"),
        (f"Retries: {result.get('retries_used', 0)}", None, "tl-status-neutral"),
    ]

    pills = []
    for label, value, css_class in pill_specs:
        text = f"{label}: {value}" if isinstance(value, bool) else label
        pills.append(f'<span class="tl-status {css_class}">{text}</span>')

    st.markdown("".join(pills), unsafe_allow_html=True)


def render_evidence_card(item: dict, idx: int):
    doc = item["doc"]
    meta = doc.metadata
    st.markdown(
        f"""
        <div class="tl-evidence-card">
            <div class="tl-evidence-kicker">Evidence {idx}</div>
            <div class="tl-evidence-title">{meta.get('title', 'Unknown')}</div>
            <div class="tl-evidence-meta">
                <strong>Authors:</strong> {format_authors(meta.get('authors'))}<br/>
                <strong>Domain:</strong> {meta.get('domain', 'Unknown')}<br/>
                <strong>File:</strong> {meta.get('file_name', 'Unknown')}<br/>
                <strong>Page:</strong> {meta.get('page_number', 'Unknown')}<br/>
                <strong>Chunk ID:</strong> {meta.get('chunk_id', 'Unknown')}<br/>
                <strong>Rerank Score:</strong> {item.get('score', 0.0):.4f}
            </div>
            <div class="tl-evidence-snippet">{doc.page_content[:1200]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_authors(authors):
    if authors is None:
        return "Unknown"

    if isinstance(authors, list):
        cleaned = [str(a).strip() for a in authors if str(a).strip()]
        return ", ".join(cleaned) if cleaned else "Unknown"

    if isinstance(authors, str):
        return authors.strip() if authors.strip() else "Unknown"

    return str(authors)


def build_paper_catalog(documents):
    """
    Build unique paper catalog from loaded documents.
    User can view papers, but not select them.
    """
    paper_map = {}

    for doc in documents:
        meta = doc.metadata
        doc_id = meta["doc_id"]

        if doc_id not in paper_map:
            paper_map[doc_id] = {
                "doc_id": doc_id,
                "title": meta.get("title", "Unknown"),
                "authors": format_authors(meta.get("authors")),
                "domain": meta.get("domain", "unknown"),
                "file_name": meta.get("file_name", "Unknown"),
                "metadata_source": meta.get("metadata_source", "unknown"),
            }

    return dict(sorted(paper_map.items(), key=lambda x: (x[1]["domain"], x[1]["title"])))


def parse_manual_chunk_ids(raw_text: str):
    if not raw_text:
        return set()

    normalized = raw_text.replace(",", "|").replace("\n", "|")
    parts = normalized.split("|")

    chunk_ids = set()
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("chunk_id:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        chunk_ids.add(cleaned)

    return chunk_ids


def compute_manual_retrieval_metrics(retrieved_chunk_ids, relevant_chunk_ids, k):
    if k <= 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hits": 0,
        }

    top_k = retrieved_chunk_ids[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_chunk_ids)
    precision_at_k = hits / k
    recall_at_k = hits / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hits": hits,
    }


def init_auth_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("auth_user", "")
    st.session_state.setdefault("chat_histories", {})
    st.session_state.setdefault("use_chat_context", True)


def get_app_credentials():
    username = os.getenv("TRUSTLAYER_APP_USERNAME", "admin")
    password = os.getenv("TRUSTLAYER_APP_PASSWORD", "trustlayer123")
    using_default = (
        "TRUSTLAYER_APP_USERNAME" not in os.environ or
        "TRUSTLAYER_APP_PASSWORD" not in os.environ
    )
    return username, password, using_default


def verify_login(username: str, password: str):
    expected_username, expected_password, _ = get_app_credentials()
    return (
        hmac.compare_digest(username, expected_username) and
        hmac.compare_digest(password, expected_password)
    )


def render_auth_ui():
    init_auth_state()
    configured_username, _, using_default_creds = get_app_credentials()

    if not st.session_state["authenticated"]:
        set_login_mode_styles()

    if st.session_state["authenticated"]:
        st.sidebar.success(f"Logged in as `{st.session_state['auth_user']}`")
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["auth_user"] = ""
            st.rerun()
        return True

    st.sidebar.markdown(
        """
        <div class="tl-sidebar-card">
            <div class="tl-sidebar-title">Access</div>
            <div class="tl-sidebar-label">
                Log in from the main panel to unlock the research assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([0.16, 0.68, 0.16])
    with center:
        st.markdown(
            """
            <div class="tl-login-shell">
                <div class="tl-login-card">
                    <div class="tl-login-eyebrow">Secure Workspace</div>
                    <div class="tl-login-title">Sign in to TrustLayer</div>
                    <div class="tl-login-copy">
                        Access your grounded research assistant, inspect retrieved evidence,
                        and review verification signals in one workspace designed for careful reading.
                    </div>
                    <div class="tl-login-grid">
                        <div class="tl-login-panel">
                            <strong>What you get after login</strong>
                            <ul class="tl-login-list">
                                <li>Grounded answers over your indexed papers</li>
                                <li>Verification and retrieval confidence signals</li>
                                <li>Evidence chunks and manual precision/recall review</li>
                            </ul>
                        </div>
                        <div class="tl-login-panel">
                            <span class="tl-login-badge">Private session</span>
                            <span class="tl-login-badge">Corpus-aware chat</span>
                            <span class="tl-login-badge">Evidence first</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="tl-login-form-wrap">', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if verify_login(username, password):
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            with center:
                st.error("Invalid username or password.")

    with center:
        st.caption("Enter your credentials to continue into the TrustLayer workspace.")

        if using_default_creds:
            st.warning(
                "Using default local credentials right now. Set "
                "`TRUSTLAYER_APP_USERNAME` and `TRUSTLAYER_APP_PASSWORD` in `.env` "
                "to replace them."
            )
            st.code(
                f"Username: {configured_username}\nPassword: trustlayer123",
                language="text",
            )

    return False


def get_user_chat_history():
    username = st.session_state.get("auth_user", "")
    histories = st.session_state["chat_histories"]
    histories.setdefault(username, [])
    return histories[username]


def clear_user_chat_history():
    username = st.session_state.get("auth_user", "")
    st.session_state["chat_histories"][username] = []


def build_contextual_query(query: str, chat_history, max_turns: int = 3):
    if not st.session_state.get("use_chat_context", True) or not chat_history:
        return query

    recent_turns = chat_history[-max_turns:]
    history_lines = []

    for turn in recent_turns:
        history_lines.append(f"User: {turn['query']}")
        history_lines.append(f"Assistant: {turn['result'].get('answer', '')}")

    history_block = "\n".join(history_lines)
    return (
        "Use the conversation history below to interpret the current user query, "
        "especially for follow-up references.\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"Current query: {query}"
    )


def render_verification_params(params: dict):
    if not params:
        st.info("No verification parameters available.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Retrieval Confidence",
        f"{params.get('retrieval_confidence', 0.0):.3f}"
    )
    col2.metric(
        "Reranker Confidence",
        f"{params.get('reranker_confidence', 0.0):.3f}"
    )
    col3.metric(
        "Evidence Similarity",
        f"{params.get('evidence_similarity', 0.0):.3f}"
    )
    col4.metric(
        "Grounding Score",
        f"{params.get('grounding_score', 0.0):.3f}"
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Evidence Coverage",
        f"{params.get('evidence_coverage', 0.0):.3f}"
    )
    col6.metric(
        "Combined Confidence",
        f"{params.get('combined_confidence', 0.0):.3f}"
    )
    col7.metric(
        "NLI Max Entailment",
        f"{params.get('nli_max_entailment', 0.0):.3f}"
    )
    col8.metric(
        "NLI Max Contradiction",
        f"{params.get('nli_contradiction_max', 0.0):.3f}"
    )


def render_assistant_response(result: dict, query_key: str):
    render_status_pills(result)

    answer_col, justification_col = st.columns([1.1, 1], gap="large")

    with answer_col:
        st.markdown(
            f"""
            <div class="tl-answer">
                <div class="tl-mini-title">Answer</div>
                <div>{result["answer"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with justification_col:
        st.markdown(
            f"""
            <div class="tl-justification">
                <div class="tl-mini-title">Justification</div>
                <div>{result["justification"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="tl-section-card">', unsafe_allow_html=True)
    st.subheader("Pipeline Summary")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Corrected", str(result.get("corrected", False)))
    c2.metric("Abstained", str(result.get("abstained", False)))
    c3.metric("Retries Used", str(result.get("retries_used", 0)))
    c4.metric("Verified", str(result.get("verification", {}).get("verified", False)))
    st.markdown("</div>", unsafe_allow_html=True)

    final_docs = result.get("final_docs", [])
    retrieved_chunk_ids = [
        str(item["doc"].metadata.get("chunk_id", "")).strip()
        for item in final_docs
        if str(item["doc"].metadata.get("chunk_id", "")).strip()
    ]

    overview_tab, verification_tab, evidence_tab, eval_tab, context_tab = st.tabs(
        ["Overview", "Verification", "Evidence", "Manual Eval", "Generator Context"]
    )

    with overview_tab:
        st.markdown('<div class="tl-section-card">', unsafe_allow_html=True)
        st.subheader("Used Queries")
        for used_query in result.get("used_queries", []):
            st.write(f"- {used_query}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="tl-section-card">', unsafe_allow_html=True)
        st.subheader("Evidence Snapshot")
        if not final_docs:
            st.info("No evidence retrieved.")
        else:
            top_item = final_docs[0]
            top_meta = top_item["doc"].metadata
            st.write(f"**Top Paper:** {top_meta.get('title', 'Unknown')}")
            st.write(f"**Authors:** {format_authors(top_meta.get('authors'))}")
            st.write(f"**Chunk ID:** {top_meta.get('chunk_id', 'Unknown')}")
            st.write(f"**Rerank Score:** {top_item.get('score', 0.0):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with verification_tab:
        st.markdown('<div class="tl-section-card">', unsafe_allow_html=True)
        st.subheader("Verification Parameters")
        render_verification_params(result.get("verification_params", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    with evidence_tab:
        st.subheader("Top Evidence")
        if not final_docs:
            st.info("No evidence retrieved.")
        else:
            for idx, item in enumerate(final_docs, start=1):
                doc = item["doc"]
                meta = doc.metadata

                with st.expander(f"Evidence {idx}: {meta.get('title', 'Unknown')}", expanded=(idx == 1)):
                    render_evidence_card(item, idx)

    with eval_tab:
        st.subheader("Manual Retrieval Evaluation")
        st.caption(
            "Paste the relevant ground-truth chunk IDs for this query to compute "
            "Precision@K and Recall@K for the currently displayed evidence."
        )

        st.text_area(
            "All Retrieved Chunk IDs",
            value="|".join(retrieved_chunk_ids),
            height=100,
            key=f"all_retrieved_chunk_ids_{query_key}",
        )

        max_k = max(1, len(retrieved_chunk_ids))

        with st.form(key=f"manual_eval_form_{query_key}"):
            manual_ids_text = st.text_area(
                "Relevant chunk IDs",
                placeholder="Example: 7ccca590b6e50bb2|53fbf9b4cdeae2be",
                help="Separate chunk IDs with |, commas, or new lines.",
                key=f"manual_chunk_ids_{query_key}",
            )
            selected_k = st.slider(
                "K",
                min_value=1,
                max_value=max_k,
                value=max_k,
                key=f"manual_eval_k_{query_key}",
            )
            submitted = st.form_submit_button("Check Precision/Recall")

        if submitted:
            relevant_chunk_ids = parse_manual_chunk_ids(manual_ids_text)

            if not relevant_chunk_ids:
                st.warning("No valid chunk IDs were parsed from the input.")
            elif not retrieved_chunk_ids:
                st.info("No retrieved chunk IDs are available for this query.")
            else:
                manual_metrics = compute_manual_retrieval_metrics(
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    relevant_chunk_ids=relevant_chunk_ids,
                    k=selected_k,
                )

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Precision@K", f"{manual_metrics['precision_at_k']:.3f}")
                mc2.metric("Recall@K", f"{manual_metrics['recall_at_k']:.3f}")
                mc3.metric("Relevant Hits", f"{manual_metrics['hits']}/{selected_k}")
                st.text_area(
                    "Relevant Chunk IDs",
                    value="|".join(sorted(relevant_chunk_ids)),
                    height=80,
                    key=f"relevant_chunk_ids_{query_key}",
                )

    with context_tab:
        st.subheader("Context Sent to Generator")
        st.markdown(
            f'<div class="tl-context-box">{result.get("context", "")}</div>',
            unsafe_allow_html=True,
        )


def render_sidebar_chat_history(chat_history):
    st.sidebar.markdown(
        """
        <div class="tl-sidebar-card">
            <div class="tl-sidebar-title">My Chat History</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not chat_history:
        st.sidebar.caption("No chat history yet for this user.")
        return

    for idx, turn in enumerate(reversed(chat_history), start=1):
        title = turn["query"].strip() or f"Query {idx}"
        short_title = title if len(title) <= 60 else f"{title[:57]}..."

        with st.sidebar.expander(short_title, expanded=False):
            st.write(f"**User:** {turn['query']}")
            st.write(f"**Answer:** {turn['result'].get('answer', '')}")


inject_custom_styles()


# ============================================================
# LOAD PIPELINE ONCE
# ============================================================
@st.cache_resource(show_spinner=True)
def load_pipeline():
    return prepare_pipeline(
        force_rebuild_documents=False,
        force_rebuild_chunks=False,
        force_rebuild_vectordb=False,
        use_api_enrichment=True,
        chunk_size=800,
        chunk_overlap=150,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        device="mps",   # change to "mps" if stable on your setup
    )


if not render_auth_ui():
    st.stop()

with st.spinner("Preparing TrustLayer pipeline..."):
    pipeline = load_pipeline()

documents = pipeline["documents"]
chunks = pipeline["chunks"]
vectorstore = pipeline["vectorstore"]
bm25 = pipeline["bm25"]
reranker = pipeline["reranker"]

paper_catalog = build_paper_catalog(documents)
domains = sorted({meta["domain"] for meta in paper_catalog.values()})

render_hero(
    documents_count=len(documents),
    paper_count=len(paper_catalog),
    domains=domains,
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Corpus Overview")
render_sidebar_overview(
    paper_count=len(paper_catalog),
    domains=domains,
    chat_count=len(get_user_chat_history()),
)

st.sidebar.divider()
st.sidebar.subheader("Chat Settings")
st.sidebar.checkbox(
    "Use recent chat history as context",
    key="use_chat_context",
    help="When enabled, recent turns from the current logged-in user are used to interpret follow-up questions.",
)

if st.sidebar.button("Clear My Chat History", use_container_width=True):
    clear_user_chat_history()
    st.rerun()

st.sidebar.divider()
render_sidebar_paper_browser(paper_catalog)


# ============================================================
# MAIN QUERY AREA
# ============================================================
user_history = get_user_chat_history()
render_sidebar_chat_history(user_history)

if user_history:
    latest_turn = user_history[-1]
    with st.chat_message("user"):
        st.write(latest_turn["query"])
    with st.chat_message("assistant"):
        render_assistant_response(latest_turn["result"], query_key="latest_turn")
else:
    render_empty_chat_state()

query = st.chat_input("Ask a question about the research papers...")

if query:
    contextual_query = build_contextual_query(query, user_history)

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Running corrective retrieval and grounded answer generation..."):
        result = corrective_rag_pipeline_v2(
            query=contextual_query,
            vectorstore=vectorstore,
            bm25=bm25,
            chunks=chunks,
            reranker=reranker,
            llm_generate_fn=llm_generate_fn,
            dense_k=20,
            sparse_k=20,
            fusion_k=50,
            final_k=5,
            max_retries=2,
        )

    with st.chat_message("assistant"):
        render_assistant_response(result, query_key=f"current_{len(user_history) + 1}")

    user_history.append({
        "query": query,
        "contextual_query": contextual_query,
        "result": result,
    })
