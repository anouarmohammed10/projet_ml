import requests
import streamlit as st

from textwrap import dedent


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FSBM Semantic Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML HELPER
# ------------------------------------------------------------
# IMPORTANT: textwrap.dedent() only strips whitespace that is
# common to EVERY line. If the first line of the block starts
# at column 0 (e.g. "<div class=...>") while nested lines are
# indented, dedent() cannot remove that inner indentation.
# Markdown then treats any line indented by 4+ spaces as a
# CODE BLOCK instead of rendering it as HTML — which is exactly
# the bug in the screenshot (raw HTML shown as text).
#
# Fix: strip leading/trailing whitespace on EVERY line before
# handing the string to st.markdown().
# ============================================================

def render_html(html: str) -> None:
    """Render a multi-line HTML string safely, regardless of
    how it was indented in the Python source."""
    cleaned_lines = [line.strip() for line in dedent(html).strip("\n").splitlines()]
    st.markdown("\n".join(cleaned_lines), unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {
        background: #f5f7fb;
    }

    /* =====================================================
       HEADER
    ===================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e3a8a 55%,
            #2563eb 100%
        );

        padding: 38px 45px;
        border-radius: 22px;
        margin-bottom: 30px;

        box-shadow:
            0 12px 35px rgba(15, 23, 42, 0.18);
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 17px;
        line-height: 1.6;
    }

    /* =====================================================
       SEARCH
    ===================================================== */

    .search-label {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 8px;
    }

    /* =====================================================
       RESULT CARD
    ===================================================== */

    .result-card {
        background: white;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;

        border: 1px solid #e5e7eb;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.06);

        transition: all 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);

        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.10);
    }

    .result-number {
        display: inline-block;

        background: #dbeafe;
        color: #1d4ed8;

        font-size: 13px;
        font-weight: 700;

        padding: 5px 10px;
        border-radius: 20px;

        margin-bottom: 10px;
    }

    .result-title {
        color: #111827;
        font-size: 21px;
        font-weight: 750;
        line-height: 1.4;

        margin-bottom: 10px;
    }

    .result-author {
        color: #4b5563;
        font-size: 14px;
        margin-bottom: 14px;
    }

    .result-abstract {
        color: #374151;
        font-size: 14px;
        line-height: 1.7;
    }

    .metadata {
        margin-top: 18px;
        padding-top: 15px;

        border-top: 1px solid #eef2f7;
    }

    .badge {
        display: inline-block;

        background: #f3f4f6;
        color: #374151;

        padding: 6px 11px;
        border-radius: 20px;

        font-size: 12px;
        font-weight: 600;

        margin-right: 6px;
        margin-bottom: 5px;
    }

    .similarity {
        background: #ecfdf5;
        color: #047857;
    }

    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* =====================================================
       METRICS
    ===================================================== */

    .metric-card {
        background: white;

        padding: 20px;

        border-radius: 16px;

        border: 1px solid #e5e7eb;

        text-align: center;

        box-shadow:
            0 4px 15px rgba(15, 23, 42, 0.05);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #1d4ed8;
    }

    .metric-label {
        font-size: 13px;
        color: #6b7280;
        margin-top: 5px;
    }

    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        text-align: center;
        color: #9ca3af;

        margin-top: 50px;
        padding: 25px;

        font-size: 13px;
    }

    </style>
    """
)


# ============================================================
# API FUNCTIONS
# ============================================================

@st.cache_data(ttl=30)
def get_health():
    """Check FastAPI health."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except Exception:
        return None


def search_papers(query: str, top_k: int):
    """Call FastAPI semantic search endpoint."""

    response = requests.get(
        f"{API_URL}/search",
        params={
            "query": query,
            "top_k": top_k,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # Logo / Brand
    # --------------------------------------------------------

    render_html(
        """
        <div style="text-align:center; padding:20px 5px 30px 5px;">
            <div style="font-size:50px;">🔬</div>
            <div style="font-size:22px; font-weight:800;">FSBM NLP</div>
            <div style="font-size:13px; opacity:0.7;">Semantic Search Engine</div>
        </div>
        """
    )

    # --------------------------------------------------------
    # Search settings
    # --------------------------------------------------------

    st.markdown("### ⚙️ Search Settings")

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # API status
    # --------------------------------------------------------

    st.markdown("### 📡 API Status")

    health = get_health()

    if health:

        st.success("API Online")

        st.caption(
            f"Model: {health.get('model', 'N/A')}"
        )

        st.caption(
            f"Documents: {health.get('documents', 0)}"
        )

        st.caption(
            f"Dimension: "
            f"{health.get('embedding_dimension', 'N/A')}"
        )

    else:

        st.error("API Offline")

        st.caption(
            "Start FastAPI first."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Technologies
    # --------------------------------------------------------

    render_html(
        """
        <div style="font-size:14px; line-height:2;">
            <strong>🛠️ Technology</strong><br>
            🧠 Sentence Transformers<br>
            🗄️ ChromaDB<br>
            ⚡ FastAPI<br>
            🎨 Streamlit
        </div>
        """
    )


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-title">🔬 FSBM Semantic Search</div>
        <div class="hero-subtitle">
            Explore research publications using
            Artificial Intelligence and semantic search.
            Search by meaning, not only by exact keywords.
        </div>
    </div>
    """
)


# ============================================================
# SEARCH SECTION
# ============================================================

render_html('<div class="search-label">🔎 What are you looking for?</div>')


query = st.text_input(
    "Search",
    placeholder=(
        "Example: machine learning for breast cancer prediction..."
    ),
    label_visibility="collapsed",
)


# ============================================================
# SEARCH BUTTONS
# ============================================================

col1, col2 = st.columns([5, 1])

with col1:

    search_button = st.button(
        "🔍 Search Research Papers",
        use_container_width=True,
        type="primary",
    )

with col2:

    clear_button = st.button(
        "Clear",
        use_container_width=True,
    )


# ============================================================
# CLEAR
# ============================================================

if clear_button:

    st.session_state.pop("results", None)
    st.session_state.pop("last_query", None)

    st.rerun()


# ============================================================
# SEARCH EXECUTION
# ============================================================

if search_button:

    if not query.strip():

        st.warning(
            "Please enter a search query."
        )

    else:

        with st.spinner(
            "🧠 Searching semantically..."
        ):

            try:

                data = search_papers(
                    query.strip(),
                    top_k
                )

                results = data.get(
                    "results",
                    []
                )

                st.session_state["results"] = results

                st.session_state["last_query"] = query.strip()

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI. "
                    "Make sure the API is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The search request timed out. "
                    "Please try again."
                )

            except requests.exceptions.HTTPError as exc:

                st.error(
                    f"❌ FastAPI returned an error: {exc}"
                )

            except Exception as exc:

                st.error(
                    f"❌ Search failed: {exc}"
                )


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    last_query = st.session_state.get(
        "last_query",
        ""
    )

    st.markdown("---")

    if results:

        # ----------------------------------------------------
        # Results title
        # ----------------------------------------------------

        st.markdown(
            f"### 📚 Search Results\n\n"
            f"**{len(results)}** relevant papers for **\"{last_query}\"**"
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        # Number of results

        with col1:

            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Results</div>
                </div>
                """
            )

        # Best distance

        with col2:

            valid_distances = [
                r.get("distance")
                for r in results
                if r.get("distance") is not None
            ]

            best_distance = min(
                valid_distances,
                default=0
            )

            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{best_distance:.3f}</div>
                    <div class="metric-label">Best Distance</div>
                </div>
                """
            )

        # Unique researchers

        with col3:

            researcher_ids = set()

            for result in results:

                metadata = result.get(
                    "metadata",
                    {}
                )

                researcher_id = metadata.get(
                    "chercheur_id"
                )

                if researcher_id:
                    researcher_ids.add(
                        researcher_id
                    )

            unique_researchers = len(
                researcher_ids
            )

            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_researchers}</div>
                    <div class="metric-label">Researchers</div>
                </div>
                """
            )

        st.markdown("")

        # ----------------------------------------------------
        # Result cards
        # ----------------------------------------------------

        for index, result in enumerate(
            results,
            start=1
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            title = metadata.get(
                "titre",
                "Untitled paper"
            )

            author = metadata.get(
                "nom_complet",
                "Unknown researcher"
            )

            year = metadata.get(
                "date_publication",
                "N/A"
            )

            citations = metadata.get(
                "citations",
                0
            )

            researcher_id = metadata.get(
                "chercheur_id",
                "N/A"
            )

            abstract = result.get(
                "document",
                ""
            )

            distance = result.get(
                "distance"
            )

            # ------------------------------------------------
            # Similarity indicator
            # ------------------------------------------------

            if distance is not None:

                similarity = max(
                    0,
                    min(
                        100,
                        (1 - distance) * 100
                    )
                )

            else:

                similarity = 0

            # ------------------------------------------------
            # Shorten abstract
            # ------------------------------------------------

            if len(abstract) > 650:

                abstract_display = (
                    abstract[:650]
                    + "..."
                )

            else:

                abstract_display = abstract

            # ------------------------------------------------
            # Result card
            # ------------------------------------------------

            render_html(
                f"""
                <div class="result-card">
                    <div class="result-number">RESULT #{index}</div>
                    <div class="result-title">{title}</div>
                    <div class="result-author">👨‍🔬 <strong>{author}</strong></div>
                    <div class="result-abstract">{abstract_display}</div>
                    <div class="metadata">
                        <span class="badge">📅 {year}</span>
                        <span class="badge">📚 {citations} citations</span>
                        <span class="badge">🆔 {researcher_id}</span>
                        <span class="badge similarity">🎯 Similarity: {similarity:.1f}%</span>
                    </div>
                </div>
                """
            )

    else:

        st.info(
            "🔎 No research papers found for this query."
        )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        <strong>FSBM NLP & Semantic Search Engine</strong><br>
        Powered by Sentence Transformers · ChromaDB · FastAPI · Streamlit
    </div>
    """
)