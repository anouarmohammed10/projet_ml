# FSBM NLP & Semantic Search Engine - Baseline Starter

This folder contains the core baseline code to run the pipeline from raw Scholar profiles to a searchable vector store.

### 1. Environment Setup
Create and activate an isolated Python environment:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```


### 2. Execution Order

*   **`python src/scraper.py`**: Reads `data/raw/input_faculty_list.json` and outputs `data/raw/raw_scholar_data.json`.
*   **`python src/cleaner.py`**: Cleans HTML/punctuation, removes missing abstracts, and outputs cleaned `.parquet` and `.json` files in `data/processed/`.
*   **`python src/generate_vectors_and_index.py`**: Encodes the abstracts with `zeroentropy/zembed-1-embedding` and creates the local ChromaDB store in `data/vector_store/`.

### 3. Recommended Upgrades to Implement (Individual Work)

To ensure code uniqueness and avoid identical submissions, each team member should consider customizing and improving their own repository:

*   **Idempotency & Checkpointing:** Modify `scraper.py` and `cleaner.py` to check what has already been downloaded or processed before running, so crashes don't force a full re-run from scratch.
*   **Anti-Scraping & Proxy Rotation:** Google Scholar flags high-volume traffic. Add proxy rotation, randomized user-agents, or dynamic backoff (`random.uniform(5, 12)`) instead of static sleep calls.
*   **Data Augmentation / Advanced Cleaning:** Extract or normalize conference acronyms, standardize author lists, and handle language detection (some abstracts may be in French, others in English).
*   **Serving Layer:** Wrap the ChromaDB search function inside a FastAPI endpoint (`/search`) or build a Streamlit / Next.js interface for presentation day.