from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "vector_store"


# IMPORTANT:
# This must be the collection created with Zembed-1.
COLLECTION_NAME = "research_papers_db"


# IMPORTANT:
# This must be exactly the same model used during indexing.
MODEL_NAME = "zeroentropy/zembed-1-embedding"


# ============================================================
# RESPONSE MODELS
# ============================================================

class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the Zembed-1 embedding model and ChromaDB collection
    once when the API starts.
    """

    print("=" * 60)
    print("[*] Starting FSBM Semantic Search API")
    print("=" * 60)

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(f"[*] Loading embedding model: {MODEL_NAME}")
    print("[*] This may take some time if Zembed-1 is not cached...")

    try:
        model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True
        )

    except Exception as exc:
        print(f"[!] Failed to load embedding model: {exc}")

        raise RuntimeError(
            f"Could not load embedding model: {MODEL_NAME}"
        ) from exc

    model_dimension = model.get_embedding_dimension()

    print("[+] Embedding model loaded successfully")
    print(f"[+] Model: {MODEL_NAME}")
    print(f"[+] Embedding dimension: {model_dimension}")

    # --------------------------------------------------------
    # Check ChromaDB directory
    # --------------------------------------------------------

    if not CHROMA_PERSIST_DIR.exists():
        raise RuntimeError(
            f"ChromaDB directory does not exist: "
            f"{CHROMA_PERSIST_DIR}"
        )

    print(
        f"[*] ChromaDB directory: "
        f"{CHROMA_PERSIST_DIR}"
    )

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR)
        )

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as exc:
        print(
            f"[!] Failed to load ChromaDB collection: "
            f"{exc}"
        )

        raise RuntimeError(
            f"Could not load ChromaDB collection "
            f"'{COLLECTION_NAME}'."
        ) from exc

    document_count = collection.count()

    print(
        f"[+] ChromaDB collection loaded: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"[+] Documents indexed: "
        f"{document_count}"
    )

    if document_count == 0:
        raise RuntimeError(
            f"ChromaDB collection "
            f"'{COLLECTION_NAME}' is empty."
        )

    # --------------------------------------------------------
    # Verify embedding dimensions
    # --------------------------------------------------------

    try:
        sample = collection.peek(limit=1)
        embeddings = sample.get("embeddings")

        if embeddings is not None and len(embeddings) > 0:

            chroma_dimension = len(embeddings[0])

            print(
                f"[+] ChromaDB embedding dimension: "
                f"{chroma_dimension}"
            )

            if chroma_dimension != model_dimension:
                raise RuntimeError(
                    "Embedding dimension mismatch: "
                    f"ChromaDB={chroma_dimension}, "
                    f"Model={model_dimension}"
                )

            print(
                "[+] Embedding dimensions match"
            )

    except RuntimeError:
        raise

    except Exception as exc:
        print(
            "[!] Warning: Could not verify "
            f"embedding dimension: {exc}"
        )

    # --------------------------------------------------------
    # Store resources in FastAPI application state
    # --------------------------------------------------------

    app.state.model = model
    app.state.model_name = MODEL_NAME
    app.state.collection = collection
    app.state.model_dimension = model_dimension

    print("=" * 60)
    print("[+] API startup complete")
    print("=" * 60)

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    print("[*] Shutting down API...")

    if hasattr(app.state, "model"):
        del app.state.model

    if hasattr(app.state, "collection"):
        del app.state.collection


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FSBM NLP & Semantic Search API",
    description=(
        "Semantic search API over FSBM research papers "
        "using Zembed-1 and ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "FSBM Semantic Search API is running"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """
    Return API and vector database status.
    """

    collection = request.app.state.collection

    return {
        "status": "healthy",
        "model": request.app.state.model_name,
        "embedding_dimension": request.app.state.model_dimension,
        "collection": COLLECTION_NAME,
        "documents": collection.count(),
    }


# ============================================================
# SEMANTIC SEARCH ENDPOINT
# ============================================================

@app.get(
    "/search",
    response_model=SearchResponse
)
def search(
    request: Request,
    query: str = Query(
        ...,
        min_length=1,
        description="Natural language text to search for."
    ),
    top_k: int = Query(
        5,
        ge=1,
        le=100,
        description="Maximum number of results to return."
    ),
) -> SearchResponse:

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    normalized_query = query.strip()

    if not normalized_query:
        raise HTTPException(
            status_code=422,
            detail="query must not be blank"
        )

    # --------------------------------------------------------
    # Get model and collection
    # --------------------------------------------------------

    try:
        model: SentenceTransformer = request.app.state.model
        collection = request.app.state.collection

        document_count = collection.count()

        if document_count == 0:
            raise HTTPException(
                status_code=503,
                detail="The vector collection is empty."
            )

        # Never request more results than available documents.
        effective_top_k = min(
            top_k,
            document_count
        )

        # ----------------------------------------------------
        # Convert query to embedding
        # ----------------------------------------------------

        print(
            f"[*] Searching for: {normalized_query}"
        )

        query_embedding = model.encode(
            [normalized_query],
            convert_to_numpy=True
        ).tolist()

        # ----------------------------------------------------
        # Verify query dimension
        # ----------------------------------------------------

        query_dimension = len(query_embedding[0])

        if query_dimension != request.app.state.model_dimension:
            raise RuntimeError(
                "Query embedding dimension mismatch: "
                f"Query={query_dimension}, "
                f"Model={request.app.state.model_dimension}"
            )

        # ----------------------------------------------------
        # Semantic search in ChromaDB
        # ----------------------------------------------------

        response = collection.query(
            query_embeddings=query_embedding,
            n_results=effective_top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except HTTPException:
        raise

    except Exception as exc:

        print("=" * 60)
        print("[!] SEARCH ERROR")
        print(f"[!] Query: {normalized_query}")
        print(f"[!] Error type: {type(exc).__name__}")
        print(f"[!] Error: {exc}")
        print("=" * 60)

        raise HTTPException(
            status_code=503,
            detail=(
                "The semantic search service is "
                "temporarily unavailable."
            ),
        ) from exc

    # --------------------------------------------------------
    # Extract ChromaDB results
    # --------------------------------------------------------

    documents = response.get("documents")
    metadatas = response.get("metadatas")
    distances = response.get("distances")

    document_rows = (
        documents[0]
        if documents and len(documents) > 0
        else []
    )

    metadata_rows = (
        metadatas[0]
        if metadatas and len(metadatas) > 0
        else []
    )

    distance_rows = (
        distances[0]
        if distances and len(distances) > 0
        else []
    )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    results = []

    result_count = max(
        len(document_rows),
        len(metadata_rows),
        len(distance_rows),
    )

    for index in range(result_count):

        document = (
            document_rows[index]
            if index < len(document_rows)
            else None
        )

        metadata = (
            metadata_rows[index] or {}
            if index < len(metadata_rows)
            else {}
        )

        distance = (
            distance_rows[index]
            if index < len(distance_rows)
            else None
        )

        results.append(
            SearchResult(
                document=document,
                metadata=metadata,
                distance=distance,
            )
        )

    return SearchResponse(
        query=normalized_query,
        results=results,
    )


# ============================================================
# LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )