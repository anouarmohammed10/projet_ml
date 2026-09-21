"""
Génération des représentations vectorielles (Embeddings) et Indexation ChromaDB.

Modèle d'embedding requis par le projet académique :
- zeroentropy/zembed-1-embedding (Dimension : 2560)

Ce module assure :
- Chargement des résumés nettoyés depuis le fichier Parquet
- Encodage vectoriel avec SentenceTransformer (zembed-1) et invite 'document'
- Stockage persistant dans ChromaDB avec métadonnées complètes
- Vérification d'intégrité et test unitaire de recherche sémantique
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import chromadb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION DES CHEMINS & CONSTANTES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "processed_abstracts.parquet"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "vector_store"

# Paramètres par défaut alignés sur les critères d'évaluation académiques
DEFAULT_MODEL_NAME = "zeroentropy/zembed-1-embedding"
DEFAULT_COLLECTION_NAME = "research_papers_db"
DEFAULT_BATCH_SIZE = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FSBM_Indexer")


# ============================================================
# FONCTIONS DE VECTORISATION ET D'INDEXATION
# ============================================================

def vectorize_and_index(
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True
) -> None:
    """
    Encode les résumés avec le modèle zembed-1 et les indexe dans ChromaDB.
    """
    logger.info("=" * 65)
    logger.info("[*] DÉMARRAGE DU PIPELINE DE VECTORISATION ET D'INDEXATION FSBM")
    logger.info(f"[*] Modèle d'embedding cible : {model_name}")
    logger.info(f"[*] Collection ChromaDB : {collection_name}")
    logger.info("=" * 65)

    # 1. Vérification du fichier Parquet
    if not INPUT_CLEAN_PARQUET.exists():
        logger.error(f"[!] Fichier introuvable : {INPUT_CLEAN_PARQUET}")
        logger.error("[!] Veuillez exécuter src/cleaner.py au préalable.")
        return

    df = pd.read_parquet(INPUT_CLEAN_PARQUET)
    logger.info(f"[*] {len(df)} articles chargés depuis {INPUT_CLEAN_PARQUET}.")

    if df.empty:
        logger.error("[!] Aucun article disponible dans le dataset.")
        return

    # 2. Détection du matériel d'accélération (GPU CUDA ou CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[*] Dispositif de calcul utilisé : {device.upper()}")

    # 3. Chargement du modèle SentenceTransformer
    logger.info(f"[*] Chargement du modèle {model_name}...")
    start_load = time.time()
    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"[!] Échec de chargement du modèle {model_name} : {e}")
        raise e

    embedding_dimension = model.get_embedding_dimension()
    logger.info(
        f"[+] Modèle chargé en {time.time() - start_load:.1f}s. "
        f"Dimension des vecteurs : {embedding_dimension}"
    )

    # 4. Préparation des textes d'abstracts
    abstracts: List[str] = (
        df["abstract_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # 5. Génération des embeddings
    logger.info(f"[*] Encodage vectoriel de {len(abstracts)} articles (taille de batch = {batch_size})...")
    start_encode = time.time()

    # zembed-1 est configuré avec l'invite 'document' par défaut pour l'indexation
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True  # Normalisation L2 pour recherche cosinus
    }

    if "document" in getattr(model, "prompts", {}):
        encode_kwargs["prompt_name"] = "document"

    embeddings = model.encode(abstracts, **encode_kwargs)
    embeddings_list = embeddings.tolist()
    logger.info(f"[+] {len(embeddings_list)} embeddings générés en {time.time() - start_encode:.1f}s.")

    # 6. Connexion à ChromaDB
    logger.info(f"[*] Connexion à la base vectorielle ChromaDB : {CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if recreate:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"[*] Ancienne collection '{collection_name}' supprimée pour réindexation propre.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # 7. Préparation des métadonnées et identifiants
    ids: List[str] = []
    metadatas: List[dict] = []
    documents: List[str] = abstracts

    for index, row in df.iterrows():
        raw_art_id = str(row.get("article_id", f"art_{index}")).strip()
        ids.append(raw_art_id if raw_art_id else f"art_{index}")

        metadatas.append({
            "chercheur_id": str(row.get("chercheur_id", "")),
            "nom_complet": str(row.get("nom_complet", "")),
            "affiliation": str(row.get("affiliation", ""))[:300],
            "titre": str(row.get("titre", ""))[:300],
            "date_publication": str(row.get("date_publication", "")),
            "journal": str(row.get("journal", ""))[:200],
            "citations": int(row.get("citations", 0) or 0),
            "lang": str(row.get("lang", "en")),
            "word_count": int(row.get("word_count", 0) or 0)
        })

    # 8. Insertion dans ChromaDB par tranches (chunks)
    logger.info(f"[*] Insertion des vecteurs et métadonnées dans ChromaDB...")
    chunk_size = 100
    for i in range(0, len(ids), chunk_size):
        end_idx = min(i + chunk_size, len(ids))
        collection.upsert(
            ids=ids[i:end_idx],
            embeddings=embeddings_list[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        logger.info(f"    Indexés {end_idx}/{len(ids)} articles...")

    doc_count = collection.count()

    logger.info("=" * 65)
    logger.info("[+] INDEXATION VECTORIELLE TERMINÉE AVEC SUCCÈS")
    logger.info(f"    - Collection : {collection_name}")
    logger.info(f"    - Modèle : {model_name}")
    logger.info(f"    - Dimension : {embedding_dimension}")
    logger.info(f"    - Documents indexés : {doc_count}")
    logger.info(f"    - Emplacement : {CHROMA_PERSIST_DIR}")
    logger.info("=" * 65)

    # 9. Test unitaire de vérification de la recherche sémantique
    logger.info("[*] Vérification de bon fonctionnement avec une requête test...")
    test_query = "deep learning neural network classification"
    query_kwargs = {"normalize_embeddings": True}
    if "query" in getattr(model, "prompts", {}):
        query_kwargs["prompt_name"] = "query"

    query_emb = model.encode(test_query, **query_kwargs).tolist()
    sample_results = collection.query(
        query_embeddings=[query_emb],
        n_results=2
    )

    if sample_results and sample_results["metadatas"]:
        top_meta = sample_results["metadatas"][0]
        top_dist = sample_results["distances"][0] if "distances" in sample_results else [0, 0]
        logger.info("[+] Résultat du test unitaire :")
        for rank, (meta, dist) in enumerate(zip(top_meta, top_dist), start=1):
            logger.info(
                f"    {rank}. [{meta.get('nom_complet')}] \"{meta.get('titre')[:70]}...\" "
                f"(Distance cosinus: {dist:.4f})"
            )
    logger.info("[+] Validation de la recherche sémantique réussie.")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorisation et Indexation ChromaDB FSBM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Nom ou chemin du modèle HuggingFace")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="Nom de la collection ChromaDB")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Taille de batch pour l'encodage")
    parser.add_argument("--no-recreate", action="store_true", help="Ne pas recréer la collection si elle existe")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vectorize_and_index(
        model_name=args.model,
        collection_name=args.collection,
        batch_size=args.batch_size,
        recreate=not args.no_recreate
    )