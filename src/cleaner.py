"""
Prétraitement et nettoyage de haute qualité des données scientifiques (chercheurs FSBM).

Ce module assure :
- Décodage des entités HTML (html.unescape)
- Préservation rigoureuse des caractères accentués (français / multilingue)
- Élimination des bruits typographiques, liens URLs et mentions de copyright
- Détection de la langue de l'article (Français / Anglais)
- Normalisation des années de publication et dédoublonnage des articles
- Export synchronisé aux formats JSON hiérarchique et Parquet tabulaire optimisé
- Rapport statistique d'exécution détaillé
"""

import html
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "raw_scholar_data.json"
OUTPUT_CLEAN_JSON = PROJECT_ROOT / "data" / "processed" / "processed_abstracts.json"
OUTPUT_CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "processed_abstracts.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FSBM_Cleaner")


# ============================================================
# DÉTECTION DE LANGUE (HEURISTIQUE ROBUSTE FR/EN)
# ============================================================

FRENCH_STOPWORDS: Set[str] = {
    "le", "la", "les", "des", "un", "une", "du", "de", "dans", "pour",
    "avec", "sur", "ce", "cette", "ces", "qui", "que", "nous", "sont",
    "est", "ont", "par", "plus", "comme", "dont", "mais", "notre",
    "nos", "leur", "leurs", "cette", "ainsi", "entre", "également",
    "méthode", "résultats", "données", "étude", "système", "travail"
}

ENGLISH_STOPWORDS: Set[str] = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on",
    "with", "as", "by", "at", "from", "this", "an", "be", "are", "was",
    "were", "which", "or", "have", "has", "we", "paper", "presents",
    "results", "proposed", "method", "based", "system", "data", "model"
}


def detect_language(text: str) -> str:
    """
    Identifie si le texte est majoritairement en français ('fr') ou en anglais ('en').
    """
    if not text:
        return "unknown"

    words = set(re.findall(r"\b[a-zàâäéèêëîïôöùûüç]+\b", text.lower()))
    if not words:
        return "en"

    fr_hits = len(words.intersection(FRENCH_STOPWORDS))
    en_hits = len(words.intersection(ENGLISH_STOPWORDS))

    if fr_hits > en_hits and fr_hits >= 2:
        return "fr"
    elif en_hits >= fr_hits and en_hits >= 2:
        return "en"
    elif fr_hits > 0:
        return "fr"
    return "en"


# ============================================================
# FONCTIONS DE NETTOYAGE TEXTUEL
# ============================================================

# Patterns de bruits d'éditeurs et mentions de copyright
NOISE_PATTERNS = [
    re.compile(r"copyright\s*(?:©|\(c\))\s*\d{4}.*?(?:\.|$)", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"tous droits réservés\.?", re.IGNORECASE),
    re.compile(r"elsevier\s*(?:b\.v\.|ltd\.|inc\.)?", re.IGNORECASE),
    re.compile(r"ieee\s*(?:tran|conf|proc|access)?", re.IGNORECASE),
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"doi:\s*\S+", re.IGNORECASE),
    re.compile(r"^(?:abstract|résumé)\s*[:\-]\s*", re.IGNORECASE),
]

PLACEHOLDER_ABSTRACTS = {
    "no abstract available",
    "abstract not available",
    "no abstract provided",
    "aucun résumé disponible",
    "none",
    "n/a"
}


def clean_text(text: str, keep_accents: bool = True) -> str:
    """
    Nettoie et normalise le texte de l'abstract :
    - Décode les entités HTML
    - Supprime les balises HTML
    - Supprime les mentions de copyright et URLs
    - Préserve rigoureusement les caractères accentués (é, è, à, ç, ...)
    - Normalise la ponctuation et les espaces multiples
    """
    if not text or not isinstance(text, str):
        return ""

    # Réparation des artefacts d'encodage fréquents (ex: smart quotes ou tirets devenus \ufffd)
    text = re.sub(r"(\w)\ufffd([sS]\b)", r"\1'\2", text)
    text = re.sub(r"(\d)\ufffd(\d)", r"\1-\2", text)
    text = re.sub(r"(\w)\ufffd(\w)", r"\1'\2", text)
    text = text.replace("\ufffd", " ")

    # Décodage des entités HTML (&amp; -> &, &eacute; -> é, etc.)
    text = html.unescape(text)

    # Suppression des balises HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Suppression des URLs et métadonnées parasites
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    if keep_accents:
        # Conservation des lettres latines et accentuées (Unicode Latin)
        # \u00C0-\u017F englobe les accents majuscules et minuscules français
        text = re.sub(r"[^a-zA-Z0-9\u00C0-\u017F\s.,;:!?'\"%\-\(\)/]", " ", text)
    else:
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        text = re.sub(r"[^a-zA-Z0-9\s.,;:!?'\"%\-\(\)/]", " ", text)

    # Suppression des ellipses terminales tronquées par Google Scholar ("…")
    text = re.sub(r"[\.…]+\s*$", ".", text)

    # Remplacement des espaces multiples et sauts de ligne
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_title(title: str) -> str:
    """Normalise un titre pour le dédoublonnage."""
    if not title:
        return ""
    clean = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", clean).strip()


def extract_clean_year(year_val: Any) -> str:
    """Extrait une année propre à 4 chiffres (ex: '2021')."""
    if not year_val:
        return ""
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(year_val))
    return match.group(1) if match else str(year_val).strip()


# ============================================================
# PIPELINE GLOBAL DE NETTOYAGE
# ============================================================

def run_cleaning() -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Exécute le pipeline complet de nettoyage et génère les fichiers finaux JSON et Parquet.
    """
    logger.info("=" * 60)
    logger.info("[*] Démarrage du prétraitement et nettoyage des résumés FSBM")
    logger.info(f"[*] Fichier d'entrée : {INPUT_RAW_FILE}")
    logger.info("=" * 60)

    if not INPUT_RAW_FILE.exists():
        logger.error(f"[!] Fichier source introuvable : {INPUT_RAW_FILE}")
        raise FileNotFoundError(f"Missing {INPUT_RAW_FILE}")

    with open(INPUT_RAW_FILE, "r", encoding="utf-8") as f:
        raw_profiles = json.load(f)

    logger.info(f"[*] {len(raw_profiles)} profils de chercheurs chargés.")

    flat_records: List[Dict[str, Any]] = []
    clean_profiles: List[Dict[str, Any]] = []

    seen_titles_by_author: Dict[str, Set[str]] = {}
    stats = {
        "total_raw_articles": 0,
        "skipped_too_short": 0,
        "skipped_placeholder": 0,
        "skipped_duplicate": 0,
        "kept_articles": 0,
        "languages": {"fr": 0, "en": 0, "unknown": 0}
    }

    for profile in raw_profiles:
        scholar_id = str(profile.get("chercheur_id", "")).strip()
        author_name = str(profile.get("nom_complet", "")).strip()
        affiliation = str(profile.get("affiliation", "")).strip()
        metrics = profile.get("metriques", {})

        if scholar_id not in seen_titles_by_author:
            seen_titles_by_author[scholar_id] = set()

        clean_articles_for_profile: List[Dict[str, Any]] = []

        for art in profile.get("articles", []):
            stats["total_raw_articles"] += 1
            raw_abstract = str(art.get("abstract", "")).strip()
            raw_title = str(art.get("titre", "")).strip()

            # 1. Filtre résumés absents ou trop courts (< 25 caractères)
            if not raw_abstract or len(raw_abstract) < 25:
                stats["skipped_too_short"] += 1
                continue

            # 2. Filtre résumés placeholders
            if raw_abstract.lower() in PLACEHOLDER_ABSTRACTS:
                stats["skipped_placeholder"] += 1
                continue

            # 3. Dédoublonnage au sein des articles du chercheur
            norm_title = normalize_title(raw_title)
            if norm_title and norm_title in seen_titles_by_author[scholar_id]:
                stats["skipped_duplicate"] += 1
                continue
            if norm_title:
                seen_titles_by_author[scholar_id].add(norm_title)

            # 4. Nettoyage du texte avec conservation des accents français
            clean_abs = clean_text(raw_abstract, keep_accents=True)
            if len(clean_abs) < 25:
                stats["skipped_too_short"] += 1
                continue

            # 5. Détection de langue
            lang = detect_language(clean_abs)
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

            # 6. Extraction et nettoyage des métadonnées
            clean_year = extract_clean_year(art.get("date_publication"))
            citations = int(art.get("citations", 0) or 0)
            journal = str(art.get("journal", "")).strip()
            authors_list = art.get("auteurs", [])
            if isinstance(authors_list, str):
                authors_list = [a.strip() for a in authors_list.split(",") if a.strip()]

            clean_article_dict = {
                "article_id": str(art.get("article_id", f"{scholar_id}:{len(clean_articles_for_profile)}")),
                "titre": raw_title,
                "auteurs": authors_list,
                "date_publication": clean_year,
                "journal": journal,
                "citations": citations,
                "lang": lang,
                "word_count": len(clean_abs.split()),
                "abstract_raw": raw_abstract,
                "abstract_clean": clean_abs
            }

            clean_articles_for_profile.append(clean_article_dict)

            flat_records.append({
                "chercheur_id": scholar_id,
                "nom_complet": author_name,
                "affiliation": affiliation,
                "article_id": clean_article_dict["article_id"],
                "titre": clean_article_dict["titre"],
                "auteurs_str": ", ".join(authors_list),
                "date_publication": clean_year,
                "journal": journal,
                "citations": citations,
                "lang": lang,
                "word_count": clean_article_dict["word_count"],
                "abstract_clean": clean_abs
            })

            stats["kept_articles"] += 1

        if clean_articles_for_profile:
            profile_copy = {
                "chercheur_id": scholar_id,
                "nom_complet": author_name,
                "affiliation": affiliation,
                "metriques": metrics,
                "nombre_articles_nettoyes": len(clean_articles_for_profile),
                "articles": clean_articles_for_profile
            }
            clean_profiles.append(profile_copy)

    # ============================================================
    # EXPORT DES FICHIERS
    # ============================================================

    OUTPUT_CLEAN_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sauvegarde JSON hiérarchique
    with open(OUTPUT_CLEAN_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_profiles, f, ensure_ascii=False, indent=2)

    # 2. Sauvegarde Parquet tabulaire optimisé
    df = pd.DataFrame(flat_records)
    df.to_parquet(OUTPUT_CLEAN_PARQUET, index=False, engine="pyarrow")

    # ============================================================
    # RAPPORT STATISTIQUE
    # ============================================================

    logger.info("=" * 60)
    logger.info("[+] PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS")
    logger.info(f"    - Chercheurs avec articles valides : {len(clean_profiles)}/{len(raw_profiles)}")
    logger.info(f"    - Total articles bruts : {stats['total_raw_articles']}")
    logger.info(f"    - Articles écartés (résumé manquant/<25 car.) : {stats['skipped_too_short']}")
    logger.info(f"    - Articles écartés (placeholders) : {stats['skipped_placeholder']}")
    logger.info(f"    - Doublons éliminés : {stats['skipped_duplicate']}")
    logger.info(f"    - Total articles propres sauvegardés : {stats['kept_articles']}")
    logger.info(f"    - Répartition langues : {stats['languages']}")
    logger.info(f"[+] Fichier JSON : {OUTPUT_CLEAN_JSON} ({OUTPUT_CLEAN_JSON.stat().st_size / 1024:.1f} KB)")
    logger.info(f"[+] Fichier Parquet : {OUTPUT_CLEAN_PARQUET} ({OUTPUT_CLEAN_PARQUET.stat().st_size / 1024:.1f} KB)")
    logger.info("=" * 60)

    return clean_profiles, df


if __name__ == "__main__":
    run_cleaning()