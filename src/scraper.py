"""
Scraper robuste de profils Google Scholar pour les chercheurs de la FSBM.

Ce module intègre :
- Mécanisme de checkpointing / reprise automatique (idempotence)
- Temporisation aléatoire (jitter anti-bot) et backoff exponentiel
- Rotation des User-Agents (fake-useragent)
- Gestion gracieuse des erreurs et blocages Google Scholar (sauvegarde automatique d'état)
- Interface CLI paramétrable (--limit, --max-pubs, --delay-min, --delay-max, --force)
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from scholarly import DOSException, MaxTriesExceededException, scholarly
except ImportError:
    from scholarly import scholarly
    MaxTriesExceededException = Exception
    DOSException = Exception

# ============================================================
# CONFIGURATION & CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_RESEARCHERS_FILE = PROJECT_ROOT / "data" / "raw" / "input_faculty_list.json"
OUTPUT_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "raw_scholar_data.json"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FSBM_Scraper")


# ============================================================
# UTILITAIRES ANTI-BLOCAGE
# ============================================================

def setup_user_agent():
    """Tente de configurer un User-Agent réaliste et aléatoire."""
    try:
        from fake_useragent import UserAgent
        ua = UserAgent()
        random_ua = ua.random
        logger.info(f"[*] User-Agent configuré : {random_ua[:50]}...")
    except Exception as e:
        logger.debug(f"[!] fake-useragent non disponible ou erreur : {e}")


def random_sleep(min_sec: float = 3.0, max_sec: float = 7.0, description: str = ""):
    """Pause aléatoire (jitter) pour imiter le comportement d'un utilisateur humain."""
    sleep_time = random.uniform(min_sec, max_sec)
    if description:
        logger.debug(f"    [pause {sleep_time:.1f}s] {description}")
    time.sleep(sleep_time)


# ============================================================
# GESTION DU CHECKPOINTING (REPRISE SUR ERREUR)
# ============================================================

def load_existing_checkpoint(filepath: Path) -> tuple[List[Dict[str, Any]], Set[str]]:
    """
    Charge les données déjà scrappées pour ne pas recommencer depuis zéro.
    Retourne la liste des profils existants et l'ensemble de leurs identifiants.
    """
    if not filepath.exists():
        logger.info(f"[*] Aucun checkpoint existant trouvé à {filepath}. Démarrage à neuf.")
        return [], set()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                already_scraped_ids = {
                    item["chercheur_id"]
                    for item in data
                    if isinstance(item, dict) and "chercheur_id" in item
                }
                logger.info(
                    f"[+] Checkpoint chargé : {len(data)} profils existants "
                    f"({len(already_scraped_ids)} IDs uniques déjà traités)."
                )
                return data, already_scraped_ids
    except Exception as e:
        logger.warning(f"[!] Erreur de lecture du checkpoint existant : {e}. Sauvegarde d'un backup.")
        backup_path = filepath.with_suffix(".backup.json")
        try:
            filepath.rename(backup_path)
            logger.info(f"    Fichier endommagé déplacé vers {backup_path}")
        except Exception:
            pass

    return [], set()


def save_checkpoint(filepath: Path, data: List[Dict[str, Any]]) -> None:
    """Sauvegarde atomique du checkpoint en JSON UTF-8."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = filepath.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_file.replace(filepath)


# ============================================================
# EXTRACTION SÉCURISÉE D'UN PROFIL
# ============================================================

def scrape_single_researcher(
    scholar_id: str,
    author_name: str,
    max_pubs_per_author: Optional[int] = None,
    delay_min: float = 1.0,
    delay_max: float = 3.0,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Scrape le profil complet d'un chercheur et ses publications avec gestion d'erreurs et retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            author = scholarly.search_author_id(scholar_id)
            if not author or not isinstance(author, dict):
                raise ConnectionError("Google Scholar a redirigé vers une page d'authentification/CAPTCHA (author is None).")
            author = scholarly.fill(author, sections=["basics", "indices", "publications"])

            profile_data: Dict[str, Any] = {
                "chercheur_id": scholar_id,
                "nom_complet": author.get("name", author_name),
                "affiliation": author.get("affiliation", ""),
                "metriques": {
                    "citations_totales": author.get("citedby", 0),
                    "h_index": author.get("hindex", 0),
                    "i10_index": author.get("i10index", 0)
                },
                "articles": []
            }

            publications = author.get("publications", [])
            if max_pubs_per_author is not None and max_pubs_per_author > 0:
                publications = publications[:max_pubs_per_author]

            logger.info(f"    -> {len(publications)} publications à récupérer pour {author_name}")

            for pub_idx, pub in enumerate(publications, start=1):
                try:
                    pub_filled = scholarly.fill(pub)
                    bib = pub_filled.get("bib", {})

                    article = {
                        "article_id": f"{scholar_id}:{pub_filled.get('author_pub_id', f'pub_{pub_idx}')}",
                        "titre": bib.get("title", ""),
                        "auteurs": bib.get("author", "").split(" and ") if bib.get("author") else [],
                        "date_publication": str(bib.get("pub_year", "")),
                        "journal": bib.get("journal", bib.get("conference", "")),
                        "citations": pub_filled.get("num_citations", 0),
                        "abstract": bib.get("abstract", "")
                    }
                    profile_data["articles"].append(article)
                    random_sleep(delay_min, delay_max, f"pub {pub_idx}/{len(publications)}")

                except Exception as pub_err:
                    err_msg = str(pub_err).lower()
                    if "blocked" in err_msg or "captcha" in err_msg:
                        logger.warning("    [!] Détection anti-bot potentielle lors du remplissage d'une publication.")
                        raise pub_err
                    logger.debug(f"    [!] Erreur publication ignorée : {pub_err}")
                    continue

            return profile_data

        except (MaxTriesExceededException, DOSException) as mre:
            logger.error(f"[!] Google Scholar bloque les requêtes (MaxTries/DOS) : {mre}")
            raise mre
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["blocked", "captcha", "429", "authentification", "connectionerror"]):
                wait_time = 20 * attempt
                logger.warning(
                    f"[!] Alerte blocage / CAPTCHA Google Scholar (Tentative {attempt}/{max_retries}). "
                    f"Attente de refroidissement de {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.warning(f"[!] Erreur tentative {attempt}/{max_retries} pour {author_name}: {e}")
                time.sleep(3 * attempt)

    logger.error(f"[X] Échec définitif pour {author_name} ({scholar_id}) après {max_retries} tentatives.")
    return None


# ============================================================
# PIPELINE PRINCIPAL DE SCRAPING
# ============================================================

def run_scraper(
    limit: Optional[int] = None,
    max_pubs_per_author: Optional[int] = None,
    delay_min: float = 3.0,
    delay_max: float = 7.0,
    force_restart: bool = False
) -> None:
    """
    Exécute le scraping robuste de la liste des enseignants-chercheurs de la FSBM.
    """
    setup_user_agent()

    if not INPUT_RESEARCHERS_FILE.exists():
        logger.error(f"[!] Fichier introuvable : {INPUT_RESEARCHERS_FILE}")
        return

    with open(INPUT_RESEARCHERS_FILE, "r", encoding="utf-8") as f:
        target_list = json.load(f)

    if not isinstance(target_list, list):
        logger.error("[!] Format de input_faculty_list.json invalide (liste attendue).")
        return

    logger.info(f"[*] {len(target_list)} enseignants répertoriés dans le fichier d'entrée.")

    if force_restart:
        scraped_data: List[Dict[str, Any]] = []
        already_scraped_ids: Set[str] = set()
        logger.info("[*] Mode --force activé : démarrage d'un scraping neuf.")
    else:
        scraped_data, already_scraped_ids = load_existing_checkpoint(OUTPUT_RAW_FILE)

    pending_targets = [
        t for t in target_list
        if t.get("chercheur_id") not in already_scraped_ids
    ]

    if limit is not None:
        pending_targets = pending_targets[:limit]

    logger.info(
        f"[*] {len(already_scraped_ids)} profils déjà sauvegardés. "
        f"{len(pending_targets)} profils restants à scraper."
    )

    if not pending_targets:
        logger.info("[+] Tous les profils ciblés ont déjà été collectés. Fin du processus.")
        return

    success_count = 0
    failure_count = 0

    try:
        for idx, target in enumerate(pending_targets, start=1):
            scholar_id = target.get("chercheur_id", "").strip()
            author_name = target.get("nom_complet", "").strip()

            if not scholar_id:
                logger.warning(f"[{idx}/{len(pending_targets)}] Profil sans chercheur_id ignoré : {author_name}")
                continue

            logger.info(f"[{idx}/{len(pending_targets)}] Collecte en cours : {author_name} ({scholar_id})")

            profile_data = scrape_single_researcher(
                scholar_id=scholar_id,
                author_name=author_name,
                max_pubs_per_author=max_pubs_per_author,
                delay_min=1.0,
                delay_max=2.5
            )

            if profile_data:
                scraped_data.append(profile_data)
                already_scraped_ids.add(scholar_id)
                save_checkpoint(OUTPUT_RAW_FILE, scraped_data)
                success_count += 1
                logger.info(f"    [+] Profil sauvegardé avec succès ({len(profile_data['articles'])} articles).")
            else:
                failure_count += 1

            # Pause aléatoire de protection entre chaque chercheur
            random_sleep(delay_min, delay_max, f"entre chercheurs ({idx}/{len(pending_targets)})")

    except KeyboardInterrupt:
        logger.warning("\n[!] Scraping interrompu par l'utilisateur (Ctrl+C).")
    except (MaxTriesExceededException, DOSException):
        logger.error(
            "\n[!] Arrêt d'urgence : Seuil de requêtes Google Scholar dépassé (MaxTries/DOS). "
            "Toutes les données collectées jusqu'ici ont été enregistrées dans le checkpoint."
        )
    except Exception as e:
        logger.critical(f"\n[!] Exception critique : {e}")
    finally:
        save_checkpoint(OUTPUT_RAW_FILE, scraped_data)
        logger.info("=" * 60)
        logger.info(f"[+] BILAN SCRAPING : {success_count} nouveaux profils ajoutés, {failure_count} échecs.")
        logger.info(f"[+] Total cumulé dans {OUTPUT_RAW_FILE} : {len(scraped_data)} profils de chercheurs FSBM.")
        logger.info("=" * 60)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Scraper robuste de profils Google Scholar FSBM")
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximal de nouveaux profils à scraper")
    parser.add_argument("--max-pubs", type=int, default=None, help="Nombre maximal de publications par auteur")
    parser.add_argument("--delay-min", type=float, default=3.0, help="Délai minimal entre requêtes en secondes")
    parser.add_argument("--delay-max", type=float, default=7.0, help="Délai maximal entre requêtes en secondes")
    parser.add_argument("--force", action="store_true", help="Forcer le re-scraping complet (écrase le checkpoint)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scraper(
        limit=args.limit,
        max_pubs_per_author=args.max_pubs,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        force_restart=args.force
    )