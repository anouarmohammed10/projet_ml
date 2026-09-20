import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "raw_scholar_data.json"
OUTPUT_CLEAN_JSON = PROJECT_ROOT / "data" / "processed" / "processed_abstracts.json"
OUTPUT_CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "processed_abstracts.parquet"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_cleaning():
    print("[*] Starting data cleaning process...")

    with open(INPUT_RAW_FILE, "r", encoding="utf-8") as f:
        raw_profiles = json.load(f)

    flat_records = []
    clean_profiles = []

    for profile in raw_profiles:
        clean_articles = []
        for art in profile.get("articles", []):
            raw_abstract = art.get("abstract", "")

            if not raw_abstract or len(raw_abstract.strip()) < 20:
                continue

            cleaned_abstract = clean_text(raw_abstract)
            
            clean_article = {
                "article_id": art.get("article_id"),
                "titre": art.get("titre"),
                "auteurs": art.get("auteurs"),
                "date_publication": art.get("date_publication"),
                "journal": art.get("journal"),
                "citations": art.get("citations", 0),
                "abstract": raw_abstract,
                "abstract_clean": cleaned_abstract
            }
            clean_articles.append(clean_article)

            flat_records.append({
                "chercheur_id": profile.get("chercheur_id"),
                "nom_complet": profile.get("nom_complet"),
                "affiliation": profile.get("affiliation"),
                "article_id": clean_article["article_id"],
                "titre": clean_article["titre"],
                "date_publication": clean_article["date_publication"],
                "citations": clean_article["citations"],
                "abstract_clean": clean_article["abstract_clean"]
            })

        if clean_articles:
            profile_copy = dict(profile)
            profile_copy["articles"] = clean_articles
            clean_profiles.append(profile_copy)

    with open(OUTPUT_CLEAN_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_profiles, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(flat_records)
    df.to_parquet(OUTPUT_CLEAN_PARQUET, index=False)

    print(f"[+] Cleaned profiles saved: {len(clean_profiles)}")
    print(f"[+] Total clean articles saved: {len(df)}")
    print(f"[+] Outputs: {OUTPUT_CLEAN_JSON} and {OUTPUT_CLEAN_PARQUET}")

if __name__ == "__main__":
    run_cleaning()