import json
import time
from pathlib import Path

from scholarly import scholarly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_RESEARCHERS_FILE = PROJECT_ROOT / "data" / "raw" / "input_faculty_list.json"
OUTPUT_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "raw_scholar_data.json"

def scrape_profiles():
    with open(INPUT_RESEARCHERS_FILE, "r", encoding="utf-8") as f:
        target_list = json.load(f)

    scraped_data = []
    print(f"[*] Starting scrape for {len(target_list)} profiles...")

    for index, target in enumerate(target_list, start=1):
        scholar_id = target.get("chercheur_id")
        author_name = target.get("nom_complet")

        print(f"[{index}/{len(target_list)}] Scraping: {author_name} ({scholar_id})")

        try:
            author = scholarly.search_author_id(scholar_id)
            author = scholarly.fill(author, sections=["basics", "indices", "publications"])

            profile_data = {
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
            for pub in publications:
                try:
                    pub_filled = scholarly.fill(pub)
                    bib = pub_filled.get("bib", {})

                    article = {
                        "article_id": f"{scholar_id}:{pub_filled.get('author_pub_id', '')}",
                        "titre": bib.get("title", ""),
                        "auteurs": bib.get("author", "").split(" and ") if bib.get("author") else [],
                        "date_publication": str(bib.get("pub_year", "")),
                        "journal": bib.get("journal", bib.get("conference", "")),
                        "citations": pub_filled.get("num_citations", 0),
                        "abstract": bib.get("abstract", "")
                    }
                    profile_data["articles"].append(article)
                    time.sleep(1)
                except Exception as pub_err:
                    print(f"    [!] Skipping publication due to error: {pub_err}")

            scraped_data.append(profile_data)

            with open(OUTPUT_RAW_FILE, "w", encoding="utf-8") as out:
                json.dump(scraped_data, out, ensure_ascii=False, indent=2)

            time.sleep(3)

        except Exception as e:
            print(f"[!] Error processing {author_name}: {e}")

    print(f"[+] Scraping complete. Saved to {OUTPUT_RAW_FILE}")

if __name__ == "__main__":
    scrape_profiles()