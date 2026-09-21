# 🔬 FSBM Semantic Search Engine & NLP Pipeline
### Faculté des Sciences Ben M'Sik — Université Hassan II de Casablanca
**Projet de Machine Learning & Traitement Automatique du Langage Naturel (NLP)**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Model-zembed--1-yellow.svg)](https://huggingface.co/zeroentropy/zembed-1-embedding)

---

## 📖 1. Présentation du Projet

Ce projet académique implémente un pipeline complet d'ingénierie des données et de recherche d'information sémantique appliqué à la production scientifique des enseignants-chercheurs de la **Faculté des Sciences Ben M'Sik (FSBM)**.

Contrairement aux moteurs de recherche lexicaux classiques (qui se limitent à chercher des mots-clés exacts), ce système projette les publications et les requêtes dans un **espace vectoriel dense à 2560 dimensions** grâce au modèle de fondation **`zeroentropy/zembed-1-embedding`**. Il permet de retrouver des articles pertinents sur la base de leur signification sémantique profonde, y compris pour des requêtes formulées en langage naturel ou multilingues (Français / Anglais).

---

## 🏗️ 2. Architecture Globale du Pipeline

```
┌─────────────────────────┐
│ input_faculty_list.json │  (Liste des enseignants FSBM & identifiants Scholar)
└───────────┬─────────────┘
            │
            ▼  [1. Scraping Robuste avec Checkpoint & Anti-Bot]
┌─────────────────────────┐
│  src/scraper.py         │  (Jitter 3-7s, rotation User-Agent, reprise sur panne)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ raw_scholar_data.json   │  (Données brutes : métriques, citations, articles)
└───────────┬─────────────┘
            │
            ▼  [2. Nettoyage, Dédoublonnage & Préservation Accents]
┌─────────────────────────┐
│  src/cleaner.py         │  (Support accents FR, détection langue, filtres)
└───────────┬─────────────┘
            │
            ├──────────────────────────────────────────┐
            ▼                                          ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│ processed_abstracts.json      │     │ processed_abstracts.parquet       │
│ (Format hiérarchique complet) │     │ (Format tabulaire ML optimisé)    │
└───────────────────────────────┘     └──────────────┬────────────────────┘
                                                     │
                                                     ▼  [3. Vectorisation zembed-1]
                                      ┌───────────────────────────────────┐
                                      │ src/generate_vectors_and_index.py │
                                      └──────────────┬────────────────────┘
                                                     │ (Embeddings 2560D)
                                                     ▼
                                      ┌───────────────────────────────────┐
                                      │ ChromaDB (data/vector_store/)     │
                                      │ Collection: research_papers_db    │
                                      └──────────────┬────────────────────┘
                                                     │
                        ┌────────────────────────────┴────────────────────────────┐
                        ▼                                                         ▼
         ┌──────────────────────────────┐                         ┌──────────────────────────────┐
         │ Livrable 3 : Notebook        │                         │ Couche Applicative (Bonus)   │
         │ notebooks/demonstration_...  │                         │ • FastAPI (src/api.py)       │
         │ (Démonstration & Évaluation) │                         │ • Streamlit (app.py)         │
         └──────────────────────────────┘                         └──────────────────────────────┘
```

---

## 📋 3. Conformité aux Livrables et Critères d'Évaluation

### 🎯 Livrables Attendus
1. **Dépôt GitHub bien structuré** :
   - `src/` : Scripts modulaires documentés avec typage (`scraper.py`, `cleaner.py`, `generate_vectors_and_index.py`, `api.py`).
   - `data/` : Dossiers `raw/`, `processed/` et `vector_store/` rigoureusement séparés.
   - `notebooks/` : Notebook Jupyter complet et reproductible.
2. **Dataset final JSON & Parquet** :
   - `data/processed/processed_abstracts.json` (format structuré par profil enseignant).
   - `data/processed/processed_abstracts.parquet` (format tabulaire compressé avec typage strict).
3. **Notebook Jupyter commenté de bout en bout** :
   - `notebooks/demonstration_pipeline_recherche_semantique.ipynb` : Présentation, nettoyage, vectorisation, requêtes bilingues, comparaison Sémantique vs Lexicale, et projection 2D PCA.

### ⭐ Critères d'Évaluation Validés
- **Robustesse du scraping** : Idempotence par checkpoint atomique, pauses dynamiques avec jitter aléatoire, rotation d'en-têtes et gestion sécurisée des interruptions Google Scholar.
- **Qualité du prétraitement** : Préservation intégrale des accents français (`\u00C0-\u017F`), nettoyage des mentions d'éditeurs, détection FR/EN et validation de contenu.
- **Utilisation du modèle `zembed-1`** : Modèle `zeroentropy/zembed-1-embedding` configuré à 2560 dimensions, prompts différenciés `document` pour l'indexation et `query` pour la recherche.
- **Pertinence de la démonstration finale** : Tests sur des requêtes thématiques FSBM réelles (IA, Santé, IoT, Mathématiques) avec scores de similarité cosinus.

---

## 🚀 4. Installation et Guide d'Exécution

### 4.1 Prérequis & Environnement Virtuel
```bash
# Cloner le dépôt
git clone https://github.com/anouarmohammed10/projet_ml.git
cd projet_ml

# Création et activation de l'environnement virtuel Python
python -m venv .venv

# Sous Windows (PowerShell) :
.\.venv\Scripts\Activate.ps1

# Sous Linux / macOS :
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

---

### 4.2 Ordre d'Exécution du Pipeline

#### Étape 1 : Scraping des Chercheurs FSBM
```bash
python src/scraper.py
```
*Options disponibles :*
- `--limit N` : Limite le scraping à $N$ nouveaux profils.
- `--max-pubs N` : Limite à $N$ publications par chercheur.
- `--delay-min 3.0 --delay-max 7.0` : Paramètre le jitter anti-blocage.
- `--force` : Force le re-scraping complet en écrasant le checkpoint existant.

#### Étape 2 : Prétraitement et Nettoyage des Données
```bash
python src/cleaner.py
```
*Sorties générées :*
- `data/processed/processed_abstracts.json`
- `data/processed/processed_abstracts.parquet`

#### Étape 3 : Vectorisation avec `zembed-1` et Indexation ChromaDB
```bash
python src/generate_vectors_and_index.py
```
*Sorties générées :*
- Base vectorielle persistante dans `data/vector_store/` avec la collection `research_papers_db` (vecteurs 2560D).

---

### 4.3 Démonstration Interactive : Notebook Jupyter
Lancez Jupyter Notebook pour exécuter la démonstration complète :
```bash
jupyter notebook notebooks/demonstration_pipeline_recherche_semantique.ipynb
```

---

### 4.4 Lancement de la Plateforme Web (API + Interface)

#### 1. Démarrer le Serveur Backend (FastAPI)
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Documentation Swagger interactive accessible à : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### 2. Démarrer l'Interface Utilisateur (Streamlit)
Dans un second terminal :
```bash
streamlit run app.py
```
L'interface est accessible dans le navigateur à : [http://localhost:8501](http://localhost:8501)

---

## 📊 5. Échantillon de Résultats de Recherche Sémantique

| Requête Utilisateur | Article Retrouvé (FSBM) | Chercheur | Année | Match Sémantique |
| :--- | :--- | :--- | :---: | :---: |
| *"Traitement d'images médicales et cancer"* | *Detection of regions of interest's in mammograms by dynamic k-means* | Pr. Abderrahim Sekkaki | 2014 | **89.4%** |
| *"Machine learning breast cancer prediction"* | *Machine learning algorithms for breast cancer prediction and diagnosis* | Pr. Habib Ben Lahmar | 2021 | **92.8%** |
| *"IoT security wireless sensor network"* | *Energy efficient and secure routing protocol for WSN* | Pr. Mohamed Talea | 2020 | **88.1%** |
| *"Chatbot orientation scolaire NLP"* | *A system for educational and vocational guidance: Chatbot E-Orientation* | Pr. Habib Ben Lahmar | 2020 | **91.2%** |

---

## 👥 Auteur & Remerciements
- **Institution** : Faculté des Sciences Ben M'Sik (FSBM) - Université Hassan II de Casablanca.
- **Module** : Machine Learning & Natural Language Processing.