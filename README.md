# Activité pratique n°3 — RAG multimodal agentique

**Nom :** Mohamed Ouallal  
**Classe :** IADATA — Groupe 4

## Objectif

Créer une application qui recherche du texte et des images dans l’article *Attention Is All You Need*, puis utilise un agent pour répondre aux questions avec les sources retrouvées.

## Organisation

- `rag_multimodal.ipynb` : déroulement du travail et résultats.
- `rag_pipeline.py` : extraction, embeddings, indexation et recherche.
- `rag_agent.py` : outil de recherche et agent.
- `pyproject.toml` : configuration du projet Python.
- `uv.lock` : versions des dépendances pour reproduire l’environnement.

## Exécution

Avec UV installé, ouvrir un terminal dans ce dossier :

```bash
uv sync --locked
source .venv/bin/activate
```

Ouvrir `rag_multimodal.ipynb` dans VS Code et choisir le noyau `.venv/bin/python` (Python 3.11), puis exécuter les cellules dans l’ordre.
