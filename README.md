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

## Étape 1 — Extraction du PDF

Le document source est [Attention Is All You Need](<attention is all you need.pdf>), de Vaswani et al. Il est inclus dans le projet.

La fonction `extraire_pdf` utilise PyMuPDF et produit :

```text
resultats_pdf/
├── texte_complet.txt
├── textes/                 # Un fichier par page
└── images/                 # Images intégrées, nommées par page et xref
```

Le notebook affiche un bilan, un extrait de texte et les images. Le découpage est effectué par page ; les fragments pour CLIP seront ajoutés à l’étape 2. L’extraction ne convertit pas les tableaux textuels et les dessins vectoriels en images.

Les fichiers extraits et `.venv` sont exclus de Git. Ils sont recréés en exécutant le projet. Aucune clé API n’est nécessaire pour cette étape.

## Avancement

- [x] Configuration UV et environnement Python 3.11.
- [x] Étape 1 : extraction du texte et des images.
- [ ] Étape 2 : embeddings CLIP et normalisation.
- [ ] Étape 3 : indexation et métadonnées.
- [ ] Étape 4 : recherche multimodale.
- [ ] Étape 5 : outil de recherche pour l’agent.
- [ ] Étape 6 : agent et boucle ReAct.
- [ ] Étape 7 : tests de l’agent.
