"""Extraction du PDF, embeddings multimodaux et indexation ChromaDB."""

from pathlib import Path

import pymupdf


def extraire_pdf(pdf_path, output_dir="resultats_pdf"):
    """Sauvegarder le texte par page et les images intégrées au PDF.

    Les numéros de page commencent à 1. Les dessins vectoriels ne sont
    pas extraits comme images par cette fonction.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path.resolve()}")

    pages = []
    images = []
    textes = []

    with pymupdf.open(pdf_path) as document:
        text_dir = output_dir / "textes"
        image_dir = output_dir / "images"
        text_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for numero, page in enumerate(document, start=1):
            texte = page.get_text("text", sort=True)
            text_path = text_dir / f"page_{numero:03d}.txt"
            text_path.write_text(texte, encoding="utf-8")
            pages.append({"page": numero, "texte": texte, "chemin": text_path})
            textes.append(f"===== PAGE {numero} =====\n\n{texte}")

            # Une même image peut être référencée plusieurs fois sur une page.
            xrefs_vus = set()
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                if xref in xrefs_vus:
                    continue
                xrefs_vus.add(xref)

                image = document.extract_image(xref)
                if not image:
                    raise ValueError(f"Image non extractible : page {numero}, xref {xref}")
                image_path = image_dir / f"page_{numero:03d}_image_{xref}.{image['ext']}"
                image_path.write_bytes(image["image"])
                images.append({"page": numero, "xref": xref, "chemin": image_path})

    texte_complet = output_dir / "texte_complet.txt"
    texte_complet.write_text("\n\n".join(textes), encoding="utf-8")
    return {"pages": pages, "images": images, "texte_complet": texte_complet}


def decouper_texte(texte, tokenizer, taille=60, chevauchement=10):
    """Découper par mots en respectant une limite de tokens CLIP.

    Le chevauchement reprend au plus 10 tokens par défaut. Les caractères
    des mots sont conservés ; les espaces sont uniformisés.
    """
    if not 0 <= chevauchement < taille <= tokenizer.context_length - 2:
        raise ValueError("Vérifier la taille des fragments et le chevauchement.")
    mots = texte.split()
    fragments = []
    debut = 0
    while debut < len(mots):
        fin = debut
        while fin < len(mots):
            candidat = " ".join(mots[debut:fin + 1])
            if len(tokenizer.encode(candidat)) > taille:
                break
            fin += 1
        if fin == debut:
            raise ValueError(f"Mot trop long pour un fragment CLIP : {mots[debut]!r}")
        fragments.append(" ".join(mots[debut:fin]))
        if fin == len(mots):
            break

        # Reprendre les derniers mots sans dépasser le chevauchement demandé.
        prochain = fin
        while prochain > debut + 1:
            suffixe = " ".join(mots[prochain - 1:fin])
            if len(tokenizer.encode(suffixe)) > chevauchement:
                break
            prochain -= 1
        debut = prochain
    return fragments


class EncodeurCLIP:
    """Encoder textes et images dans le même espace, puis normaliser en L2."""

    def __init__(self, nom_modele="ViT-B-32", poids="laion2b_s34b_b79k",
                 cache_dir=".cache/clip"):
        import open_clip
        import torch

        self.nom_modele = nom_modele
        self.poids = poids
        self.appareil = "cuda" if torch.cuda.is_available() else "cpu"
        self.modele, _, self.preprocess = open_clip.create_model_and_transforms(
            nom_modele, pretrained=poids, cache_dir=str(cache_dir)
        )
        self.modele = self.modele.to(self.appareil).eval()
        self.tokenizer = open_clip.get_tokenizer(nom_modele)

    def encoder_textes(self, textes, taille_lot=32):
        import torch
        import torch.nn.functional as F

        if not textes or taille_lot < 1:
            raise ValueError("Fournir des textes et une taille de lot positive.")
        # Le tokenizer CLIP tronque sinon les textes trop longs.
        limite = self.tokenizer.context_length - 2
        if any(len(self.tokenizer.encode(texte)) > limite for texte in textes):
            raise ValueError("Texte trop long pour CLIP : utiliser decouper_texte.")
        vecteurs = []
        with torch.inference_mode():
            for debut in range(0, len(textes), taille_lot):
                tokens = self.tokenizer(textes[debut:debut + taille_lot]).to(self.appareil)
                embeddings = self.modele.encode_text(tokens).float()
                vecteurs.append(F.normalize(embeddings, dim=-1).cpu())
        return torch.cat(vecteurs)

    def encoder_images(self, chemins, taille_lot=16):
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if not chemins or taille_lot < 1:
            raise ValueError("Fournir des images et une taille de lot positive.")
        vecteurs = []
        with torch.inference_mode():
            for debut in range(0, len(chemins), taille_lot):
                images = []
                for chemin in chemins[debut:debut + taille_lot]:
                    with Image.open(chemin) as image:
                        images.append(self.preprocess(image.convert("RGB")))
                lot = torch.stack(images).to(self.appareil)
                embeddings = self.modele.encode_image(lot).float()
                vecteurs.append(F.normalize(embeddings, dim=-1).cpu())
        return torch.cat(vecteurs)



def ouvrir_collection(nom_modele="ViT-B-32", poids="laion2b_s34b_b79k",
                      db_path="chroma_pdf"):
    """Ouvrir une collection persistante propre au modèle et à ses poids."""
    import hashlib
    import chromadb

    signature = hashlib.sha256(f"{nom_modele}:{poids}".encode()).hexdigest()[:12]
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=f"pdf_clip_{signature}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"modele": nom_modele, "poids": poids}
    )
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("La collection existante n’utilise pas la distance cosinus.")
    return collection


def indexer_pdf(collection, pdf_path, fragments, images,
                embeddings_textes, embeddings_images):
    """Remplacer l’index de ce PDF sans conserver de doublons ou anciens fragments."""
    import hashlib
    import torch

    pdf_path = Path(pdf_path)
    document_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    source_pdf = str(pdf_path.resolve())

    # Vérifier les données avant de modifier la collection.
    for elements, vecteurs in [(fragments, embeddings_textes), (images, embeddings_images)]:
        if vecteurs.ndim != 2 or vecteurs.shape != (len(elements), 512):
            raise ValueError("Les embeddings doivent correspondre aux éléments (N, 512).")
        if not torch.isfinite(vecteurs).all():
            raise ValueError("Les embeddings contiennent des valeurs non finies.")
        if not torch.allclose(vecteurs.norm(dim=1),
                              torch.ones(len(elements), device=vecteurs.device), atol=1e-5):
            raise ValueError("Les embeddings doivent être normalisés.")

    ids, documents, metadatas, embeddings = [], [], [], []
    commun = {"document_id": document_id, "pdf": pdf_path.name, "source_pdf": source_pdf}
    for fragment, vecteur in zip(fragments, embeddings_textes):
        page, numero = fragment["page"], fragment["fragment"]
        ids.append(f"{document_id}:texte:{page}:{numero}")
        documents.append(fragment["texte"])
        metadatas.append({**commun, "type": "texte", "page": page, "fragment": numero,
                          "source": str(Path(fragment["source"]).resolve())})
        embeddings.append(vecteur.tolist())
    for image, vecteur in zip(images, embeddings_images):
        chemin = Path(image["source"])
        ids.append(f"{document_id}:image:{image['page']}:{chemin.name}")
        documents.append(f"Image du PDF {pdf_path.name}, page {image['page']}")
        metadatas.append({**commun, "type": "image", "page": image["page"],
                          "source": str(chemin.resolve())})
        embeddings.append(vecteur.tolist())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Les éléments doivent avoir des identifiants uniques et non vides.")

    # Même contenu, ou nouvelle version du fichier au même emplacement.
    collection.delete(where={"$or": [
        {"document_id": document_id}, {"source_pdf": source_pdf}
    ]})
    for debut in range(0, len(ids), 64):
        fin = debut + 64
        collection.upsert(ids=ids[debut:fin], documents=documents[debut:fin],
                          metadatas=metadatas[debut:fin], embeddings=embeddings[debut:fin])
    return document_id


def rechercher(collection, encodeur, texte=None, image=None, k=5,
               type_resultat=None, document_id=None):
    """Rechercher les voisins d’un texte ou d’une image dans l’index CLIP."""
    if (texte is None) == (image is None):
        raise ValueError("Fournir soit un texte, soit une image.")
    if texte is not None and (not isinstance(texte, str) or not texte.strip()):
        raise ValueError("La requête textuelle doit être non vide.")
    if image is not None and not Path(image).is_file():
        raise FileNotFoundError(f"Image introuvable : {image}")
    if type(k) is not int or k < 1:
        raise ValueError("k doit être un entier strictement positif.")
    if type_resultat not in (None, "texte", "image"):
        raise ValueError("Le type de résultat doit être 'texte', 'image' ou None.")
    if (collection.metadata.get("modele") != encodeur.nom_modele
            or collection.metadata.get("poids") != encodeur.poids):
        raise ValueError("Utiliser le même modèle CLIP que pour l’indexation.")
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("La collection doit utiliser la distance cosinus.")

    filtres = []
    if type_resultat is not None:
        filtres.append({"type": type_resultat})
    if document_id is not None:
        filtres.append({"document_id": document_id})
    options = {}
    if filtres:
        options["where"] = filtres[0] if len(filtres) == 1 else {"$and": filtres}
        nombre = len(collection.get(include=[], **options)["ids"])
    else:
        nombre = collection.count()
    if nombre == 0:
        return []

    vecteur = (encodeur.encoder_textes([texte.strip()])[0] if texte is not None
               else encodeur.encoder_images([image])[0])
    reponse = collection.query(
        query_embeddings=[vecteur.tolist()], n_results=min(k, nombre),
        include=["documents", "metadatas", "distances"], **options
    )
    resultats = []
    for identifiant, contenu, metadata, distance in zip(
        reponse["ids"][0], reponse["documents"][0],
        reponse["metadatas"][0], reponse["distances"][0]
    ):
        resultats.append({**metadata, "id": identifiant, "contenu": contenu,
                          "distance": float(distance), "similarite": 1 - float(distance)})
    return sorted(resultats, key=lambda resultat: resultat["distance"])


def afficher_resultats(resultats):
    """Afficher les passages et les aperçus d’images avec leurs sources."""
    from IPython.display import display
    from PIL import Image

    if not resultats:
        print("Aucun résultat pour ces filtres.")
    for rang, resultat in enumerate(resultats, start=1):
        print(f"{rang}. {resultat['type'].upper()} | {resultat['pdf']} | "
              f"Page {resultat['page']} | Similarité : {resultat['similarite']:.4f}")
        print("Source :", resultat["source"])
        if resultat["type"] == "texte":
            print(resultat["contenu"])
        else:
            with Image.open(resultat["source"]) as image:
                apercu = image.convert("RGB")
                apercu.thumbnail((600, 600))
                display(apercu)
        print()
