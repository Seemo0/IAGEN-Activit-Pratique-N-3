"""Extraction du PDF et embeddings multimodaux avec CLIP."""

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
