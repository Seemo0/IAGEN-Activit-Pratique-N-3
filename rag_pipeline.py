"""Extraction du texte et des images d’un document PDF."""

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
