"""Outil de recherche documentaire utilisable par un agent."""

import json
from typing import Literal

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline import rechercher


class RechercheArgs(BaseModel):
    """Paramètres de la recherche multimodale."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    texte: str | None = Field(
        default=None, min_length=1,
        description="Requête courte et précise, de préférence en anglais pour cet article."
    )
    image: str | None = Field(
        default=None, min_length=1,
        description="Chemin d’une image fournie par l’utilisateur ou retrouvée dans les sources. Ne pas inventer de chemin."
    )
    type_resultat: Literal["tous", "texte", "image"] = Field(
        default="tous", description="Type de sources à récupérer : passages, figures ou les deux."
    )
    k: int = Field(default=5, ge=1, le=10, strict=True,
                   description="Nombre maximum de résultats, de 1 à 10.")

    @model_validator(mode="after")
    def verifier_requete(self):
        if (self.texte is None) == (self.image is None):
            raise ValueError("Fournir un texte ou une image, mais pas les deux.")
        return self


def creer_outil_recherche(collection, encodeur, document_id=None):
    """Créer l’outil en lui associant l’index et le modèle déjà chargés."""

    @tool("retriever_tool", args_schema=RechercheArgs,
          response_format="content_and_artifact")
    def retriever_tool(texte: str | None = None, image: str | None = None,
                       type_resultat: str = "tous", k: int = 5):
        """Rechercher des passages et des figures dans les PDF indexés.

        Utiliser cet outil avant de répondre à une question documentaire.
        Fournir soit une requête textuelle courte, soit un chemin d’image.
        Choisir 'texte' pour les explications, 'image' pour les figures,
        ou 'tous' pour une recherche mixte. Reformuler si les sources sont
        insuffisantes. Le résultat contient des sources classées par similarité,
        avec PDF, page, extrait ou chemin d’image. Les scores ne sont pas des
        probabilités. Les chemins d’images ne constituent pas une analyse visuelle.
        L’outil recherche uniquement : il ne génère pas de réponse finale.
        """
        try:
            resultats = rechercher(
                collection, encodeur, texte=texte, image=image, k=k,
                type_resultat=None if type_resultat == "tous" else type_resultat,
                document_id=document_id
            )
        except (ValueError, FileNotFoundError) as erreur:
            raise ToolException(str(erreur)) from erreur

        sources = []
        for resultat in resultats:
            source = {cle: resultat[cle] for cle in
                      ("id", "document_id", "pdf", "page", "type", "source", "similarite")}
            if resultat["type"] == "texte":
                source["extrait"] = resultat["contenu"]
            sources.append(source)
        contenu = {
            "statut": "ok" if sources else "aucun_resultat",
            "nombre_resultats": len(sources),
            "sources": sources
        }
        # Les résultats complets restent accessibles au code via ToolMessage.artifact.
        return json.dumps(contenu, ensure_ascii=False), resultats

    retriever_tool.tags = ["retrieval", "multimodal"]
    retriever_tool.metadata = {"modele": encodeur.nom_modele, "poids": encodeur.poids}
    retriever_tool.handle_tool_error = True
    retriever_tool.handle_validation_error = (
        "Arguments invalides : fournir soit un texte non vide, soit une image, "
        "avec k entre 1 et 10 et type_resultat parmi tous, texte, image."
    )
    return retriever_tool
