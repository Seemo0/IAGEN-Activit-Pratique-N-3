"""Tests du graphe avec un modèle programmé, sans appel à l’API."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from rag_agent import charger_modele, creer_agent, interroger_agent


class ModeleProgramme:
    def __init__(self, reponses):
        self.reponses = iter(reponses)
        self.historiques = []

    def bind_tools(self, outils):
        return self

    def invoke(self, messages):
        self.historiques.append(messages)
        return next(self.reponses)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.appels = []
        appels = self.appels

        @tool("retriever_tool", response_format="content_and_artifact")
        def outil(texte: str):
            """Outil de test renvoyant un passage documentaire fixe."""
            appels.append(texte)
            source = {"type": "texte", "pdf": "article.pdf", "page": 4,
                      "source": "page_004.txt", "contenu": "Passage de test."}
            return json.dumps({"sources": [source]}), [source]
        self.outil = outil

    def appel(self, identifiant, nom="retriever_tool"):
        return {"name": nom, "args": {"texte": "attention"}, "id": identifiant,
                "type": "tool_call"}

    def test_action_observation_reponse(self):
        modele = ModeleProgramme([
            AIMessage(content="", tool_calls=[self.appel("a")]),
            AIMessage(content="Réponse de test [article.pdf, p. 4].")
        ])
        r = interroger_agent(creer_agent(modele, self.outil), "Question de test")
        self.assertEqual(r["nb_appels"], 1)
        self.assertEqual(len(self.appels), 1)
        observations = [m for m in modele.historiques[1] if isinstance(m, ToolMessage)]
        self.assertEqual(observations[0].tool_call_id, "a")
        self.assertIn("Passage de test", observations[0].artifact[0]["contenu"])
        self.assertFalse(r["messages"][-1].tool_calls)

    def test_budget_meme_avec_plusieurs_appels_dans_un_message(self):
        modele = ModeleProgramme([
            AIMessage(content="", tool_calls=[self.appel(str(i)) for i in range(4)]),
            AIMessage(content="Synthèse de test.")
        ])
        r = interroger_agent(creer_agent(modele, self.outil, max_appels=2), "Question")
        self.assertEqual(len(self.appels), 2)
        self.assertEqual(r["nb_appels"], 2)
        observations = [m for m in r["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(observations), 4)
        self.assertEqual(sum(m.status == "error" for m in observations), 2)
        self.assertFalse(any(isinstance(m, AIMessage) and m.tool_calls
                             for m in modele.historiques[-1]))
        self.assertTrue(any(isinstance(m, HumanMessage) and isinstance(m.content, str)
                            and "Passage de test" in m.content
                            for m in modele.historiques[-1]))

    def test_outil_inconnu_et_salutation(self):
        modele = ModeleProgramme([
            AIMessage(content="", tool_calls=[self.appel("a", "inconnu")]),
            AIMessage(content="Fin du test.")
        ])
        r = interroger_agent(creer_agent(modele, self.outil, max_appels=1), "Question")
        self.assertEqual(self.appels, [])
        self.assertTrue(any(isinstance(m, ToolMessage) and m.status == "error"
                            for m in r["messages"]))
        modele = ModeleProgramme([AIMessage(content="Bonjour !")])
        r = interroger_agent(creer_agent(modele, self.outil), "Bonjour")
        self.assertEqual(r["nb_appels"], 0)

    def test_une_nouvelle_question_repart_de_zero(self):
        modele = ModeleProgramme([AIMessage(content="Bonjour"), AIMessage(content="Salut")])
        agent = creer_agent(modele, self.outil)
        interroger_agent(agent, "Bonjour")
        r = interroger_agent(agent, "Salut")
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in r["messages"]), 1)

    def test_images_transmises_avec_les_sources_et_limitees_a_trois(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            sources = []
            for i in range(4):
                chemin = Path(tmp) / f"image_{i}.png"
                Image.new("RGB", (8, 8), color=(i * 50, 0, 0)).save(chemin)
                sources.append({"type": "image", "pdf": "article.pdf", "page": i + 1,
                                "source": str(chemin)})

            @tool("retriever_tool", response_format="content_and_artifact")
            def outil_images(texte: str):
                """Outil de test renvoyant quatre images temporaires."""
                return json.dumps({"sources": sources}), sources

            modele = ModeleProgramme([
                AIMessage(content="", tool_calls=[self.appel("images")]),
                AIMessage(content="Réponse de test sur les images.")
            ])
            r = interroger_agent(creer_agent(modele, outil_images), "Trouve les figures")
            self.assertEqual(len(r["images_envoyees"]), 3)
            messages_images = [m for m in modele.historiques[1]
                               if isinstance(m, HumanMessage) and isinstance(m.content, list)
                               and any(b.get("type") == "image_url" for b in m.content)]
            blocs = [b for m in messages_images for b in m.content if b.get("type") == "image_url"]
            self.assertEqual(len(blocs), 3)
            self.assertTrue(all(b["image_url"]["url"].startswith("data:image/jpeg;base64,")
                                for b in blocs))


class TestConfigurationLocale(unittest.TestCase):
    def test_modele_local_sans_cle_api(self):
        with patch.dict(os.environ, {}, clear=True):
            modele = charger_modele(env_path="fichier-env-inexistant")
            self.assertEqual(modele.model, "qwen3-vl:4b-instruct")
            self.assertEqual(modele.base_url, "http://127.0.0.1:11435")

    def test_configuration_personnalisee(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "modele-local",
                                     "OLLAMA_BASE_URL": "http://localhost:11435"}, clear=True):
            modele = charger_modele(env_path="fichier-env-inexistant")
            self.assertEqual(modele.model, "modele-local")

    def test_adresse_distante_refusee(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "https://example.com"}, clear=True):
            with self.assertRaises(ValueError):
                charger_modele(env_path="fichier-env-inexistant")


if __name__ == "__main__":
    unittest.main()
