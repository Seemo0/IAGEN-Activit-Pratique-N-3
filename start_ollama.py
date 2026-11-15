"""Lancer le serveur Ollama du projet ou télécharger son modèle."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="Télécharger le modèle sur le serveur lancé.")
    args = parser.parse_args()

    runtime = root / ".cache/ollama-runtime/ollama"
    executable = str(runtime) if runtime.is_file() else shutil.which("ollama")
    if not executable:
        raise SystemExit("Installer Ollama, puis relancer cette commande.")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    env["OLLAMA_MODELS"] = str(root / ".cache/ollama-models")
    env["OLLAMA_NO_CLOUD"] = "1"
    modele = os.environ.get("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
    commande = [executable, "pull", modele] if args.pull else [executable, "serve"]
    print("Serveur du projet :", env["OLLAMA_HOST"], flush=True)
    print("Modèles :", env["OLLAMA_MODELS"], flush=True)
    raise SystemExit(subprocess.call(commande, env=env, cwd=root))


if __name__ == "__main__":
    main()
