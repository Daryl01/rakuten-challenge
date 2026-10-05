"""Chemins absolus du projet, indépendants du répertoire de lancement."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baselines" / "tfidf_linearsvc.pkl"
# Permet de pointer vers un autre emplacement sans modifier le code (ex. volume monté).
MODEL_PATH = Path(os.environ.get("RAKUTEN_MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_PATH = PROJECT_ROOT / "models" / "artifacts" / "resultats_comparaison.csv"
APP_DATA_DIR = APP_DIR / "data"
TEAM_PHOTOS_DIR = APP_DIR / "assets" / "team"
CLASS_METRICS_PATH = APP_DATA_DIR / "metriques_par_classe.csv"
CONFUSIONS_PATH = APP_DATA_DIR / "principales_confusions.csv"


def relative_to_project(path: Path) -> str:
    """Affiche un chemin court et lisible pour les messages d'erreur."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
