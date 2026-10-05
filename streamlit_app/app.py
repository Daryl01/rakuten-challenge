"""Point d'entrée de l'application de soutenance du projet Rakuten.

Lancement depuis la racine du projet : streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.content import MENTOR, SERVED_MODEL_F1, SERVED_MODEL_NAME, TEAM  # noqa: E402
from core.model_service import inspect_artifact  # noqa: E402
from core.ui import fr, inject_css  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="Rakuten · Classification de produits",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

pages = {
    "Projet": [
        st.Page(APP_DIR / "views" / "accueil.py", title="Accueil", default=True),
        st.Page(APP_DIR / "views" / "donnees.py", title="Données"),
        st.Page(APP_DIR / "views" / "modelisation.py", title="Modélisation"),
    ],
    "Démonstration": [
        st.Page(APP_DIR / "views" / "demo.py", title="Prédiction en direct"),
        st.Page(APP_DIR / "views" / "analyse.py", title="Analyse du modèle"),
    ],
    "Synthèse": [
        st.Page(APP_DIR / "views" / "conclusion.py", title="Conclusion"),
        st.Page(APP_DIR / "views" / "equipe.py", title="Équipe"),
    ],
}
navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("**Rakuten · Classification multimodale**")
    st.caption(" · ".join(member["name"] for member in TEAM))
    st.caption(f"Mentor : {MENTOR}")
    artifact = inspect_artifact()
    if artifact.exists and not artifact.is_lfs_pointer:
        st.caption(f"Modèle disponible ({artifact.size_mb:.0f} Mo)")
    else:
        st.caption("Modèle absent")
    st.caption(f"{SERVED_MODEL_NAME} · F1 pondéré {fr(SERVED_MODEL_F1)}")

navigation.run()
