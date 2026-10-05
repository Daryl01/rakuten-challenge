"""Contenus validés affichés par l'application.

Les scores proviennent de la réexécution des notebooks 02 et 03 qui a produit le
modèle servi (models/baselines/tfidf_linearsvc.pkl). Les agrégats par classe de
streamlit_app/data/ ont été recalculés avec ce même modèle sur le split de
validation du notebook 02 (test_size=0.2, random_state=42, stratifié).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.paths import CLASS_METRICS_PATH, CONFUSIONS_PATH, RESULTS_PATH

SERVED_MODEL_NAME = "TF-IDF + LinearSVC"
SERVED_MODEL_F1 = 0.8316
REPORTED_F1_PREVIOUS_RUN = 0.8403
TEXT_BENCHMARK_F1 = 0.8113
VALIDATION_SIZE = 16_984
VALIDATION_ERRORS = 2_833

# Un portrait est affiché si le fichier existe dans streamlit_app/assets/team/, sinon les initiales.
# `scale` réduit l'affichage d'une photo cadrée plus serré (fond blanc) pour harmoniser la taille des visages,
# sans recadrer ni modifier l'image.
TEAM = [
    {"name": "Khoty WOLIE", "initials": "KW", "photo": "khoty_wolie.jpg", "scale": 1.0},
    {"name": "HaoYue Yuan", "initials": "HY", "photo": "haoyue_yuan.jpg", "scale": 1.0},
    {"name": "Loyck Bibissi", "initials": "LB", "photo": "loyck_bibissi.jpg", "scale": 0.8},
]
MENTOR = "Kylian Santos"
TRAINING = "Formation Ingénieur IA, Liora (ex DataScientest)"
DEFENSE_DATE = "16 octobre 2026"
REPO_URL = "https://github.com/Daryl01/rakuten-challenge"

# Libellés indicatifs choisis par l'équipe : Rakuten ne fournit que les codes.
LABEL_NAMES = {
    10: "Livres d'occasion",
    40: "Jeux vidéo et accessoires",
    50: "Accessoires gaming",
    60: "Consoles de jeux",
    1140: "Figurines et produits dérivés",
    1160: "Cartes de collection",
    1180: "Jeux de rôle et figurines",
    1280: "Jouets pour enfants",
    1281: "Jeux de société",
    1300: "Modélisme et télécommandé",
    1301: "Accessoires enfants",
    1302: "Jeux d'extérieur",
    1320: "Puériculture",
    1560: "Mobilier intérieur",
    1920: "Linge de maison",
    1940: "Alimentation et boissons",
    2060: "Décoration intérieure",
    2220: "Animalerie",
    2280: "Magazines et revues",
    2403: "Livres et bandes dessinées",
    2462: "Jeux et jouets",
    2522: "Papeterie et fournitures",
    2582: "Mobilier et jardin",
    2583: "Piscine et accessoires",
    2585: "Outillage et bricolage",
    2705: "Livres neufs",
    2905: "Jeux PC",
}

# Fiches réelles du jeu d'entraînement Rakuten (désignation et code d'origine).
EXAMPLES = {
    "Jeu vidéo importé": {"designation": "Shin Masoukishin Panzer Warfare (Import Jap)", "code": 40},
    "Kit piscine": {"designation": "Kit de désinfection pour piscines enfants 20990", "code": 2583},
    "Carte à collectionner": {"designation": "Phyllali Réverse 7/111 - XY Poings Furieux", "code": 1160},
    "Jeu Game Boy": {"designation": "Dragon Warrior Monsters 2: Tara's Adventure (Version US)", "code": 40},
    "Livre classique": {"designation": "Télémaque - Je Ne Sais Quoi De Pur Et De Sublime", "code": 10},
    "Titre ambigu": {"designation": "La Promenade Au Parc", "code": 10},
}

RESULT_NAME_ALIASES = {
    "Benchmark Rakuten - Image (ResNet50)": "Benchmark Rakuten image",
    "Benchmark Rakuten - Texte (CNN)": "Benchmark Rakuten texte",
    "ResNet50 Fine-tuning": "ResNet50 fine-tuning",
    "Fusion Multimodale (SBERT + ResNet50)": "Fusion SBERT + ResNet50",
}
MODALITIES = {
    "TF-IDF + LinearSVC": "Texte",
    "Benchmark Rakuten texte": "Texte",
    "SBERT + MLP": "Texte",
    "SBERT + LogisticRegression": "Texte",
    "Fusion SBERT + ResNet50": "Texte + image",
    "ResNet50 fine-tuning": "Image",
    "Benchmark Rakuten image": "Image",
}


def label(code: int) -> str:
    return LABEL_NAMES.get(int(code), "Catégorie non libellée")


class ContentUnavailableError(RuntimeError):
    """Fichier de résultats absent ou illisible."""


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Scores F1 pondérés de validation, tels qu'exportés par le notebook 02."""
    try:
        frame = pd.read_csv(RESULTS_PATH)
        frame = frame[["Modèle", "Weighted F1"]].dropna()
    except (OSError, KeyError, pd.errors.ParserError) as exc:
        raise ContentUnavailableError(f"Résultats illisibles ({RESULTS_PATH.name}) : {exc}") from exc
    frame["Modèle"] = frame["Modèle"].replace(RESULT_NAME_ALIASES)
    frame["Modalité"] = frame["Modèle"].map(MODALITIES).fillna("Autre")
    frame["Type"] = frame["Modèle"].str.startswith("Benchmark").map({True: "Benchmark officiel", False: "Modèle du projet"})
    return frame.sort_values("Weighted F1", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_class_metrics() -> pd.DataFrame:
    try:
        frame = pd.read_csv(CLASS_METRICS_PATH)
    except OSError as exc:
        raise ContentUnavailableError(f"Métriques par classe introuvables : {exc}") from exc
    frame["libelle"] = frame["code"].map(label)
    frame["classe"] = frame["code"].astype(str) + " · " + frame["libelle"]
    return frame


@st.cache_data(show_spinner=False)
def load_confusions() -> pd.DataFrame:
    try:
        frame = pd.read_csv(CONFUSIONS_PATH)
    except OSError as exc:
        raise ContentUnavailableError(f"Confusions introuvables : {exc}") from exc
    frame["Classe réelle"] = frame["classe_reelle"].map(lambda c: f"{c} · {label(c)}")
    frame["Classe prédite"] = frame["classe_predite"].map(lambda c: f"{c} · {label(c)}")
    return frame.rename(columns={"nombre": "Erreurs"})[["Classe réelle", "Classe prédite", "Erreurs"]]
