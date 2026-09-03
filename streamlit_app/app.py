"""Application de démonstration du projet Rakuten."""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_preprocessing import PreparedText, prepare_text_for_model


LOGGER = logging.getLogger(__name__)
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_PATH = MODELS_DIR / "artifacts" / "resultats_comparaison.csv"
TEXT_MODEL_PATH = MODELS_DIR / "baselines" / "tfidf_linearsvc.pkl"
RESULT_NAME_ALIASES = {
    "Benchmark Rakuten - Image (ResNet50)": "Benchmark image",
    "ResNet50 Fine-tuning": "ResNet50",
    "Fusion Multimodale (SBERT + ResNet50)": "Fusion multimodale",
    "Benchmark Rakuten - Texte (CNN)": "Benchmark texte",
}

MODEL_ARTIFACTS = [
    {
        "Modèle": "TF-IDF + LinearSVC",
        "Modalité": "Texte",
        "Artefact": "tfidf_linearsvc.pkl",
        "Chemin": TEXT_MODEL_PATH,
        "Inférence directe": "Oui",
        "Limitation": "Environ 51 Mo, compatibilité scikit-learn requise",
    },
    {
        "Modèle": "SBERT + MLP",
        "Modalité": "Texte",
        "Artefact": "mlp_sbert_best.pt",
        "Chemin": MODELS_DIR / "text" / "mlp_sbert_best.pt",
        "Inférence directe": "Non",
        "Limitation": "Encodeur SBERT et pipeline complet non embarqués",
    },
    {
        "Modèle": "ResNet50",
        "Modalité": "Image",
        "Artefact": "resnet50_phase2_best.pt",
        "Chemin": MODELS_DIR / "image" / "resnet50_phase2_best.pt",
        "Inférence directe": "Non",
        "Limitation": "Architecture, mapping et transformations à figer pour le service",
    },
    {
        "Modèle": "Fusion multimodale",
        "Modalité": "Texte + image",
        "Artefact": "mlp_fusion_best.pt",
        "Chemin": MODELS_DIR / "text" / "mlp_fusion_best.pt",
        "Inférence directe": "Non",
        "Limitation": "Chaîne SBERT, ResNet et ordre des features non empaquetés",
    },
]

FALLBACK_RESULTS = pd.DataFrame(
    {
        "Modèle": [
            "TF-IDF + LinearSVC",
            "Benchmark texte",
            "SBERT + MLP",
            "Fusion multimodale",
            "SBERT + LogisticRegression",
            "ResNet50",
            "Benchmark image",
        ],
        "Weighted F1": [0.8316, 0.8113, 0.7542, 0.7105, 0.6829, 0.6799, 0.5534],
    }
)

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

EXAMPLES = {
    "Livre jeunesse": (
        "Le Petit Prince édition illustrée",
        "Livre jeunesse relié en français, texte intégral et illustrations couleur.",
    ),
    "Jeu de société": (
        "Jeu de société familial stratégie",
        "Jeu de plateau pour 2 à 6 joueurs à partir de 8 ans.",
    ),
    "Mobilier jardin": (
        "Salon de jardin quatre places",
        "Ensemble extérieur avec table et fauteuils résistants aux intempéries.",
    ),
}

DIFFICULT_CLASSES = pd.DataFrame(
    {
        "Classe": [10, 1281, 1180, 1280],
        "Précision": [0.5368, 0.6479, 0.8381, 0.7187],
        "Rappel": [0.5152, 0.5556, 0.5752, 0.6663],
        "F1": [0.5258, 0.5982, 0.6822, 0.6915],
    }
)

TOP_CONFUSIONS = pd.DataFrame(
    {
        "Classe réelle": [2403, 1280, 10, 1281, 10, 2705],
        "Classe prédite": [2280, 1300, 2280, 1280, 2403, 10],
        "Nombre": [131, 115, 90, 88, 83, 73],
    }
)


st.set_page_config(
    page_title="Rakuten | Classification multimodale",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {--navy:#10233f; --blue:#2e75b6; --sky:#eaf3fc; --coral:#ed5a52;
        --ink:#182536; --muted:#65758b; --line:#dbe5f0;}
    .stApp {background: linear-gradient(180deg, #f8fbff 0, #ffffff 420px);}
    .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg,#0f1e38 0%,#17345d 55%,#2e75b6 100%);}
    [data-testid="stSidebar"] * {color: #fff !important;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:rgba(255,255,255,.65)!important;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:.34rem .45rem;border-radius:7px;}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(255,255,255,.09);}
    [data-testid="stMetric"] {background: linear-gradient(145deg,#ffffff,#f0f6fd);
        border: 1px solid var(--line); border-top: 4px solid var(--blue);
        border-radius: 14px; padding: 1rem 1.1rem; box-shadow:0 7px 24px rgba(25,65,110,.08);}
    div[data-testid="stAlert"] {border-radius: 9px;}
    h1, h2, h3 {letter-spacing: -0.025em; color:var(--ink);}
    .eyebrow {color: #2457a7; font-size: .82rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;}
    .hero {color:#fff; background:linear-gradient(125deg,#10233f 0%,#245d94 68%,#2e75b6 100%);
        border-radius:18px; padding:2rem 2.2rem; margin:.8rem 0 1.4rem;
        box-shadow:0 16px 40px rgba(16,35,63,.18); position:relative; overflow:hidden;}
    .hero:after {content:'TEXT  +  IMAGE'; position:absolute;right:1.5rem;bottom:1rem;
        color:rgba(255,255,255,.12);font-weight:800;font-size:2.3rem;letter-spacing:.06em;}
    .hero h2 {color:#fff;margin:0 0 .5rem;font-size:2rem;max-width:780px;}
    .hero p {color:#dcecff;margin:0;max-width:730px;font-size:1.02rem;}
    .hero-badge {display:inline-block;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.24);
        color:#fff;border-radius:999px;padding:.32rem .72rem;margin-bottom:.9rem;font-size:.78rem;font-weight:700;}
    .metric-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:1rem 0 1.5rem;}
    .metric-card {background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem;
        box-shadow:0 7px 22px rgba(25,65,110,.07);min-width:0;}
    .metric-label {color:var(--muted);font-size:.78rem;font-weight:650;text-transform:uppercase;letter-spacing:.04em;}
    .metric-value {color:var(--navy);font-size:1.55rem;line-height:1.18;font-weight:760;margin-top:.35rem;overflow-wrap:anywhere;}
    .metric-note {color:var(--blue);font-size:.76rem;margin-top:.35rem;}
    .section-card {background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.15rem 1.25rem;
        box-shadow:0 6px 22px rgba(25,65,110,.06);height:100%;}
    .section-icon {font-size:1.65rem;margin-bottom:.4rem;}
    .section-card strong {color:var(--navy);}
    .section-card small {color:var(--muted);line-height:1.45;display:block;margin-top:.25rem;}
    .active-model {background:linear-gradient(135deg,#eaf3fc,#fff);border:1px solid #bcd3eb;
        border-left:5px solid var(--blue);border-radius:13px;padding:1rem 1.2rem;margin:.8rem 0 1rem;}
    .active-model .name {font-size:1.25rem;color:var(--navy);font-weight:750;}
    .active-model .meta {color:var(--muted);font-size:.84rem;margin-top:.25rem;}
    .result-hero {background:linear-gradient(125deg,#10233f,#2e75b6);color:#fff;border-radius:16px;
        padding:1.35rem 1.5rem;margin:.8rem 0 1rem;box-shadow:0 12px 30px rgba(16,35,63,.18);}
    .result-hero .result-label {color:#cfe4f8;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;}
    .result-hero .result-name {font-size:1.65rem;font-weight:760;margin:.28rem 0;}
    .result-hero .result-code {color:#fff;font-size:.9rem;}
    .flow {display:grid;grid-template-columns:repeat(5,1fr);gap:.55rem;align-items:stretch;margin:1rem 0 1.5rem;}
    .flow-step {background:#fff;border:1px solid var(--line);border-radius:11px;padding:.8rem .55rem;text-align:center;
        color:var(--navy);font-weight:680;font-size:.88rem;box-shadow:0 4px 15px rgba(25,65,110,.05);}
    .flow-step span {display:block;color:var(--blue);font-size:1.2rem;margin-bottom:.25rem;}
    [data-baseweb="tab-list"] {gap:.3rem;border-bottom:2px solid #e1eaf4;}
    [data-baseweb="tab"] {font-weight:650;border-radius:8px 8px 0 0;padding:.65rem 1rem;}
    .stDataFrame {border:1px solid var(--line);border-radius:10px;overflow:hidden;}
    .muted {color: #5f6b7a;}
    @media (max-width: 800px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        [data-testid="column"] {min-width: 100% !important;}
        .metric-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
        .flow {grid-template-columns:1fr;}
        .hero:after {display:none;}
        .hero {padding:1.45rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement du modèle texte...")
def load_text_model(path: str):
    """Charge une seule fois le pipeline TF-IDF + LinearSVC."""
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_results(path: str) -> tuple[pd.DataFrame, str]:
    """Charge la source centrale des scores avec un repli documenté."""
    source = Path(path)
    try:
        frame = pd.read_csv(source)
        if not {"Modèle", "Weighted F1"}.issubset(frame.columns):
            raise ValueError("colonnes 'Modèle' et 'Weighted F1' absentes")
        frame = frame[["Modèle", "Weighted F1"]].dropna().copy()
        frame["Modèle"] = frame["Modèle"].replace(RESULT_NAME_ALIASES)
        frame["Weighted F1"] = pd.to_numeric(frame["Weighted F1"], errors="raise")
        return frame.sort_values("Weighted F1", ascending=False), source.name
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        LOGGER.warning("Résultats centraux indisponibles: %s", exc)
        return FALLBACK_RESULTS.copy(), "valeurs de référence intégrées"


def find_figure(filename: str) -> Path | None:
    """Localise une figure existante sans masquer les chemins manquants."""
    candidates = [FIGURES_DIR / filename, PROJECT_ROOT / "reports" / filename]
    return next((path for path in candidates if path.is_file()), None)


def display_figure(filename: str, caption: str) -> bool:
    path = find_figure(filename)
    if path is None:
        st.info(f"Figure indisponible : {filename}")
        return False
    st.image(str(path), caption=caption, width="stretch")
    return True


def artifact_status() -> pd.DataFrame:
    rows = []
    for artifact in MODEL_ARTIFACTS:
        path = artifact["Chemin"]
        rows.append(
            {
                "Modèle": artifact["Modèle"],
                "Modalité": artifact["Modalité"],
                "Fichier": artifact["Artefact"],
                "Statut": "Disponible" if path.is_file() else "Absent",
                "Inférence directe": artifact["Inférence directe"],
                "Limitation": artifact["Limitation"],
            }
        )
    return pd.DataFrame(rows)


def validate_image(uploaded_file) -> tuple[Image.Image | None, str | None]:
    """Valide et décode une image envoyée par l'utilisateur."""
    if uploaded_file is None:
        return None, None
    if uploaded_file.size == 0:
        return None, "Le fichier image est vide."
    if uploaded_file.size > 5 * 1024 * 1024:
        return None, "L'image dépasse la limite de 5 Mo."
    try:
        raw = uploaded_file.getvalue()
        image = Image.open(BytesIO(raw))
        image.verify()
        image = Image.open(BytesIO(raw)).convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        LOGGER.warning("Image invalide: %s", exc)
        return None, "Le fichier ne contient pas une image JPG ou PNG valide."


def reset_demo() -> None:
    st.session_state["designation"] = ""
    st.session_state["description"] = ""
    st.session_state["prediction_result"] = None
    st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1


def fill_example(name: str) -> None:
    designation, description = EXAMPLES[name]
    st.session_state["designation"] = designation
    st.session_state["description"] = description
    st.session_state["prediction_result"] = None


def local_contributions(pipeline, prepared: PreparedText, predicted_class: int) -> pd.DataFrame:
    """Calcule coefficient x TF-IDF pour la classe prédite, hors intercept."""
    vectorizer = pipeline.steps[0][1]
    classifier = pipeline.steps[-1][1]
    vector = vectorizer.transform([prepared.text_combined])
    classes = np.asarray(classifier.classes_)
    positions = np.flatnonzero(classes == predicted_class)
    if len(positions) != 1:
        return pd.DataFrame(columns=["Terme", "Contribution"])
    class_index = int(positions[0])
    row = vector.getrow(0)
    class_coefficients = classifier.coef_[class_index, row.indices]
    if hasattr(class_coefficients, "toarray"):
        class_coefficients = class_coefficients.toarray()
    contributions = row.data * np.asarray(class_coefficients).ravel()
    names = vectorizer.get_feature_names_out()[row.indices]
    return pd.DataFrame({"Terme": names, "Contribution": contributions})


def predict_text(pipeline, designation: str, description: str) -> dict:
    prepared = prepare_text_for_model(designation, description)
    if not prepared.text_combined.strip():
        raise ValueError("Le texte est vide après préprocessing.")
    predicted_class = int(pipeline.predict([prepared.text_combined])[0])
    raw_scores = np.asarray(pipeline.decision_function([prepared.text_combined])).reshape(-1)
    classes = np.asarray(pipeline.classes_)
    order = np.argsort(raw_scores)[::-1][:3]
    top3 = pd.DataFrame(
        {
            "Rang": [1, 2, 3],
            "Code": [int(classes[index]) for index in order],
            "Catégorie": [LABEL_NAMES.get(int(classes[index]), "Catégorie produit") for index in order],
            "Score de décision": [float(raw_scores[index]) for index in order],
        }
    )
    contributions = local_contributions(pipeline, prepared, predicted_class)
    positive = contributions[contributions["Contribution"] > 0].nlargest(10, "Contribution")
    negative = contributions[contributions["Contribution"] < 0].nsmallest(10, "Contribution")
    predicted_index = int(np.flatnonzero(classes == predicted_class)[0])
    return {
        "class": predicted_class,
        "name": LABEL_NAMES.get(predicted_class, "Catégorie produit"),
        "score": float(raw_scores[predicted_index]),
        "top3": top3,
        "positive": positive,
        "negative": negative,
        "prepared": prepared,
    }


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted">{description}</p>', unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>'
        for label, value, note in items
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def render_home() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-badge">PROJET DATA SCIENCE ET IA</div>'
        '<h2>Classification multimodale de produits Rakuten</h2>'
        '<p>Du texte et de l’image vers une catégorie e-commerce parmi 27 classes. '
        'Une étude comparative conçue pour comprendre ce que chaque modalité apporte.</p></div>',
        unsafe_allow_html=True,
    )
    metric_cards(
        [
            ("Produits d'entraînement", "84 916", "Catalogue Rakuten"),
            ("Catégories", "27", "Codes prdtypecode"),
            ("Meilleur modèle", "TF-IDF + LinearSVC", "Pipeline texte"),
            ("F1 pondéré", "0,8316", "Évaluation actuelle"),
        ]
    )
    st.success("Le modèle texte TF-IDF + LinearSVC reste le plus performant avec un F1 pondéré de 0,8316.")
    st.subheader("Du catalogue à la prédiction")
    st.markdown(
        '<div class="flow">'
        '<div class="flow-step"><span>▦</span>Données</div>'
        '<div class="flow-step"><span>✦</span>Préprocessing</div>'
        '<div class="flow-step"><span>TXT</span>Modèles texte</div>'
        '<div class="flow-step"><span>IMG</span>Image et fusion</div>'
        '<div class="flow-step"><span>✓</span>Évaluation</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    stages = [
        ("🔤", "Baseline classique", "TF-IDF + LinearSVC", "Termes et bigrammes discriminants."),
        ("🧠", "Représentation dense", "SBERT + MLP", "Embeddings textuels multilingues."),
        ("🖼️", "Vision", "ResNet50", "Transfer learning sur les images."),
        ("🔗", "Multimodal", "Fusion tardive", "Combinaison texte et image."),
    ]
    for column, (icon, title, model, note) in zip(columns, stages):
        column.markdown(
            f'<div class="section-card"><div class="section-icon">{icon}</div>'
            f'<strong>{title}</strong><small>{model}<br>{note}</small></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Exploration visuelle")
    st.write("Trois vues synthétiques issues de l'analyse exploratoire, sans recharger le dataset complet.")
    distribution, text, products = st.tabs(["Répartition des classes", "Texte et langues", "Produits en images"])
    with distribution:
        display_figure("01_distribution_classes.png", "Distribution des 84 916 produits parmi les 27 catégories.")
        st.caption("Le déséquilibre entre catégories justifie l'utilisation du F1 pondéré.")
    with text:
        text_col, language_col = st.columns(2)
        with text_col:
            display_figure("02_longueur_textes.png", "Longueur des désignations et descriptions.")
        with language_col:
            display_figure("03_langues_designations.png", "Langues détectées dans les désignations.")
    with products:
        display_figure("06_exemples_images_classes.png", "Exemples de produits issus de plusieurs catégories.")
        st.caption("Les images apportent un signal complémentaire, mais plus variable que le texte.")


def render_demo() -> None:
    page_header(
        "Inférence locale",
        "Démonstration",
        "Testez le pipeline réellement sauvegardé, avec le même préprocessing que celui de l'entraînement.",
    )
    st.markdown(
        '<div class="active-model"><div class="eyebrow">MODÈLE ACTIF</div>'
        '<div class="name">TF-IDF + LinearSVC</div>'
        '<div class="meta">Artefact : tfidf_linearsvc.pkl &nbsp; | &nbsp; Weighted F1 : 0,8316 '
        '&nbsp; | &nbsp; Modalité : texte</div></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Les checkpoints image et fusion sont présentés dans l'application, mais leur chaîne "
        "d'inférence complète n'est pas suffisamment empaquetée pour une prédiction fiable."
    )

    st.markdown("### 1. Entrées produit")
    st.caption("Choisissez un exemple ou saisissez une nouvelle fiche produit.")
    st.markdown("#### Exemples")
    example_columns = st.columns(3)
    for column, name in zip(example_columns, EXAMPLES):
        column.button(name, on_click=fill_example, args=(name,), width="stretch")

    st.text_area("Désignation", key="designation", height=90, placeholder="Exemple : jeu de société familial")
    st.text_area("Description facultative", key="description", height=120)

    upload_key = f"product_image_{st.session_state.get('uploader_version', 0)}"
    uploaded = st.file_uploader(
        "Image facultative (JPG, JPEG ou PNG, 5 Mo maximum)",
        type=["jpg", "jpeg", "png"],
        key=upload_key,
        max_upload_size=5,
    )
    image, image_error = validate_image(uploaded)
    if image_error:
        st.error(image_error)
    elif image is not None:
        st.image(image, caption="Aperçu de l'image chargée", width=320)
        st.caption("Image chargée pour visualisation. La prédiction principale utilise actuellement le modèle texte.")

    action, reset, space = st.columns([1, 1, 3])
    predict_clicked = action.button("Prédire", type="primary", width="stretch")
    reset.button("Réinitialiser", on_click=reset_demo, width="stretch")

    if predict_clicked:
        if not st.session_state.get("designation", "").strip() and not st.session_state.get("description", "").strip():
            st.warning("Saisissez au moins une désignation ou une description.")
            st.session_state["prediction_result"] = None
        elif not TEXT_MODEL_PATH.is_file():
            st.error(f"Artefact texte introuvable : {TEXT_MODEL_PATH.relative_to(PROJECT_ROOT)}")
        else:
            try:
                model = load_text_model(str(TEXT_MODEL_PATH))
                st.session_state["prediction_result"] = predict_text(
                    model,
                    st.session_state.get("designation", ""),
                    st.session_state.get("description", ""),
                )
            except (OSError, ValueError, RuntimeError, AttributeError) as exc:
                LOGGER.exception("Échec de la prédiction")
                st.error(f"La prédiction n'a pas pu être produite : {exc}")

    result = st.session_state.get("prediction_result")
    if result:
        st.divider()
        st.markdown("### 2. Résultat du modèle")
        st.markdown(
            f'<div class="result-hero"><div class="result-label">Catégorie prédite</div>'
            f'<div class="result-name">{result["name"]}</div>'
            f'<div class="result-code">prdtypecode {result["class"]} &nbsp; | &nbsp; '
            f'Score de décision relatif {result["score"]:.3f}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Le score de décision est relatif. Il ne s'agit pas d'une probabilité calibrée.")
        st.markdown("### 3. Top 3 des classes")
        st.dataframe(result["top3"], hide_index=True, width="stretch")
        st.markdown("### 4. Interprétation locale")
        st.caption(
            "Contribution additive locale au score de décision, calculée par coefficient × valeur TF-IDF, hors intercept."
        )
        pos, neg = st.columns(2)
        with pos:
            st.markdown("**Mots favorisant la classe**")
            st.dataframe(result["positive"], hide_index=True, width="stretch")
        with neg:
            st.markdown("**Mots défavorisant la classe**")
            st.dataframe(result["negative"], hide_index=True, width="stretch")
        with st.expander("Voir le texte transmis au vectoriseur"):
            st.write(result["prepared"].text_combined or "Texte vide")


def render_performance_chart(results: pd.DataFrame) -> None:
    selected_names = [
        "TF-IDF + LinearSVC",
        "SBERT + MLP",
        "Fusion multimodale",
        "SBERT + LogisticRegression",
        "ResNet50",
    ]
    chart = results[results["Modèle"].isin(selected_names)].sort_values("Weighted F1")
    if chart.empty:
        st.info("Aucun score exploitable dans la source centrale.")
        return
    fig, axis = plt.subplots(figsize=(8, 3.5))
    colors = ["#2457a7" if name == "TF-IDF + LinearSVC" else "#91a7c7" for name in chart["Modèle"]]
    bars = axis.barh(chart["Modèle"], chart["Weighted F1"], color=colors)
    axis.set_xlim(0.60, 0.86)
    axis.set_xlabel("F1 pondéré")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    axis.bar_label(bars, fmt="%.4f", padding=4)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_performances() -> None:
    page_header(
        "Résultats de la dernière réexécution",
        "Performances",
        "Une lecture compacte des modèles, de leur apprentissage et des erreurs les plus fréquentes.",
    )
    results, source = load_results(str(RESULTS_PATH))
    st.caption(f"Source des métriques : {source}")
    metric_cards(
        [
            ("Référence", "0,8316", "TF-IDF + LinearSVC"),
            ("Meilleur réseau texte", "0,7542", "SBERT + MLP"),
            ("Meilleure fusion", "0,7105", "Texte + image"),
            ("Meilleur modèle image", "0,6799", "ResNet50"),
        ]
    )
    st.subheader("Comparaison globale")
    render_performance_chart(results)
    displayed = results[results["Modèle"].isin(FALLBACK_RESULTS["Modèle"])].copy()
    st.dataframe(
        displayed.style.format({"Weighted F1": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        '<div class="active-model"><div class="eyebrow">LECTURE PÉDAGOGIQUE</div>'
        '<div class="name">Plus complexe ne signifie pas nécessairement plus performant.</div>'
        '<div class="meta">TF-IDF + LinearSVC atteint 0,8316, devant SBERT + MLP à 0,7542 '
        'et la fusion multimodale à 0,7105.</div></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Courbes d'apprentissage")
    st.caption("Loss d'entraînement et F1 de validation au fil des époques.")
    mlp_tab, resnet_tab = st.tabs(["MLP sur embeddings SBERT", "ResNet50"])
    with mlp_tab:
        display_figure(
            "10_learning_curve_mlp_sbert.png",
            "Évolution de la loss et du F1 du MLP sur embeddings SBERT.",
        )
        st.write(
            "L'axe X représente les époques. Les axes Y suivent la loss et le F1. "
            "Le meilleur F1 atteint 0,7542 à l'époque 30, avec une progression devenue limitée en fin d'apprentissage."
        )
    with resnet_tab:
        display_figure(
            "11_learning_curve_resnet50.png",
            "Apprentissage de ResNet50 pendant les phases de gel puis d'ajustement fin.",
        )
        st.write(
            "L'axe X représente les époques des deux phases. Les axes Y suivent la loss et le F1. "
            "Le meilleur F1 atteint 0,6799 vers l'époque 8 de la seconde phase, puis les résultats fluctuent. "
            "Cette évolution invite à conserver le meilleur checkpoint plutôt que la dernière époque."
        )

    st.subheader("Classes difficiles")
    st.write("Ces quatre classes présentent un F1 inférieur à 0,70 dans l'évaluation actuelle.")
    st.dataframe(
        DIFFICULT_CLASSES.style.format({"Précision": "{:.4f}", "Rappel": "{:.4f}", "F1": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### Principales confusions")
    st.dataframe(TOP_CONFUSIONS, hide_index=True, width="stretch")
    st.caption("Les erreurs concernent surtout des catégories lexicalement ou visuellement proches.")
    with st.expander("Afficher la matrice de confusion complète"):
        display_figure("08_confusion_matrix_baseline.png", "Matrice de confusion du modèle texte retenu.")


def render_contribution_example() -> None:
    """Trace un exemple réel de contributions sans utiliser l'ancienne figure mal nommée."""
    if not TEXT_MODEL_PATH.is_file():
        st.info("Le modèle texte est nécessaire pour construire cet exemple local.")
        return
    try:
        pipeline = load_text_model(str(TEXT_MODEL_PATH))
        designation, description = EXAMPLES["Livre jeunesse"]
        result = predict_text(pipeline, designation, description)
        contributions = pd.concat([result["positive"], result["negative"]])
        contributions = contributions.loc[
            contributions["Contribution"].abs().nlargest(10).index
        ].sort_values("Contribution")
        fig, axis = plt.subplots(figsize=(9, 4.2))
        colors = ["#ed5a52" if value < 0 else "#2e75b6" for value in contributions["Contribution"]]
        axis.barh(contributions["Terme"], contributions["Contribution"], color=colors, alpha=0.9)
        axis.axvline(0, color="#182536", linewidth=0.8)
        axis.set_xlabel("Contribution locale (coefficient × TF-IDF)")
        axis.set_title(f'Contributions locales au score de décision, classe prédite {result["class"]}')
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.18)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Exemple calculé avec le pipeline sauvegardé à partir d'une fiche livre jeunesse.")
    except (OSError, ValueError, RuntimeError, AttributeError) as exc:
        LOGGER.exception("Impossible de construire l'exemple de contributions")
        st.warning(f"Exemple de contributions indisponible : {exc}")


def render_understanding() -> None:
    page_header(
        "Transparence technique",
        "Comprendre le modèle",
        "Du texte brut au score de décision, avec le statut réel des artefacts disponibles.",
    )
    st.subheader("Pipeline de prédiction")
    st.markdown(
        '<div class="flow">'
        '<div class="flow-step"><span>Aa</span>Texte brut</div>'
        '<div class="flow-step"><span>✦</span>Nettoyage</div>'
        '<div class="flow-step"><span>#</span>TF-IDF</div>'
        '<div class="flow-step"><span>ƒ</span>LinearSVC</div>'
        '<div class="flow-step"><span>✓</span>Classe prédite</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Détails du préprocessing"):
        st.markdown(
            "Nettoyage séparé de la désignation et de la description, minuscules, retrait du HTML, "
            "des URL et des caractères spéciaux, stopwords français, stemming et mots personnalisés. "
            "Le pipeline construit ensuite `text_concat`, puis `text_combined` avec repli sur la désignation."
        )
    st.subheader("Modèles entraînés disponibles")
    st.dataframe(artifact_status(), hide_index=True, width="stretch")
    st.caption(
        "Le tableau vérifie uniquement la présence des fichiers. Les checkpoints lourds ne sont pas chargés au démarrage."
    )
    st.subheader("Interpréter une prédiction")
    st.write(
        "Pour la classe prédite, l'application calcule une contribution additive locale au score de décision, "
        "calculée par coefficient × valeur TF-IDF, hors intercept. Ce calcul décrit le modèle linéaire, "
        "mais ne constitue pas un calcul SHAP."
    )
    render_contribution_example()
    st.subheader("Limites actuelles")
    st.markdown(
        "- Les scores LinearSVC ne sont pas des probabilités calibrées.\n"
        "- La prédiction image libre exige un paquet d'inférence reproductible avec architecture, mapping et transformations.\n"
        "- La fusion exige aussi les encodeurs, les dimensions et l'ordre exact des features.\n"
        "- L'artefact texte local fait environ 51 Mo et n'est actuellement pas suivi par Git."
    )


st.sidebar.markdown(
    "### 🛍️ Rakuten IA\n"
    "Classification multimodale de produits\n\n"
    "**84 916** produits &nbsp; | &nbsp; **27** classes"
)
section = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Démonstration", "Performances", "Comprendre le modèle"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Meilleur modèle")
st.sidebar.markdown("**TF-IDF + LinearSVC**  \nF1 pondéré : **0,8316**")
st.sidebar.caption("Application de soutenance Data Science et IA")

if section == "Accueil":
    render_home()
elif section == "Démonstration":
    render_demo()
elif section == "Performances":
    render_performances()
else:
    render_understanding()
