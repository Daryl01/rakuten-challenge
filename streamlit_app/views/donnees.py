"""Page données : volumétrie, visualisations et prétraitement."""

from __future__ import annotations

import altair as alt
import streamlit as st

from core.content import ContentUnavailableError, load_class_metrics
from core.ui import ACCENT, INK, card, figure, kpis, page_header
from src.text_preprocessing import prepare_text_for_model


def _class_distribution() -> None:
    try:
        metrics = load_class_metrics()
    except ContentUnavailableError as exc:
        st.error(str(exc))
        return
    data = metrics.assign(Extrême=metrics["code"].isin([2583, 1180]))
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=2, height=16)
        .encode(
            y=alt.Y("classe:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230)),
            x=alt.X("effectif_train:Q", title="Produits dans le jeu d'entraînement"),
            color=alt.condition(alt.datum["Extrême"], alt.value(ACCENT), alt.value(INK)),
            tooltip=[alt.Tooltip("classe:N", title="Classe"), alt.Tooltip("effectif_train:Q", title="Produits", format=",")],
        )
        .properties(height=alt.Step(23))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "De 10 209 produits (2583) à 764 (1180) : ce déséquilibre justifie le F1 pondéré, métrique officielle du "
        "challenge, et un split de validation stratifié."
    )


def _preprocessing_preview() -> None:
    sample = "Piscine Tubulaire <b>ronde</b> 366 x 76 cm - Livrée avec pompe ! www.exemple.fr"
    text = st.text_input("Texte à nettoyer", value=sample, max_chars=500)
    if not text.strip():
        st.caption("Saisissez un texte pour voir sa transformation.")
        return
    try:
        prepared = prepare_text_for_model(text)
    except RuntimeError as exc:
        st.warning(f"Prétraitement indisponible : {exc}")
        return
    st.code(prepared.text_combined or "(texte vide après nettoyage)", language=None, wrap_lines=True)
    st.caption("Sortie exacte de la fonction utilisée par le modèle servi (`src/text_preprocessing.py`).")


def render() -> None:
    page_header(
        "Exploration et prétraitement",
        "Les données",
        "Un catalogue réel, multilingue et déséquilibré : ses caractéristiques orientent tous les choix de modélisation.",
    )
    kpis(
        [
            ("Jeu d'entraînement", "84 916", "produits étiquetés"),
            ("Jeu de test", "13 812", "sans étiquette"),
            ("Descriptions absentes", "35,09 %", "29 800 produits"),
            ("Désignations en français", "60,92 %", "23,03 % en anglais"),
        ]
    )

    with card():
        st.markdown(
            "**Variables d'entrée :** `designation` (toujours renseignée), `description` (facultative), "
            "`productid` et `imageid` (lien vers l'image 500 × 500 px). **Cible :** `prdtypecode`, 27 codes."
        )

    classes_tab, text_tab, image_tab = st.tabs(
        ["Classes", "Texte et langues", "Images"]
    )
    with classes_tab:
        _class_distribution()
    with text_tab:
        left, right = st.columns(2)
        with left:
            figure("02_longueur_textes.png", "Longueur des désignations et des descriptions.")
        with right:
            figure("03_langues_designations.png", "Langues détectées dans les désignations.")
        left, right = st.columns(2)
        with left:
            figure("04_langue_par_classe.png", "Part de français par classe.")
        with right:
            figure("05_description_par_classe.png", "Taux de description renseignée par classe.")
        st.caption(
            "Désignations de 70 caractères en moyenne. Le multilinguisme a motivé l'essai d'un encodeur SBERT "
            "multilingue en complément du TF-IDF."
        )
    with image_tab:
        figure("07_dimensions_images.png", "Dimensions des images produit.")
        st.info(
            "Toutes les images sont présentes, en RGB et au format 500 × 500 px : le signal utile se trouve dans le "
            "contenu visuel, pas dans les métadonnées. Les images brutes du challenge, confidentielles, ne sont pas "
            "embarquées dans l'application."
        )

    st.subheader("Prétraitement du texte", anchor=False)
    steps = st.columns(3)
    steps[0].markdown(
        "**1. Nettoyage**  \nMinuscules, suppression des balises HTML, des URL et des caractères spéciaux."
    )
    steps[1].markdown(
        "**2. Normalisation**  \nStopwords français NLTK, tokens d'un caractère retirés, stemming Snowball."
    )
    steps[2].markdown(
        "**3. Assemblage**  \nDésignation + description, mots fréquents non discriminants retirés, repli sur la "
        "désignation si le texte est vide."
    )
    with card():
        st.markdown("**Essayez le prétraitement**")
        _preprocessing_preview()


render()
