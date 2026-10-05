"""Page d'accueil : problème, enjeu et parcours de la soutenance."""

from __future__ import annotations

import streamlit as st

from core.content import SERVED_MODEL_F1, SERVED_MODEL_NAME, TEXT_BENCHMARK_F1
from core.ui import card, fr, hero, kpis


def render() -> None:
    hero(
        "Challenge Rakuten France · ENS Challenge Data",
        "Classer automatiquement 27 catégories de produits e-commerce",
        "À partir de la désignation, de la description et de l'image d'un produit, prédire son code type "
        "(prdtypecode). Nous comparons des approches texte, image et multimodales pour mesurer ce que chaque "
        "modalité apporte réellement.",
    )

    kpis(
        [
            ("Produits d'entraînement", "84 916", "13 812 en test"),
            ("Catégories", "27", "codes prdtypecode"),
            ("F1 pondéré du modèle servi", fr(SERVED_MODEL_F1), "validation, 16 984 produits"),
            ("Écart au benchmark texte", f"+{fr(SERVED_MODEL_F1 - TEXT_BENCHMARK_F1)}", f"référence Rakuten {fr(TEXT_BENCHMARK_F1)}"),
        ]
    )

    left, right = st.columns(2)
    with card(left, height="stretch"):
        st.markdown("#### Enjeu métier")
        st.write(
            "Sur une marketplace, des vendeurs professionnels et particuliers publient des fiches hétérogènes. "
            "Un catalogage automatique fiable améliore la recherche, la recommandation et la qualité du catalogue, "
            "tout en réduisant le classement manuel."
        )
    with card(right, height="stretch"):
        st.markdown("#### Difficultés")
        st.markdown(
            "- 27 classes déséquilibrées (rapport 13,4 entre la plus grande et la plus petite)\n"
            "- 35 % des descriptions absentes, textes multilingues et bruités\n"
            "- Catégories proches : livres, presse, jeux et jouets"
        )

    st.subheader("Démarche", anchor=False)
    steps = [
        ("01", "Exploration", "Volumétrie, déséquilibre, langues, images."),
        ("02", "Prétraitement", "Nettoyage, stopwords, stemming, text_combined."),
        ("03", "Modélisation", "TF-IDF, SBERT, ResNet50, fusion."),
        ("04", "Évaluation", "F1 pondéré, erreurs, interprétation."),
    ]
    for col, (num, title, text) in zip(st.columns(4), steps):
        with card(col, height="stretch"):
            st.markdown(f'<div class="step-num">ÉTAPE {num}</div>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.caption(text)

    st.success(
        f"Résultat principal : le modèle texte **{SERVED_MODEL_NAME}** obtient le meilleur F1 pondéré de validation "
        f"({fr(SERVED_MODEL_F1)}), devant les réseaux texte, image et la fusion multimodale.",
    )

    st.subheader("Parcours de la présentation", anchor=False)
    links = st.columns(3)
    links[0].page_link("views/donnees.py", label="Explorer les données →")
    links[1].page_link("views/modelisation.py", label="Comparer les modèles →")
    links[2].page_link("views/demo.py", label="Tester la prédiction →")


render()
