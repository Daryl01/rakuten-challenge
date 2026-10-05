"""Page modélisation : approches testées et comparaison des scores."""

from __future__ import annotations

import streamlit as st

from core.content import (
    REPORTED_F1_PREVIOUS_RUN,
    SERVED_MODEL_F1,
    SERVED_MODEL_NAME,
    ContentUnavailableError,
    load_results,
)
from core.ui import card, figure, fr, model_comparison_chart, page_header

APPROACHES = [
    (
        "Baseline classique",
        "TF-IDF + LinearSVC",
        "Unigrammes et bigrammes, 200 000 features, min_df=2, sublinear_tf. LinearSVC C=1.",
        "0,8316",
    ),
    (
        "Représentation sémantique",
        "SBERT + MLP",
        "Embeddings paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions), MLP 512 → 256, 30 époques.",
        "0,7542",
    ),
    (
        "Transfer learning image",
        "ResNet50",
        "Phase 1 : backbone gelé (10 époques). Phase 2 : layer4 dégelée (15 époques).",
        "0,6799",
    ),
    (
        "Fusion multimodale",
        "SBERT + ResNet50",
        "Concaténation 384 + 2 048 = 2 432 dimensions, MLP 1024 → 512 → 256, dropout 0,4.",
        "0,7105",
    ),
]


def render() -> None:
    page_header(
        "Modélisation",
        "Quatre approches comparées",
        "Du plus simple au plus complexe, chaque brique est évaluée sur le même split de validation stratifié "
        "(20 %, 16 984 produits).",
    )

    cols = st.columns(4)
    for col, (kind, name, details, score) in zip(cols, APPROACHES):
        with card(col, height="stretch"):
            st.caption(kind)
            st.markdown(f"#### {name}")
            st.metric("F1 pondéré", score)
            st.caption(details)

    st.subheader("Comparaison avec les benchmarks Rakuten", anchor=False)
    try:
        results = load_results()
    except ContentUnavailableError as exc:
        st.error(str(exc))
        return
    st.altair_chart(model_comparison_chart(results, SERVED_MODEL_NAME), width="stretch")
    with st.expander("Tableau des scores"):
        st.dataframe(
            results[["Modèle", "Modalité", "Type", "Weighted F1"]],
            hide_index=True,
            width="stretch",
            column_config={"Weighted F1": st.column_config.NumberColumn("F1 pondéré", format="%.4f")},
        )
        st.caption("Source : `models/artifacts/resultats_comparaison.csv`, exporté par le notebook 02.")
    st.info(
        f"Les rapports écrits citent {fr(REPORTED_F1_PREVIOUS_RUN)} pour TF-IDF + LinearSVC : cette valeur provient "
        f"d'une exécution antérieure des notebooks. L'application affiche {fr(SERVED_MODEL_F1)}, score de la "
        "réexécution qui a produit le modèle réellement servi, recalculé et vérifié sur ce modèle.",
    )

    st.subheader("Pourquoi le TF-IDF l'emporte", anchor=False)
    left, right = st.columns(2)
    with left:
        st.markdown(
            "- Les désignations contiennent des marques, références et termes métier très discriminants.\n"
            "- Les bigrammes conservent cette précision lexicale sans compression.\n"
            "- LinearSVC est efficace sur une matrice creuse de grande dimension."
        )
    with right:
        st.markdown(
            "- SBERT résume le texte en 384 dimensions et lisse ces indices fins.\n"
            "- L'image apporte un signal plus variable (photos de vendeurs non professionnels).\n"
            "- La concaténation brute laisse dominer le vecteur image (2 048 dimensions) et sur-apprend vite."
        )

    st.subheader("Courbes d'apprentissage", anchor=False)
    sbert_tab, resnet_tab, fusion_tab = st.tabs(["SBERT + MLP", "ResNet50", "Fusion : matrice de confusion"])
    with sbert_tab:
        figure("10_learning_curve_mlp_sbert.png", "Loss d'entraînement et F1 de validation du MLP sur SBERT.")
        st.caption("Progression régulière jusqu'à l'époque 30, meilleur F1 de validation : 0,7542.")
    with resnet_tab:
        figure("11_learning_curve_resnet50.png", "ResNet50 : backbone gelé puis layer4 dégelée.")
        st.caption(
            "Phase 1 plafonnée à 0,6189. En phase 2, meilleur F1 de 0,6799 à l'époque 8 puis légère baisse : "
            "le meilleur checkpoint est conservé."
        )
    with fusion_tab:
        figure("13_confusion_matrix_fusion.png", "Matrice de confusion de la fusion SBERT + ResNet50.")
        st.caption("Meilleur F1 de validation 0,7105, sous le SBERT seul : ajouter une modalité ne suffit pas.")


render()
