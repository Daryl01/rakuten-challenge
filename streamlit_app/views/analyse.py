"""Page analyse du meilleur modèle : erreurs par classe et interprétation."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.content import (
    SERVED_MODEL_F1,
    VALIDATION_ERRORS,
    VALIDATION_SIZE,
    ContentUnavailableError,
    label,
    load_class_metrics,
    load_confusions,
)
from core.model_service import ModelUnavailableError, load_model
from core.ui import ACCENT, INK, POSITIVE, card, figure, fr, kpis, page_header

F1_THRESHOLD = 0.70


def _f1_chart(metrics: pd.DataFrame) -> alt.Chart:
    data = metrics.assign(Difficile=metrics["f1"] < F1_THRESHOLD)
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=2, height=16)
        .encode(
            y=alt.Y("classe:N", sort="x", title=None, axis=alt.Axis(labelLimit=230)),
            x=alt.X("f1:Q", title="F1 (validation)", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum["Difficile"], alt.value(ACCENT), alt.value(INK)),
            tooltip=[
                alt.Tooltip("classe:N", title="Classe"),
                alt.Tooltip("f1:Q", title="F1", format=".4f"),
                alt.Tooltip("precision:Q", title="Précision", format=".4f"),
                alt.Tooltip("rappel:Q", title="Rappel", format=".4f"),
                alt.Tooltip("support_validation:Q", title="Produits en validation"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [F1_THRESHOLD]})).mark_rule(strokeDash=[4, 4], color=ACCENT).encode(x="x:Q")
    return (bars + rule).properties(height=alt.Step(23))


@st.cache_data(show_spinner=False)
def _top_terms(_pipeline, sha256: str, code: int, count: int = 12) -> pd.DataFrame:
    """Termes de plus fort coefficient positif pour une classe (calcul exact sur le modèle chargé)."""
    vectorizer = _pipeline.steps[0][1]
    classifier = _pipeline.steps[-1][1]
    index = int(np.flatnonzero(np.asarray(classifier.classes_) == code)[0])
    coefficients = np.asarray(classifier.coef_[index]).ravel()
    best = np.argsort(coefficients)[::-1][:count]
    names = vectorizer.get_feature_names_out()
    return pd.DataFrame({"Terme": names[best], "Coefficient": coefficients[best]})


def _class_terms(metrics: pd.DataFrame) -> None:
    try:
        model = load_model()
    except ModelUnavailableError as error:
        st.warning(f"{error.title} : {error.detail}")
        return
    options = metrics.sort_values("f1")["code"].tolist()
    code = st.selectbox("Classe", options, format_func=lambda c: f"{c} · {label(c)}")
    terms = _top_terms(model.pipeline, model.sha256, int(code))
    chart = (
        alt.Chart(terms)
        .mark_bar(color=POSITIVE, cornerRadiusEnd=2, height=15)
        .encode(
            y=alt.Y("Terme:N", sort="-x", title=None),
            x=alt.X("Coefficient:Q", title="Coefficient LinearSVC"),
            tooltip=["Terme", alt.Tooltip("Coefficient:Q", format=".3f")],
        )
        .properties(height=alt.Step(22))
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Termes après stemming. Un coefficient élevé pousse fortement vers la classe quand le terme est présent.")


def render() -> None:
    page_header(
        "Meilleur modèle",
        "Analyse de TF-IDF + LinearSVC",
        "Où le modèle réussit, où il se trompe, et sur quels mots il fonde ses décisions.",
    )
    try:
        metrics = load_class_metrics()
        confusions = load_confusions()
    except ContentUnavailableError as exc:
        st.error(str(exc))
        return

    difficult = metrics[metrics["f1"] < F1_THRESHOLD].sort_values("f1")
    kpis(
        [
            ("F1 pondéré", fr(SERVED_MODEL_F1), "validation"),
            ("Erreurs de validation", f"{VALIDATION_ERRORS:,}".replace(",", " "), f"sur {VALIDATION_SIZE:,} produits".replace(",", " ")),
            ("Taux d'erreur", f"{VALIDATION_ERRORS / VALIDATION_SIZE * 100:.1f} %".replace(".", ","), "erreurs / validation"),
            ("Classes sous 0,70 de F1", str(len(difficult)), "sur 27"),
        ]
    )

    st.subheader("F1 par classe", anchor=False)
    st.altair_chart(_f1_chart(metrics), width="stretch")
    st.caption("En rouge, les classes sous le seuil de 0,70. Survolez une barre pour la précision et le rappel.")

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("##### Classes difficiles")
        st.dataframe(
            difficult[["classe", "precision", "rappel", "f1"]],
            hide_index=True,
            width="stretch",
            column_config={
                "classe": "Classe",
                "precision": st.column_config.NumberColumn("Précision", format="%.3f"),
                "rappel": st.column_config.NumberColumn("Rappel", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            },
        )
    with right:
        st.markdown("##### Confusions les plus fréquentes")
        st.dataframe(confusions.head(6), hide_index=True, width="stretch")
    st.caption(
        "Les erreurs se concentrent entre catégories sémantiquement proches : livres, presse et magazines d'une part, "
        "jouets et jeux de société d'autre part."
    )

    st.subheader("Interprétabilité", anchor=False)
    with card():
        st.markdown("**Termes les plus discriminants par classe**")
        _class_terms(metrics)

    st.subheader("Figures d'analyse", anchor=False)
    matrix_tab, weak_tab, compare_tab, overlap_tab = st.tabs(
        ["Matrice de confusion", "Classes faibles", "F1 des trois modèles", "Erreurs communes"]
    )
    with matrix_tab:
        figure("08_confusion_matrix_baseline.png", "Matrice de confusion du TF-IDF + LinearSVC (validation).")
    with weak_tab:
        figure("15_confusion_classes_faibles.png", "Confusions normalisées des classes les plus faibles.")
    with compare_tab:
        figure("18_f1_par_classe_comparaison.png", "F1 par classe : TF-IDF + LinearSVC, SBERT + MLP et fusion.")
        st.caption("Certaines classes de livres profitent de l'image dans la fusion, la plupart profitent du TF-IDF.")
    with overlap_tab:
        figure("20_chevauchement_erreurs.png", "Chevauchement des erreurs des trois modèles.")
        st.caption("Une part des exemples est mal classée par les trois modèles : cas ambigus ou mal étiquetés.")


render()
