"""Page PoC : classification d'une fiche produit avec le modèle sauvegardé."""

from __future__ import annotations

import html
import logging

import altair as alt
import pandas as pd
import streamlit as st

from core.content import EXAMPLES, SERVED_MODEL_F1, SERVED_MODEL_NAME, label
from core.model_service import ModelUnavailableError, load_model, predict
from core.ui import ACCENT, MUTED, POSITIVE, card, fr, page_header

LOGGER = logging.getLogger(__name__)
MAX_CHARS = 5_000
HISTORY_SIZE = 6


def _init_state() -> None:
    st.session_state.setdefault("designation", "")
    st.session_state.setdefault("description", "")
    st.session_state.setdefault("example_name", None)
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("history", [])


def _fill_example(name: str) -> None:
    st.session_state["designation"] = EXAMPLES[name]["designation"]
    st.session_state["description"] = ""
    st.session_state["example_name"] = name
    st.session_state["prediction"] = None


def _reset() -> None:
    st.session_state["designation"] = ""
    st.session_state["description"] = ""
    st.session_state["example_name"] = None
    st.session_state["example_pills"] = None
    st.session_state["prediction"] = None


def _on_example_selected() -> None:
    if st.session_state.get("example_pills"):
        _fill_example(st.session_state["example_pills"])


def show_model_error(error: ModelUnavailableError) -> None:
    st.error(f"**{error.title}** : {error.detail}")
    st.markdown(f"**Que faire ?** {error.hint}")
    st.caption("Aucune prédiction de remplacement n'est affichée tant que le modèle réel n'est pas chargé.")


def _ranking_chart(ranking: pd.DataFrame) -> alt.Chart:
    data = ranking.assign(
        Libellé=ranking["Code"].astype(str) + " · " + ranking["Catégorie"],
        Retenue=ranking["Rang"].eq(1).map({True: "Classe prédite", False: "Autres classes"}),
        Score=ranking["Score de décision"].map(lambda v: fr(v, 3)),
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=20)
        .encode(
            y=alt.Y("Libellé:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Score de décision:Q", title="Score de décision LinearSVC"),
            color=alt.Color(
                "Retenue:N",
                scale=alt.Scale(domain=["Classe prédite", "Autres classes"], range=[ACCENT, MUTED]),
                legend=None,
            ),
            tooltip=["Rang", "Code", "Catégorie", "Score"],
        )
        .properties(height=alt.Step(30))
    )


def _contribution_chart(contributions: pd.DataFrame) -> alt.Chart:
    data = contributions.head(12).assign(
        Sens=lambda d: d["Contribution"].gt(0).map({True: "Favorise la classe", False: "Défavorise la classe"})
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadius=2, height=16)
        .encode(
            y=alt.Y("Terme:N", sort="-x", title=None),
            x=alt.X("Contribution:Q", title="Coefficient × valeur TF-IDF"),
            color=alt.Color(
                "Sens:N",
                scale=alt.Scale(domain=["Favorise la classe", "Défavorise la classe"], range=[POSITIVE, ACCENT]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["Terme", alt.Tooltip("Contribution:Q", format=".3f")],
        )
        .properties(height=alt.Step(24))
    )


def _render_result(result: dict) -> None:
    prediction = result["prediction"]
    st.markdown(
        f'<div class="result"><div class="code">Catégorie prédite · prdtypecode {prediction.code}</div>'
        f'<div class="name">{html.escape(prediction.name)}</div></div>',
        unsafe_allow_html=True,
    )
    example = result.get("example")
    if example:
        expected = EXAMPLES[example]["code"]
        verdict = "concorde avec" if expected == prediction.code else "diffère de"
        st.caption(
            f"Fiche réelle du jeu d'entraînement. La prédiction {verdict} son code d'origine "
            f"({expected} · {label(expected)}). Ce cas est illustratif, pas une mesure de performance."
        )
    st.caption(
        f"Calcul en {prediction.seconds * 1000:.0f} ms. Le LinearSVC ne fournit pas de probabilité : "
        "les scores ci-dessous sont des scores de décision relatifs, non calibrés, et ne sont pas "
        "présentés comme un niveau de confiance."
    )

    ranking_tab, words_tab, text_tab = st.tabs(
        ["Top 5 des classes", "Mots décisifs", "Texte transmis"]
    )
    with ranking_tab:
        st.altair_chart(_ranking_chart(prediction.ranking), width="stretch")
        st.caption("Plus le score est élevé, plus le texte se situe du côté de la classe dans l'espace TF-IDF.")
    with words_tab:
        if prediction.contributions.empty:
            st.info("Aucun terme du texte n'appartient au vocabulaire appris : la décision repose sur les intercepts.")
        else:
            st.altair_chart(_contribution_chart(prediction.contributions), width="stretch")
            st.caption(
                "Contribution additive locale au score de la classe prédite (coefficient × TF-IDF, hors intercept). "
                "C'est une lecture exacte du modèle linéaire, pas un calcul SHAP."
            )
    with text_tab:
        st.code(prediction.prepared.text_combined, language=None, wrap_lines=True)
        st.caption("Minuscules, suppression HTML/URL/ponctuation, stopwords français, stemming et mots retirés après l'EDA.")


def render() -> None:
    _init_state()
    page_header(
        "Proof of Concept",
        "Classer une fiche produit",
        "Saisissez une désignation : le modèle sauvegardé applique le prétraitement d'entraînement puis prédit "
        "la catégorie, sans aucun réentraînement.",
    )

    try:
        with st.spinner("Chargement du modèle TF-IDF + LinearSVC…"):
            model = load_model()
    except ModelUnavailableError as error:
        show_model_error(error)
        return

    with card():
        cols = st.columns([1.3, 1, 1])
        cols[0].markdown(f"**Modèle actif :** {SERVED_MODEL_NAME}")
        cols[1].markdown(f"**F1 pondéré :** {fr(SERVED_MODEL_F1)}")
        cols[2].markdown("**Statut :** chargé")
        for message in model.version_warnings:
            st.warning(message)

    st.markdown("##### Exemples réels")
    st.pills(
        "Exemples",
        list(EXAMPLES),
        key="example_pills",
        label_visibility="collapsed",
        on_change=_on_example_selected,
    )

    with st.form("prediction_form", border=True):
        st.text_area(
            "Désignation du produit",
            key="designation",
            height=80,
            max_chars=MAX_CHARS,
            placeholder="Ex. : Piscine tubulaire ronde 366 x 76 cm avec pompe de filtration",
        )
        st.text_area(
            "Description (facultative)",
            key="description",
            height=110,
            max_chars=MAX_CHARS,
            placeholder="Texte libre du vendeur, si disponible.",
        )
        submit_col, reset_col, _ = st.columns([1, 1, 2])
        submitted = submit_col.form_submit_button(
            "Prédire la catégorie", type="primary", width="stretch"
        )
        reset_col.form_submit_button("Effacer", on_click=_reset, width="stretch")

    if submitted:
        designation = st.session_state["designation"]
        description = st.session_state["description"]
        if not designation.strip() and not description.strip():
            st.warning("Saisissez au moins une désignation ou une description.")
            st.session_state["prediction"] = None
        else:
            example = st.session_state["example_name"]
            if example and EXAMPLES[example]["designation"] != designation:
                example = None
            try:
                result = predict(model, designation, description)
            except ValueError as exc:
                st.warning(str(exc))
                st.session_state["prediction"] = None
            except Exception:  # noqa: BLE001 - message utilisateur, trace dans les logs
                LOGGER.exception("Échec de la prédiction")
                st.error("La prédiction a échoué de manière inattendue. Le détail est consigné dans les logs du serveur.")
                st.session_state["prediction"] = None
            else:
                st.session_state["prediction"] = {"prediction": result, "example": example}
                st.session_state["history"] = (
                    [{"Désignation": designation[:80], "Code": result.code, "Catégorie": result.name}]
                    + st.session_state["history"]
                )[:HISTORY_SIZE]
                st.toast(f"Catégorie prédite : {result.code} · {result.name}")

    if st.session_state["prediction"]:
        _render_result(st.session_state["prediction"])

    if st.session_state["history"]:
        with st.expander(f"Historique de la session ({len(st.session_state['history'])})"):
            st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True, width="stretch")

    with st.expander("Diagnostic du modèle"):
        integrity = "conforme à l'artefact validé" if model.sha256_matches else "différent de l'artefact validé"
        st.markdown(
            f"- Chargé en {model.load_seconds:.2f} s, une seule fois par processus (`st.cache_resource`).\n"
            f"- SHA-256 `{model.sha256[:16]}…`, {integrity}.\n"
            "- Pipeline : `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200 000, sublinear_tf=True)` "
            "puis `LinearSVC(C=1.0)`."
        )
    st.caption(
        "Les libellés de catégorie sont indicatifs : le challenge ne fournit que les codes prdtypecode. "
        "Les modèles image et fusion ne sont pas servis car leur chaîne d'inférence complète n'est pas empaquetée."
    )


render()
