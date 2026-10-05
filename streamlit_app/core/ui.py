"""Composants d'interface partagés entre les pages."""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from core.paths import FIGURES_DIR

INK = "#14213d"
ACCENT = "#bf1e2e"
MUTED = "#9aa5b8"
POSITIVE = "#2a6f97"

CSS = """
<style>
.block-container {padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.kicker {color: #5a6578; font-size: .78rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; margin-bottom: .1rem;}
.lead {color: #4a5568; font-size: 1.05rem; max-width: 760px; margin-top: -.4rem;}
.intro {background: #f3f5f9; border: 1px solid #e3e8ef; border-radius: 12px; padding: 1.6rem 1.8rem; margin: .4rem 0 1.4rem;}
.intro h1 {font-size: 2rem; line-height: 1.2; margin: .2rem 0 .6rem; padding: 0;}
.intro p {color: #4a5568; font-size: 1.04rem; max-width: 760px; margin: 0;}
.result {border: 1px solid #dfe4ec; border-left: 5px solid #bf1e2e; background: #fff; border-radius: 10px; padding: 1rem 1.3rem;}
.result .code {color: #4a5568; font-size: .9rem;}
.result .name {color: #14213d; font-size: 1.6rem; font-weight: 700; line-height: 1.2; margin: .15rem 0;}
.step-num {color: #5a6578; font-weight: 700; font-size: .78rem; letter-spacing: .06em;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .75rem; margin: .3rem 0 1.2rem;}
.kpi {border: 1px solid #dfe4ec; border-radius: .6rem; padding: .85rem 1rem; background: #fff;}
.kpi .label {color: #4a5568; font-size: .85rem;}
.kpi .value {color: #14213d; font-size: 1.75rem; font-weight: 700; line-height: 1.25; font-variant-numeric: tabular-nums;}
.kpi .note {color: #6b7689; font-size: .8rem;}
.member {text-align: center; padding: .3rem 0 .1rem;}
.portrait {width: 168px; height: 168px; margin: 0 auto .9rem; border-radius: 10px; overflow: hidden;
    border: 1px solid #dfe4ec; background: #ffffff; display: flex; align-items: flex-end; justify-content: center;}
.portrait img {display: block; object-fit: contain;}
.portrait .initials {width: 100%; height: 100%; background: #f3f5f9; color: #4a5568; font-size: 2rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;}
.member .name {color: #14213d; font-size: 1.1rem; font-weight: 700;}
.member .role {color: #6b7689; font-size: .85rem;}
@media (max-width: 640px) {
    .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 3.6rem;}
    .intro {padding: 1.2rem 1.1rem;}
    .intro h1 {font-size: 1.5rem;}
    .result .name {font-size: 1.3rem;}
}
</style>
"""


def inject_css() -> None:
    st.session_state["_card_index"] = 0
    st.html(CSS)


def card(parent=st, **kwargs):
    """Conteneur bordé. La clé suit l'ordre d'affichage, stable d'une exécution à l'autre."""
    st.session_state["_card_index"] = st.session_state.get("_card_index", 0) + 1
    return parent.container(border=True, key=f"card-{st.session_state['_card_index']}", **kwargs)


def page_header(kicker: str, title: str, lead: str) -> None:
    st.markdown(f'<div class="kicker">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<p class="lead">{html.escape(lead)}</p>', unsafe_allow_html=True)


def hero(kicker: str, title: str, text: str) -> None:
    st.markdown(
        f'<div class="intro"><div class="kicker">{html.escape(kicker)}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>",
        unsafe_allow_html=True,
    )


def kpis(items: list[tuple[str, str, str]]) -> None:
    """Grille d'indicateurs qui passe automatiquement de 4 à 2 puis 1 colonne."""
    cards = []
    for title, value, note in items:
        cards.append(
            f'<div class="kpi"><div class="label">{html.escape(title)}</div>'
            f'<div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>'
        )
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def fr(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def figure(filename: str, caption: str) -> None:
    path = FIGURES_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure non disponible sur ce serveur : `reports/figures/{filename}`.")


def model_comparison_chart(results: pd.DataFrame, highlight: str) -> alt.Chart:
    data = results.copy()
    data["Couleur"] = data.apply(
        lambda r: "Modèle servi" if r["Modèle"] == highlight else r["Type"], axis=1
    )
    data["F1"] = data["Weighted F1"].map(lambda v: fr(v))
    order = data.sort_values("Weighted F1", ascending=False)["Modèle"].tolist()
    color = alt.Color(
        "Couleur:N",
        scale=alt.Scale(
            domain=["Modèle servi", "Modèle du projet", "Benchmark officiel"],
            range=[ACCENT, INK, MUTED],
        ),
        legend=alt.Legend(title=None, orient="bottom"),
    )
    base = alt.Chart(data).encode(
        y=alt.Y("Modèle:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Weighted F1:Q", title="F1 pondéré (validation)", scale=alt.Scale(domain=[0, 0.95])),
        tooltip=["Modèle", "Modalité", "F1"],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=22).encode(color=color)
    text = base.mark_text(align="left", dx=4, color=INK).encode(text="F1:N")
    return (bars + text).properties(height=alt.Step(34))
