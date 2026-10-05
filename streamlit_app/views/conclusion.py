"""Page conclusion : enseignements, limites et perspectives."""

from __future__ import annotations

import streamlit as st

from core.content import SERVED_MODEL_F1
from core.ui import card, fr, page_header


def render() -> None:
    page_header(
        "Conclusion",
        "Ce que nous retenons",
        "Relier les résultats à la question métier, puis identifier ce qu'il faudrait faire avec plus de temps.",
    )
    with card():
        st.markdown("#### Décision proposée")
        st.write(
            f"Retenir TF-IDF + LinearSVC (F1 pondéré {fr(SERVED_MODEL_F1)}) comme référence opérationnelle : "
            "rapide, reproductible, entraîné sur CPU et servi ici sans GPU. Prévoir une revue humaine pour les "
            "catégories livres, presse et jeux, qui concentrent les confusions."
        )

    left, middle, right = st.columns(3)
    with card(left, height="stretch"):
        st.markdown("#### Enseignements")
        st.markdown(
            "- Le texte est le signal dominant sur ce catalogue.\n"
            "- La précision lexicale du TF-IDF dépasse des représentations plus complexes.\n"
            "- L'image aide sur certains livres, mais son apport moyen reste limité.\n"
            "- Fusionner deux modalités ne suffit pas : il faut les équilibrer."
        )
    with card(middle, height="stretch"):
        st.markdown("#### Limites")
        st.markdown(
            "- Scores LinearSVC non calibrés : pas de probabilité exploitable.\n"
            "- Quatre classes sous 0,70 de F1.\n"
            "- Chaînes image et fusion non empaquetées pour une inférence en ligne.\n"
            "- Libellés de catégories déduits par l'équipe, non fournis par Rakuten."
        )
    with card(right, height="stretch"):
        st.markdown("#### Perspectives")
        st.markdown(
            "- Calibrer les scores et fixer un seuil de revue humaine.\n"
            "- Normaliser chaque modalité avant la fusion.\n"
            "- Fine-tuner un modèle de langue (ex. CamemBERT) sur la tâche.\n"
            "- Rééquilibrage et augmentation de données pour les classes rares.\n"
            "- Empaqueter le pipeline multimodal complet."
        )

    st.warning(
        "Confidentialité : les données et images du challenge restent hors de l'application et du dépôt public. "
        "Seuls des agrégats (scores, effectifs par classe) et les figures du projet sont affichés.",
    )


render()
