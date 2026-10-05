"""Page équipe : membres du groupe et encadrement."""

from __future__ import annotations

import base64
import html

import streamlit as st

from core.content import DEFENSE_DATE, MENTOR, REPO_URL, TEAM, TRAINING
from core.paths import TEAM_PHOTOS_DIR
from core.ui import card, page_header

PHOTO_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@st.cache_data(show_spinner=False)
def _photo_data_uri(filename: str, mtime: float) -> str | None:
    """Encode un portrait local ; `mtime` invalide le cache quand le fichier change."""
    path = TEAM_PHOTOS_DIR / filename
    mime = PHOTO_TYPES.get(path.suffix.lower())
    if mime is None or not path.is_file():
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _portrait(member: dict) -> str:
    path = TEAM_PHOTOS_DIR / member["photo"]
    uri = _photo_data_uri(member["photo"], path.stat().st_mtime) if path.is_file() else None
    name = html.escape(member["name"])
    if uri:
        size = f"{member.get('scale', 1.0) * 100:.0f}%"
        image = f'<img src="{uri}" alt="Portrait de {name}" style="width:{size};height:{size}">'
    else:
        image = f'<div class="initials" role="img" aria-label="{name}">{html.escape(member["initials"])}</div>'
    return f'<div class="portrait">{image}</div>'


def render() -> None:
    page_header(
        "Projet fil rouge",
        "L'équipe",
        f"{TRAINING}. Soutenance du {DEFENSE_DATE}.",
    )
    for col, member in zip(st.columns(len(TEAM)), TEAM):
        with card(col, height="stretch"):
            st.markdown(
                f'<div class="member">{_portrait(member)}'
                f'<div class="name">{html.escape(member["name"])}</div>'
                '<div class="role">Membre du groupe projet</div></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Encadrement", anchor=False)
    st.markdown(f"{MENTOR}, mentor du projet.")
    st.markdown(f"Code source : [{REPO_URL.removeprefix('https://')}]({REPO_URL})")
    st.caption(
        "Les portraits servent uniquement à présenter l'équipe. Le modèle servi, TF-IDF + LinearSVC, "
        "prédit à partir du texte seul et n'utilise aucune image."
    )


render()
