"""Preprocessing textuel partagé avec l'application Streamlit."""

from __future__ import annotations

import re
import math
import tempfile
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer


CUSTOM_STOPWORDS = {
    "the", "39", "01", "of", "cm", "and", "oui", "gi", "tai", "taill",
    "in", "oh", "10", "not", "plus", "sous", "vf", "11", "to", "34",
    "03", "fr", "12", "for", "09", "or", "04",
}


@dataclass(frozen=True)
class PreparedText:
    """Textes intermédiaires et entrée finale du modèle."""

    designation_clean: str
    description_clean: str
    text_concat: str
    text_combined: str


@lru_cache(maxsize=1)
def _language_resources() -> tuple[set[str], FrenchStemmer]:
    """Charge une seule fois les stopwords et le stemmer français NLTK."""
    try:
        french_stopwords = set(stopwords.words("french"))
    except LookupError:
        download_dir = Path(tempfile.gettempdir()) / "rakuten_nltk_data"
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded = nltk.download("stopwords", download_dir=str(download_dir), quiet=True)
        if not downloaded:
            raise RuntimeError(
                "Le corpus NLTK stopwords est indisponible. "
                "Exécutez nltk.download('stopwords') dans l'environnement de l'application."
            )
        if str(download_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(download_dir))
        french_stopwords = set(stopwords.words("french"))
    return french_stopwords, FrenchStemmer()


def clean_text(value: object) -> str:
    """Nettoie, filtre et stemme un champ comme dans le notebook 01."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    french_stopwords, stemmer = _language_resources()
    tokens = [token for token in text.split() if token not in french_stopwords and len(token) > 1]
    return " ".join(stemmer.stem(token) for token in tokens)


def remove_custom_stopwords(text: str) -> str:
    """Retire la liste de tokens définie après l'analyse exploratoire."""
    return " ".join(token for token in text.split() if token not in CUSTOM_STOPWORDS)


def prepare_text_for_model(designation: object, description: object = "") -> PreparedText:
    """Construit les champs texte attendus par le modèle entraîné."""
    designation_clean = clean_text(designation)
    description_clean = clean_text(description)
    text_concat = remove_custom_stopwords(f"{designation_clean} {description_clean}".strip())
    text_combined = text_concat if text_concat.strip() else designation_clean
    return PreparedText(designation_clean, description_clean, text_concat, text_combined)
