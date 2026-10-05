"""Chargement contrôlé du modèle servi et prédiction texte.

Le modèle n'est jamais remplacé par une prédiction simulée : si l'artefact est
absent ou illisible, une ModelUnavailableError décrit la cause et la marche à suivre.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from core.content import LABEL_NAMES, label
from core.paths import MODEL_PATH, relative_to_project
from src.text_preprocessing import PreparedText, _language_resources, prepare_text_for_model

LOGGER = logging.getLogger(__name__)

EXPECTED_SHA256 = "35ec63dec937455c575780e5ae6e83e8d3135fb36a585d028640cd7b07e750cc"
EXPECTED_SIZE = 51_357_216
TRAINING_SKLEARN_VERSION = "1.7.2"
LFS_POINTER_PREFIX = b"version https://git-lfs"


class ModelUnavailableError(RuntimeError):
    """Erreur de chargement présentable à l'utilisateur."""

    def __init__(self, title: str, detail: str, hint: str) -> None:
        super().__init__(f"{title} : {detail}")
        self.title = title
        self.detail = detail
        self.hint = hint


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path
    exists: bool
    size: int = 0
    is_lfs_pointer: bool = False

    @property
    def size_mb(self) -> float:
        return self.size / 1_000_000


@dataclass
class LoadedModel:
    pipeline: object
    load_seconds: float
    sha256: str
    version_warnings: list[str] = field(default_factory=list)

    @property
    def sha256_matches(self) -> bool:
        return self.sha256 == EXPECTED_SHA256


@dataclass(frozen=True)
class Prediction:
    code: int
    name: str
    ranking: pd.DataFrame
    contributions: pd.DataFrame
    prepared: PreparedText
    seconds: float


def inspect_artifact(path: Path = MODEL_PATH) -> ArtifactInfo:
    """Vérification légère, sans désérialiser le modèle."""
    if not path.is_file():
        return ArtifactInfo(path, exists=False)
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(len(LFS_POINTER_PREFIX))
    return ArtifactInfo(path, exists=True, size=size, is_lfs_pointer=head == LFS_POINTER_PREFIX)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def load_model(path_str: str = str(MODEL_PATH)) -> LoadedModel:
    """Charge une seule fois le pipeline par processus. Les erreurs ne sont pas mises en cache."""
    path = Path(path_str)
    info = inspect_artifact(path)
    shown = relative_to_project(path)
    if not info.exists:
        raise ModelUnavailableError(
            "Modèle introuvable",
            f"le fichier `{shown}` n'existe pas sur ce serveur.",
            "Ce fichier est exclu par la règle `*.pkl` du `.gitignore` : il n'est donc pas publié sur GitHub "
            "et n'existe pas sur la plateforme de déploiement. Publiez-le avec l'exception prévue dans le "
            "`.gitignore`, ou définissez `RAKUTEN_MODEL_PATH` vers un emplacement valide.",
        )
    if info.is_lfs_pointer:
        raise ModelUnavailableError(
            "Pointeur Git LFS au lieu du modèle",
            f"`{shown}` ne pèse que {info.size} octets : la plateforme a récupéré le pointeur LFS, pas le fichier.",
            "Publiez le fichier directement dans Git (il pèse moins de 50 Mio) ou utilisez un stockage compatible.",
        )

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pipeline = joblib.load(path)
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            raise ModelUnavailableError(
                "Modèle illisible",
                f"`{shown}` est tronqué ou corrompu ({type(exc).__name__}: {exc}).",
                f"Comparez sa taille ({info.size} octets) et son SHA-256 avec l'original ({EXPECTED_SIZE} octets).",
            ) from exc
        except (ImportError, AttributeError, TypeError) as exc:
            raise ModelUnavailableError(
                "Versions incompatibles",
                f"le modèle ne peut pas être désérialisé avec scikit-learn {sklearn.__version__} "
                f"({type(exc).__name__}: {exc}).",
                f"Installez les versions de `streamlit_app/requirements.txt` (scikit-learn=={TRAINING_SKLEARN_VERSION}).",
            ) from exc
    version_warnings = [str(w.message) for w in caught if "version" in str(w.message).lower()]
    if sklearn.__version__ != TRAINING_SKLEARN_VERSION:
        version_warnings.append(
            f"scikit-learn {sklearn.__version__} installé, modèle sérialisé avec {TRAINING_SKLEARN_VERSION}."
        )

    missing = [name for name in ("predict", "decision_function", "steps") if not hasattr(pipeline, name)]
    classes = set(int(c) for c in getattr(pipeline, "classes_", []))
    if missing or classes != set(LABEL_NAMES):
        raise ModelUnavailableError(
            "Artefact inattendu",
            f"`{shown}` n'est pas le pipeline TF-IDF + LinearSVC attendu (27 classes).",
            "Vérifiez que le fichier provient du notebook 02 (cellule `joblib.dump`).",
        )

    try:
        _language_resources()
    except (LookupError, RuntimeError, OSError) as exc:
        raise ModelUnavailableError(
            "Ressource NLTK indisponible",
            f"les stopwords français nécessaires au prétraitement n'ont pas pu être chargés ({exc}).",
            "Le serveur doit pouvoir télécharger le corpus NLTK `stopwords` au premier lancement.",
        ) from exc

    elapsed = time.perf_counter() - started
    LOGGER.info("Modèle chargé depuis %s en %.2fs", shown, elapsed)
    return LoadedModel(pipeline, elapsed, _sha256(path), version_warnings)


def _contributions(pipeline, prepared: PreparedText, class_index: int) -> pd.DataFrame:
    """Contribution additive locale coefficient x TF-IDF pour une classe, hors intercept."""
    vectorizer = pipeline.steps[0][1]
    classifier = pipeline.steps[-1][1]
    row = vectorizer.transform([prepared.text_combined]).getrow(0)
    coefficients = np.asarray(classifier.coef_[class_index, row.indices]).ravel()
    names = vectorizer.get_feature_names_out()[row.indices]
    frame = pd.DataFrame({"Terme": names, "Contribution": row.data * coefficients})
    return frame.reindex(frame["Contribution"].abs().sort_values(ascending=False).index)


def predict(model: LoadedModel, designation: str, description: str = "", top_k: int = 5) -> Prediction:
    started = time.perf_counter()
    prepared = prepare_text_for_model(designation, description)
    if not prepared.text_combined.strip():
        raise ValueError(
            "Après nettoyage, le texte ne contient plus aucun mot exploitable "
            "(seulement des mots vides, de la ponctuation ou des caractères isolés)."
        )
    pipeline = model.pipeline
    scores = np.asarray(pipeline.decision_function([prepared.text_combined])).reshape(-1)
    classes = np.asarray(pipeline.classes_)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    code = int(classes[best])
    if code != int(pipeline.predict([prepared.text_combined])[0]):
        raise RuntimeError("Incohérence entre predict et decision_function.")
    ranking = pd.DataFrame(
        {
            "Rang": range(1, top_k + 1),
            "Code": [int(classes[i]) for i in order[:top_k]],
            "Catégorie": [label(classes[i]) for i in order[:top_k]],
            "Score de décision": [float(scores[i]) for i in order[:top_k]],
        }
    )
    return Prediction(
        code=code,
        name=label(code),
        ranking=ranking,
        contributions=_contributions(pipeline, prepared, best),
        prepared=prepared,
        seconds=time.perf_counter() - started,
    )
