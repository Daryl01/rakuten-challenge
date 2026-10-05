# Application Streamlit, projet Rakuten

Application de soutenance : présentation du projet et démonstration (PoC) de classification de produits
parmi 27 catégories avec le modèle validé **TF-IDF + LinearSVC** (F1 pondéré de validation 0,8316).

## Lancement local

Depuis la racine du projet :

```bash
conda activate rakuten_env
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Le corpus NLTK `stopwords` est téléchargé au premier chargement du modèle s'il est absent.

## Structure

```text
streamlit_app/
├── app.py                 # point d'entrée : configuration, navigation, barre latérale
├── core/
│   ├── paths.py           # chemins absolus, indépendants du répertoire de lancement
│   ├── content.py         # scores validés, libellés, équipe, chargement des CSV
│   ├── model_service.py   # chargement contrôlé du modèle et prédiction
│   └── ui.py              # styles et composants partagés
├── views/                 # une page par fichier
├── data/                  # agrégats par classe recalculés avec le modèle servi
└── requirements.txt       # dépendances figées utilisées en production
```

Le prétraitement est importé depuis `src/text_preprocessing.py`, identique à celui du notebook 01.

## Artefacts

| Fichier | Rôle | Obligatoire |
|---|---|---|
| `models/baselines/tfidf_linearsvc.pkl` | Pipeline TF-IDF + LinearSVC servi (51 357 216 octets, SHA-256 `35ec63de…07e750cc`) | Oui, pour la prédiction |
| `models/artifacts/resultats_comparaison.csv` | Scores F1 de tous les modèles | Oui, page Modélisation |
| `streamlit_app/data/*.csv` | F1 par classe, effectifs, principales confusions | Oui, pages Données et Analyse |
| `reports/figures/*.png` | Figures des notebooks | Non, un message remplace une figure absente |

Les checkpoints SBERT, ResNet50 et fusion ne sont pas chargés : leur chaîne d'inférence complète
(encodeurs, transformations, ordre des features) n'est pas empaquetée.

Si le modèle est absent, illisible, remplacé par un pointeur Git LFS ou incompatible avec la version de
scikit-learn, l'application affiche la cause et la marche à suivre. Elle ne produit jamais de prédiction
de remplacement. La variable d'environnement `RAKUTEN_MODEL_PATH` permet de pointer vers un autre emplacement.

## Déploiement sur Streamlit Community Cloud

- Fichier principal : `streamlit_app/app.py`.
- Dépendances : `streamlit_app/requirements.txt`. Community Cloud lit en priorité le fichier situé dans le
  dossier du point d'entrée, ce qui évite le `requirements.txt` racine (PyTorch nightly CUDA, non installable).
- Python : 3.10 si proposé (version locale), sinon 3.11 ou 3.12. Les versions figées ne supportent pas 3.14.
- Le modèle doit être publié dans le dépôt : il est exclu par `*.pkl`, une exception dédiée figure dans `.gitignore`.

## Interprétation locale

La contribution affichée vaut `coefficient × valeur TF-IDF` pour la classe prédite, hors intercept. C'est une
lecture exacte du modèle linéaire, pas une valeur SHAP. Les scores de décision LinearSVC ne sont pas des
probabilités : aucun pourcentage de confiance n'est affiché.
