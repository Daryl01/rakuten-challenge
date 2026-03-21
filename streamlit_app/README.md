# Application Streamlit - Projet Rakuten

Classification multimodale de produits e-commerce  
Formation Ingénieur IA — Liora | DataScientest | Soutenance 16 octobre 2026

\---

## Installation et lancement

```bash
# Activer l'environnement conda du projet
conda activate rakuten\\\_env

# Installer les dépendances si nécessaire
pip install -r requirements.txt

# Lancer l'application depuis la racine du projet
streamlit run streamlit\\\_app/app.py

streamlit run streamlit\_app/app.py
```

## Structure attendue du projet

L'application s'appuie sur les artefacts produits par les notebooks.
Elle doit être lancée depuis la racine du projet :

```
project\\\_rakuten\\\_ml\\\_dl/
├── streamlit\\\_app/
│   ├── app.py                  ← application principale
│   └── requirements.txt
├── data/
│   ├── raw/
│   │   ├── image\\\_train/        ← images d'entraînement (optionnel pour la démo)
│   │   └── image\\\_test/
│   └── processed/
│       └── df\\\_train\\\_processed.csv
├── models/
│   ├── baselines/
│   │   └── tfidf\\\_linearsvc.pkl ← REQUIS pour la démo de classification
│   └── artifacts/
│       └── resultats\\\_finaux\\\_03.csv
└── reports/
    └── figures/
        ├── 01\\\_distribution\\\_classes.png
        ├── 03\\\_longueur\\\_textes.png
        ├── 05\\\_repartition\\\_langues.png
        ├── 07\\\_taux\\\_description.png
        ├── 08\\\_confusion\\\_matrix\\\_baseline.png
        ├── 09\\\_f1\\\_par\\\_classe\\\_baseline.png
        ├── 10\\\_learning\\\_curve\\\_mlp\\\_sbert.png
        ├── 11\\\_learning\\\_curve\\\_resnet50.png
        ├── 12\\\_comparaison\\\_modeles.png
        ├── 13\\\_confusion\\\_matrix\\\_fusion.png
        ├── 15\\\_confusion\\\_classes\\\_faibles.png
        ├── 16\\\_shap\\\_tokens\\\_par\\\_classe.png
        ├── 17\\\_shap\\\_exemple\\\_individuel.png
        ├── 18\\\_f1\\\_par\\\_classe\\\_comparaison.png
        ├── 19\\\_erreurs\\\_classe\\\_10.png
        ├── 19\\\_erreurs\\\_classe\\\_1281.png
        └── 20\\\_chevauchement\\\_erreurs.png
```

## Fichier requis en priorité

* `models/baselines/tfidf\\\_linearsvc.pkl` : produit par le notebook 02, section 2.7.
Sans ce fichier, l'onglet "Classification en direct" affiche un avertissement
mais les 4 autres onglets restent pleinement fonctionnels.

## Onglets de l'application

|Onglet|Contenu|
|-|-|
|Contexte|Présentation du projet, données, stratégie de modélisation|
|Exploration|Distribution des classes, langues, longueur des textes, images|
|Modélisation|Tableau comparatif, figures des 4 modèles|
|Classification en direct|Démo PoC — saisie libre + exemples prédéfinis|
|Interprétabilité|SHAP, analyse d'erreurs, chevauchement des modèles|

## Notes pour la soutenance

* Le modèle n'est pas ré-entraîné au démarrage : chargement du `.pkl` uniquement.
* L'inférence est instantanée (< 10 ms) — aucun risque de chargement long.
* En cas de fichiers figures manquants, l'application génère les graphiques
dynamiquement depuis le dataset (si `df\\\_train\\\_processed.csv` est disponible).

