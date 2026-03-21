# Projet Rakuten - Classification multimodale de produits e-commerce

> **Formation** : Ingénieur IA - Liora (ex DataScientest)  
> **Soutenance** : 16 octobre 2026  
> **Auteurs** : Khoty WOLIE, [Prénom NOM 2], [Prénom NOM 3]  
> **Mentor** : [Prénom NOM Mentor]

---

## Contexte

Ce projet s'inscrit dans le **challenge Rakuten France Multimodal Product Data Classification** ([challengedata.ens.fr/challenges/35](https://challengedata.ens.fr/challenges/35)).

L'objectif est de prédire le **code type produit** (`prdtypecode`, 27 classes) à partir de données textuelles (désignation + description) et d'images de produits, sur un catalogue de ~99 000 fiches.

**Métrique officielle** : score F1 pondéré (`sklearn`, `average='weighted'`)

---

## Résultats

| Modèle | Modalité | F1 pondéré | vs Benchmark |
|---|---|---|---|
| **TF-IDF + LinearSVC** | Texte | **0.8403** | **+0.029** |
| Benchmark Rakuten Texte (CNN) | Texte | 0.8113 | réf. |
| SBERT + MLP | Texte | 0.7542 | -0.057 |
| Fusion SBERT + ResNet50 | Texte + Image | 0.7105 | -0.101 |
| ResNet50 Fine-tuning | Image | 0.6836 | -0.128 |
| Benchmark Rakuten Image | Image | 0.5534 | -0.258 |

Le modèle **TF-IDF + LinearSVC** est le meilleur modèle du projet. Il dépasse le benchmark officiel Rakuten de **+2,9 points** de F1 pondéré.

---

## Arborescence du projet

```
project_rakuten_ml_dl/
│
├── data/
│   ├── raw/
│   │   ├── X_train_update.csv
│   │   ├── Y_train_CVw08PX.csv
│   │   ├── X_test_update.csv
│   │   ├── image_train/          # 84 916 images .jpg (500×500)
│   │   └── image_test/           # 13 812 images .jpg (500×500)
│   ├── processed/
│   │   └── df_train_processed.csv  # Dataset nettoyé (84 916 × 10 col.)
│   └── submissions/
│       └── submission_tfidf_svc.csv
│
├── models/
│   ├── baselines/
│   │   └── tfidf_linearsvc.pkl     # Pipeline TF-IDF + LinearSVC (joblib)
│   ├── text/
│   │   ├── mlp_sbert_best.pt       # MLP sur embeddings SBERT
│   │   └── mlp_fusion_best.pt      # MLP fusion texte + image
│   ├── image/
│   │   ├── resnet50_phase1_best.pt # ResNet50 phase 1 (classifieur seul)
│   │   └── resnet50_phase2_best.pt # ResNet50 phase 2 (layer4 dégelée)
│   └── artifacts/
│       ├── sbert_embeddings_train.npy
│       ├── sbert_labels_train.npy
│       ├── sbert_index_train.npy
│       ├── resnet50_features_train.npy
│       ├── images_224_uint8_train.npy  # Cache images redimensionnées
│       ├── resultats_comparaison.csv
│       └── resultats_finaux_03.csv
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb  # Exploration, DataViz, Preprocessing
│   ├── 02_modeling.ipynb           # Modélisation (baseline → deep learning)
│   └── 03_evaluation.ipynb         # Évaluation, interprétabilité, analyses
│
├── reports/
│   └── figures/                    # 20 figures PNG générées par les notebooks
│
├── references/                     # Articles et ressources bibliographiques
│
├── src/                            # Modules Python réutilisables
│
├── streamlit_app/
│   ├── app.py                      # Application Streamlit (5 onglets)
│   └── README.md
│
├── README.md                       # Ce fichier
└── requirements.txt                # Dépendances Python
```

---

## Installation

### Prérequis

- [Anaconda](https://www.anaconda.com/) ou Miniconda
- GPU NVIDIA avec CUDA 12.8+ (recommandé - le projet a été développé sur RTX 5070 Ti, architecture Blackwell)
- Windows 10/11 (les commandes ci-dessous sont testées sous Windows)

### Création de l'environnement

```bash
conda create -n rakuten_env python=3.10
conda activate rakuten_env
pip install -r requirements.txt
```

> **Note GPU** : le projet utilise PyTorch nightly cu128 pour la compatibilité avec l'architecture Blackwell (sm_120). Si vous utilisez un GPU d'une génération antérieure, remplacez la ligne `torch` dans `requirements.txt` par la version stable correspondant à votre version CUDA ([pytorch.org](https://pytorch.org/get-started/locally/)).

---

## Données

Les données sont issues du challenge Rakuten France et soumises à une **clause de confidentialité stricte**. Elles ne sont pas incluses dans ce dépôt.

Pour reproduire les expériences :

1. Télécharger les données sur [challengedata.ens.fr/challenges/35](https://challengedata.ens.fr/challenges/35)
2. Placer les fichiers CSV dans `data/raw/`
3. Décompresser les images dans `data/raw/image_train/` et `data/raw/image_test/`

**Volumétrie** :
- Données textuelles : ~60 Mo (CSV)
- Images : ~2,2 Go (JPEG)
- Dataset entraînement : 84 916 produits
- Dataset test : 13 812 produits
- Nombre de classes : 27

---

## Notebooks

Les notebooks doivent être exécutés dans l'ordre.

### `01_eda_preprocessing.ipynb` - Exploration et Preprocessing

- Analyse exploratoire complète du jeu de données
- Distribution des 27 classes (ratio déséquilibre max/min : 13,4x)
- Analyse du multilinguisme (61 % FR, 23 % EN, 6 % DE, 10 % autres)
- Traitement des valeurs manquantes (35,1 % en description)
- Construction de la colonne `text_combined` (désignation + description)
- Export de `data/processed/df_train_processed.csv`
- Génération des figures 01 à 07 dans `reports/figures/`

### `02_modeling.ipynb` - Modélisation

**Étape 1 - Baseline classique (TF-IDF + LinearSVC)**
- Vectorisation TF-IDF avec bigrammes (`ngram_range=(1,2)`)
- Pipeline sklearn sérialisé avec `joblib`
- F1 pondéré : **0.8403**

**Étape 2 - Représentation sémantique (SBERT + MLP)**
- Embeddings `paraphrase-multilingual-MiniLM-L12-v2` (384 dim)
- MLP 3 couches avec BatchNorm et Dropout
- F1 pondéré : 0.7542

**Étape 3 - Transfer learning image (ResNet50)**
- Fine-tuning progressif en 2 phases (classifieur → layer4)
- Optimiseur AdamW + OneCycleLR + label_smoothing=0.1
- F1 pondéré : 0.6836 (+13 pts vs benchmark image Rakuten)

**Étape 4 - Fusion multimodale (SBERT + ResNet50)**
- Concaténation normalisée L2 features texte (384 dim) + image (2048 dim)
- F1 pondéré : 0.7105

Génère les figures 08 à 13 dans `reports/figures/`.

### `03_evaluation.ipynb` - Évaluation et Interprétabilité

- Analyse détaillée des erreurs par classe
- Coefficients TF-IDF équivalents SHAP (analyse tokens discriminants)
- Comparaison F1 par classe sur les 3 modèles principaux
- Chevauchement des erreurs (diagramme de Venn) : plancher structurel à 7,6 %
- Distribution des prédictions sur le jeu de test
- Génère les figures 14 à 20 dans `reports/figures/`

---

## Application Streamlit

L'application permet de démontrer les capacités du système en temps réel.

### Lancement

```bash
conda activate rakuten_env
cd project_rakuten_ml_dl
streamlit run streamlit_app/app.py
```

L'application est accessible sur [http://localhost:8501](http://localhost:8501).

### Structure de l'application (5 onglets)

| Onglet | Contenu |
|---|---|
| **Contexte** | Problématique, métrique, stratégie de modélisation, environnement technique |
| **Exploration** | Statistiques descriptives, distributions, figures DataViz issues du notebook 01 |
| **Modélisation** | Tableau comparatif des modèles, matrices de confusion, courbes d'apprentissage |
| **Classification en direct** | Démo PoC : saisie libre ou exemples prédéfinis, prédiction instantanée, top 5 classes, tokens clés |
| **Interprétabilité** | Coefficients discriminants, analyse des erreurs, comparaison inter-modèles |

> Le modèle est chargé une seule fois en mémoire via `st.cache_resource`. Aucun réentraînement n'a lieu pendant la démonstration.

---

## Environnement technique

| Composant | Version / Détail |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core Ultra 9 275HX |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell, sm_120, 12 Go VRAM) |
| Driver CUDA | 576.65 |
| CUDA | 12.9 |
| Python | 3.10 |
| PyTorch | nightly 2.12.0.dev+cu128 |
| Conda env | rakuten_env |

**Contrainte Windows/Jupyter** : `num_workers=0` et `pin_memory=False` dans tous les DataLoaders (multiprocessing `spawn` incompatible avec Jupyter sous Windows).

---

## Bibliographie sélective

- Rakuten France (2020). *Rakuten France Multimodal Product Data Classification*. ENS Data Challenge.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
- Documentation sklearn : [scikit-learn.org](https://scikit-learn.org)
- Documentation sentence-transformers : [sbert.net](https://www.sbert.net)

---

## Licence et confidentialité

Les **données Rakuten** sont strictement confidentielles conformément aux conditions du challenge ENS Data. Elles ne peuvent pas être redistribuées, publiées ou utilisées à des fins commerciales.

Le **code source** de ce projet est produit dans le cadre de la formation Ingénieur IA - Liora (ex DataScientest).
