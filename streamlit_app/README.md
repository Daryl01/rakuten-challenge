# Application Streamlit, projet Rakuten

Application de démonstration pour la classification de produits e-commerce parmi 27 catégories.
Elle présente les résultats texte, image et multimodaux. L'inférence libre utilise le meilleur
modèle validé, TF-IDF + LinearSVC, avec un F1 pondéré de 0,8316.

## Lancement local

Depuis la racine du projet :

```bash
conda activate rakuten_env
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Le corpus français NLTK `stopwords` est téléchargé lors de la première prédiction s'il n'est pas
déjà présent.

## Navigation

- Accueil : contexte, chiffres clés et parcours du projet.
- Démonstration : prédiction texte, Top 3, image facultative et contributions locales.
- Performances : modèles, courbes d'apprentissage, classes difficiles et confusions.
- Comprendre le modèle : pipeline, artefacts, interprétation et limites.

## Artefacts

Artefact requis pour la prédiction directe :

```text
models/baselines/tfidf_linearsvc.pkl
```

Fichiers utilisés pour la présentation :

```text
models/artifacts/resultats_comparaison.csv
reports/figures/08_confusion_matrix_baseline.png
reports/figures/10_learning_curve_mlp_sbert.png
reports/figures/11_learning_curve_resnet50.png
reports/figures/17_shap_exemple_individuel.png
```

Les checkpoints `mlp_sbert_best.pt`, `resnet50_phase2_best.pt` et `mlp_fusion_best.pt` sont détectés
et présentés. Ils ne sont pas chargés au démarrage et ne sont pas proposés pour une inférence libre,
car leurs pipelines complets nécessitent des dépendances et transformations supplémentaires.

## Interprétation locale

L'application calcule une contribution additive locale au score de décision du LinearSVC avec
`coefficient × valeur TF-IDF`, hors intercept. Cette quantité n'est pas une valeur SHAP calculée par
un explainer.

## Image facultative

Les fichiers JPG, JPEG et PNG jusqu'à 5 Mo peuvent être affichés dans la démonstration. L'image sert
uniquement à la visualisation. La prédiction principale reste textuelle.

## Limites et déploiement Cloud

Le fichier `tfidf_linearsvc.pkl` pèse environ 51 Mo et il est actuellement ignoré par Git. Il ne sera
donc pas présent automatiquement sur Streamlit Community Cloud. Une solution doit être choisie :

1. stockage distant versionné, puis téléchargement avec contrôle du hash au démarrage ;
2. Git LFS, si le dépôt et la plateforme le prennent en charge ;
3. release GitHub contenant l'artefact versionné.

Le téléchargement avec vérification SHA-256 est recommandé. Les versions de scikit-learn et joblib
sont verrouillées pour limiter les incompatibilités de désérialisation.

La version actuelle ne charge ni PyTorch, ni Sentence Transformers, ni les caches NumPy. Elle reste
compatible avec une exécution CPU et une mémoire limitée, sous réserve de rendre le modèle disponible.
