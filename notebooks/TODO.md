# TODO — à faire avant de relancer les notebooks (à supprimer une fois fait)

Contexte : les notebooks 02/03 chargeaient un mauvais dataset (`df_train_processed.csv` au lieu
de celui produit par le notebook 01). C'est corrigé sur `haoyue/dev`. Il faut tout relancer
proprement dans l'ordre pour avoir des résultats fiables.

## 1. Récupérer le code
```
git pull origin haoyue/dev
```

## 2. Vérifier l'environnement
```
pip install -r requirements.txt
```
Dans `02_modeling.ipynb`, section "Détection du dispositif de calcul" : vérifier que ça affiche
bien `Device utilisé : cuda` + le nom de ta carte. Si c'est `cpu`, régler le driver/PyTorch
GPU avant de continuer (sinon les sections image/fusion sont ingérables en temps).

## 3. Vider les anciens caches/modèles (obligatoire, sinon rechargement silencieux du mauvais dataset)
- `data/processed/*.csv`
- `models/artifacts/*.npy`
- `models/baselines/*.pkl`
- `models/text/*.pt`
- `models/image/*.pt`

## 4. Vérifier les données brutes
`data/raw/` doit contenir : `X_train_update.csv`, `X_test_update.csv`, `Y_train_CVw08PX.csv`,
`images.zip` (dézippage automatique par le notebook 01).

## 5. Exécuter les 3 notebooks dans l'ordre, du début à la fin, sans sauter de cellule
1. `01_eda_preprocessing.ipynb` → génère `data/processed/train_clean.csv` et `test_clean.csv`
2. `02_modeling.ipynb` → TF-IDF (~15s) puis SBERT + ResNet50 + fusion (GPU requis, ~1-3h)
3. `03_evaluation.ipynb` → nécessite que le notebook 02 soit allé jusqu'au bout

## 6. Avant de push
`git status` ne doit montrer comme modifiés que :
- les `.ipynb` (nouveaux outputs)
- `models/artifacts/resultats_comparaison.csv`
- `models/artifacts/resultats_finaux_03.csv`
- `data/submissions/submission_tfidf_svc.csv`
- `reports/figures/*.png`

**Ne jamais committer** `data/raw/`, `data/processed/`, `data/extracted/`, ni de fichiers
`.pkl/.pt/.npy` (normalement gitignorés — si `git status` en montre, il y a un souci).
