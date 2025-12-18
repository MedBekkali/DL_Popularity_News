# DL_Popularity — MLPs pour Classification et Régression Multi-sorties (UCI Online News Popularity)

Projet de Deep Learning (ESAIP) : implémentation et comparaison de MLP/DNN sur **données structurées** pour :
- une tâche de **classification binaire** (article viral / non viral),
- une tâche de **régression multi-sorties** (3 sorties continues).

Le projet est implémenté avec les **3 frameworks** demandés : **Scikit-learn**, **TensorFlow/Keras** et **PyTorch**.

---

## 1) Contexte et scénario applicatif

Objectif applicatif : **aider une rédaction / équipe marketing** à estimer le potentiel de popularité d’un article avant publication.

Deux usages complémentaires :
1. **Classification (Viral / Non viral)**  
   Déclencher une alerte “article potentiellement viral” pour prioriser la promotion.
2. **Régression multi-sorties (3 cibles)**  
   Fournir une estimation robuste et interprétable de la popularité (différentes définitions de la popularité).

---

## 2) Dataset

- Dataset : **Online News Popularity** (UCI)  
- Type : données tabulaires (features numériques)  
- Variable cible brute : `shares`

Le dataset est chargé via `config.data_path`.

---

## 3) Définition des cibles (targets)

### 3.1 Classification binaire
On définit **viral** comme appartenant au **top q%** des articles selon `shares` :

- Seuil : `thr = quantile(shares, viral_quantile)`
- Label : `y_class = 1 si shares >= thr, sinon 0`

Par défaut : `viral_quantile = 0.75` (≈ 25% positifs).

**Justification :**
- Définition simple, robuste aux valeurs extrêmes.
- Problème réaliste “événement rare” (classe minoritaire).

### 3.2 Régression multi-sorties (3 outputs)
On construit un vecteur `Y_reg` de forme `(n_samples, 3)` :

1. **y1 = log1p(shares)**  
   Stabilise la distribution (shares est très asymétrique).
2. **y2 = log1p(min(shares, cap))**  
   Version robuste (cap des outliers), puis log.
3. **y3 = percentile rank dans [0, 1]**  
   Score de virality relatif (rang normalisé).

**Justification :**
- y1 : cible stable et standard pour un phénomène “heavy-tail”.
- y2 : réduit l’influence des articles extrêmement viraux.
- y3 : capture une notion “relative” utile métier (score).

---

## 4) Protocole expérimental

### 4.1 Split Train / Val / Test (un seul protocole)
On utilise **un seul split** commun aux deux tâches (classification et régression) :

- split stratifié sur `y_class` (pour conserver le taux de positifs)
- tailles contrôlées par `test_size` et `val_size`
- `random_state` fixé pour reproductibilité

### 4.2 Prétraitement
- Prétraitement des features **fit uniquement sur train** (pas de fuite de données)
- Normalisation/standardisation pour stabiliser l’entraînement des MLP
- Pour la régression : **StandardScaler sur Y_reg** (fit sur train uniquement), puis inverse_transform au test

### 4.3 Optimisation / régularisation
- **Early stopping** (sur validation) pour éviter l’overfitting
- Dropout (Torch/TF) ou `alpha` / early_stopping (Scikit-learn)

### 4.4 Choix du seuil de décision en classification
- Le seuil 0.5 n’est pas imposé.
- On sélectionne un **seuil** sur validation en maximisant le **Macro-F1**.

**Justification :**
- Dataset déséquilibré (≈ 25% positifs)
- Macro-F1 traite les deux classes de manière équilibrée, contrairement à l’accuracy.

---

## 5) Métriques de performance

### 5.1 Classification
- **Accuracy**
- **Macro-F1**
- **Matrice de confusion**

### 5.2 Régression (par output + moyenne)
- **R²** par sortie (y1, y2, y3)
- **RMSE** par sortie
- **MAE** par sortie
- `r2_mean` (moyenne simple des R²)

---

## 6) Exécution du projet

### 6.1 Installation
Créer un environnement (recommandé) puis installer les dépendances :

```bash
pip install -r requirements.txt
