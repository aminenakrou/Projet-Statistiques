# Indicateur de Performance d'un Score : La Courbe ROC

**Projet de Statistiques — Polytech Lille | Amine Nakrou | 2024–2025**

---

## Présentation

Ce projet modélise le **risque d'impayé** dans un portefeuille de crédit. Il compare trois algorithmes de classification (Régression Logistique, Arbre de Décision, Forêt Aléatoire) avec et sans rééquilibrage SMOTE, en les évaluant principalement via la **courbe ROC** et l'**AUC**.

Le site web de présentation se trouve dans **`index.html`**.

---

## Structure du projet

```
projet ma/
├── index.html                        # Site web de présentation
├── scoring_COMPLET_FINAL.Rmd         # Code source R Markdown
├── scoring_COMPLET_FINAL.html        # Rapport R généré (knitr)
├── GV_RSQ_data_scoring_risque.txt    # Données (pipe-séparé)
├── DM-ridge et lasso.Rmd             # Projet complémentaire
└── README.md                         # Ce fichier
```

---

## Données

Fichier `GV_RSQ_data_scoring_risque.txt` — séparateur `|`

| Variable | Type | Description |
|---|---|---|
| `Identifiant` | Texte | ID client (supprimé avant modélisation) |
| `Montant` | Numérique | Montant du crédit (€) |
| `Duree` | Numérique | Durée du crédit (mois) |
| `Taux_interet` | Numérique | Taux d'intérêt annuel (%) |
| `Ressources_Client` | Numérique | Revenus mensuels (€) |
| `Charges_Client` | Numérique | Charges mensuelles (€) |
| `Telephone` | Binaire | 0 = Non, 1 = Oui |
| `Assurance` | Binaire | 0 = Non, 1 = Oui |
| `Client` | Binaire | 0 = Nouveau, 1 = Ancien |
| `Nb_emprunteurs` | Facteur | Nombre d'emprunteurs |
| `Payeur` | **Cible** | 0 = Impayé (86.7%), 1 = Payeur (13.3%) |

---

## Variables dérivées (Feature Engineering)

| Variable | Formule |
|---|---|
| `Taux_endettement` | `Charges / Ressources × 100` |
| `Mensualite` | Formule des annuités : `C × (r/12) / (1 − (1 + r/12)^{−n})` |
| `Reste_a_vivre` | `Ressources − Charges − Mensualite` |

---

## Pipeline

```
Données brutes
    → Nettoyage (trimws, facteurs, suppression Identifiant)
    → EDA (boxplots, corrélations, barplots conditionnels)
    → Feature Engineering (3 variables financières)
    → Split 70% train / 30% test (seed = 42)
    → SMOTE sur train uniquement (dup_size = 4)
    → 3 modèles × 2 versions = 6 configurations
    → Évaluation : Accuracy, Précision, Rappel, F1, AUC, Youden
```

---

## Modèles

### 1 — Régression Logistique (`glm`, famille `binomial`)

$$P(Y=1|X) = \frac{1}{1+e^{-(\beta_0 + \boldsymbol{\beta}^\top X)}}$$

Référence : Hosmer, Lemeshow & Sturdivant (2013). *Applied Logistic Regression*, 3rd ed., Wiley.

### 2 — Arbre de Décision (`rpart`, CART, `cp = 0.01`)

Critère de partition : impureté de Gini $= 1 - \sum_k p_k^2$

Référence : Breiman, Friedman, Stone & Olshen (1984). *Classification and Regression Trees*, CRC Press.

### 3 — Forêt Aléatoire (`randomForest`, `ntree = 200`)

$$\hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x, \Theta_b)$$

Référence : Breiman, L. (2001). *Random Forests*, Machine Learning, 45(1), 5–32.

---

## SMOTE

$$x_{new} = x_i + \lambda \cdot (x_{voisin} - x_i), \quad \lambda \sim \mathcal{U}(0,1)$$

`dup_size = 4` → 5× plus de Payeurs dans le train set.  
Le test set reste **non augmenté** pour une évaluation honnête.

Référence : Chawla et al. (2002). *SMOTE*, JAIR, 16, 321–357.

---

## Métriques

| Métrique | Formule | Priorité |
|---|---|---|
| Accuracy | `(VP + VN) / Total` | Trompeuse sur données déséquilibrées |
| Précision | `VP / (VP + FP)` | Secondaire |
| **Rappel** | `VP / (VP + FN)` | **Prioritaire en risque de crédit** |
| F1-Score | `2 × (Préc × Rapp) / (Préc + Rapp)` | Synthèse |
| **AUC** | Aire sous la courbe ROC | **Métrique principale** |

### Seuil de Youden

$$J^* = \arg\max_\tau \left[\text{Sensibilité}(\tau) + \text{Spécificité}(\tau) - 1\right]$$

---

## Packages R

```r
library(pROC)          # Courbe ROC, AUC, coords (Youden)
library(rpart)         # Arbre de décision (CART)
library(rpart.plot)    # Visualisation de l'arbre
library(randomForest)  # Forêt aléatoire
```

---

## Exécution

```r
# Dans RStudio ou R
rmarkdown::render("scoring_COMPLET_FINAL.Rmd")
```

Ou avec le bouton **Knit** dans RStudio. Fichier de données requis dans le même répertoire.

---

## Références complètes

1. **Breiman (2001)** — Random Forests. *Machine Learning*, 45(1), 5–32. https://link.springer.com/article/10.1023/A:1010933404324
2. **Breiman et al. (1984)** — *Classification and Regression Trees*. CRC Press.
3. **Chawla et al. (2002)** — SMOTE. *JAIR*, 16, 321–357. https://arxiv.org/abs/1106.1813
4. **Fawcett (2006)** — An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8). https://www.sciencedirect.com/science/article/pii/S016786550500303X
5. **Hosmer, Lemeshow & Sturdivant (2013)** — *Applied Logistic Regression*, 3rd ed. Wiley.
6. **James et al. (2021)** — *An Introduction to Statistical Learning*, 2nd ed. Springer. https://www.statlearning.com/
7. **Hastie, Tibshirani & Friedman (2009)** — *The Elements of Statistical Learning*, 2nd ed. Springer. https://hastie.su.domains/ElemStatLearn/
8. **Robin et al. (2011)** — pROC. *BMC Bioinformatics*, 12, 77. https://web.expasy.org/pROC/
9. **Thomas, Edelman & Crook (2017)** — *Credit Scoring and Its Applications*, 2nd ed. SIAM.

---

## Auteur

**Amine Nakrou**  
Élève-ingénieur — Polytech Lille  
Contact : aminenakrou635@gmail.com  
Projet de Statistiques — 2024–2025
