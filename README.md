# Projet Complications de Santé

<p align="center">
  <img src="outputs/figures/target_distribution.png" alt="Distribution de la cible" width="720" />
</p>

## Résumé du projet

Ce projet a pour objectif de prédire le risque qu’un patient développe une complication de santé dans les 6 mois à partir de données cliniques, biométriques et comportementales. Il s’inscrit dans une logique d’aide à la décision médicale, en s’appuyant sur l’analyse exploratoire de données et la modélisation supervisée.

L’architecture du projet est pensée pour être reproductible, compréhensible et exploitable : nettoyage des données, comparaison de plusieurs modèles, sélection du meilleur modèle, sauvegarde des artefacts, et inférence sur de nouveaux patients.

---

## Objectifs métier et techniques

### Objectifs métier

- identifier les patients à risque élevé de complication à 6 mois ;
- détecter les facteurs associés au risque clinique ;
- fournir un outil de décision basé sur des variables explicatives ;
- exploiter un jeu de données hétérogène et bruité, proche d’un contexte réel.

### Objectifs techniques

- nettoyer automatiquement un dataset brut et incohérent ;
- transformer des colonnes textuelles en formats exploitables ;
- comparer des modèles de classification binaire ;
- sélectionner le modèle le plus pertinent selon une métrique robuste ;
- générer un artefact prêt à l’emploi pour la prédiction.

---

## Aperçu du dataset

Le projet utilise un fichier CSV de patients avec des variables telles que :

- âge ;
- sexe ;
- tension systolique et diastolique ;
- glycémie à jeun ;
- HbA1c ;
- cholestérol LDL / HDL ;
- triglycérides ;
- IMC ;
- tabagisme ;
- activité physique ;
- adhérence au traitement ;
- score de comorbidité ;
- variables de suivi médical.

La cible est une variable binaire :

- 0 : pas de complication dans les 6 mois ;
- 1 : complication détectée dans les 6 mois.

---

## Stack technique

- Python 3.12+
- pandas
- NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- joblib
- Jupyter Notebook

---

## Structure du dépôt

```text
.
├── data/
│   ├── raw/
│   │   └── projet3_sante_complications.csv
│   └── processed/
│       └── dataset_clean.csv
├── docs/
│   └── exemple_patient.json
├── models/
│   ├── metrics.json
│   ├── modele_final.pkl
│   └── feature_columns.pkl
├── notebooks/
│   └── rapport_projet.ipynb
├── outputs/
│   └── figures/
│       ├── age_hba1c_distribution.png
│       ├── correlation_matrix.png
│       └── target_distribution.png
├── src/
│   ├── data_cleaning.py
│   ├── predict.py
│   └── train_model.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── .venv/
```

---

## Prérequis

Avant de démarrer, vérifiez que les éléments suivants sont installés :

- Python 3.10+ (recommandé : Python 3.12)
- pip
- git
- un terminal Bash / zsh / PowerShell

---

## 1) Cloner le projet

```bash
git clone https://github.com/Adam01-i/projet_complications_sante.git
cd projet_complications_sante
```

Si le dépôt est déjà présent localement, vous pouvez simplement vous placer dans le dossier du projet :

```bash
cd /chemin/vers/projet_complications_sante
```

---

## 2) Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3) Installer les dépendances

```bash
pip install -r requirements.txt
```

Cela installera les librairies nécessaires au nettoyage des données, à la modélisation, au plotting et au notebook.

---

## 4) Nettoyer les données

Depuis la racine du projet :

```bash
python3 src/data_cleaning.py data/raw/projet3_sante_complications.csv data/processed/dataset_clean.csv
```

Cette étape nettoie les valeurs incohérentes, normalise les colonnes, gère les variables textuelles et produit un dataset exploitable pour l’apprentissage.

---

## 5) Entraîner le modèle

```bash
python3 src/train_model.py --data data/raw/projet3_sante_complications.csv --output-dir models
```

Le script entraîne plusieurs modèles, compare leurs performances et sauvegarde le meilleur dans le dossier `models`.

### Fichiers générés

- `models/modele_final.pkl` : modèle final sauvegardé
- `models/feature_columns.pkl` : colonnes utilisées par le modèle
- `models/metrics.json` : métriques d’évaluation et résultats de comparaison

---

## 6) Faire une prédiction sur un patient

Un exemple de patient est fourni dans le dossier `docs` :

```bash
python3 src/predict.py \
  --model models/modele_final.pkl \
  --features models/feature_columns.pkl \
  --input docs/exemple_patient.json
```

La sortie renvoie une prédiction binaire et la probabilité associée de complication.

---

## Exemple de sortie

```json
{
  "prediction": 0,
  "probabilite_complication": 0.3965
}
```

---

## Pipeline de traitement des données

Le workflow du projet suit cette logique :

1. lecture du CSV brut ;
2. normalisation des noms de colonnes ;
3. nettoyage des valeurs textuelles et numériques ;
4. gestion des incohérences physiologiques ;
5. nettoyage de la variable cible ;
6. validation du dataset prêt pour le ML ;
7. split train/test ;
8. entraînement de plusieurs modèles ;
9. sélection du meilleur modèle selon F1 macro ;
10. sauvegarde et inférence.

---

## Analyse exploratoire

### Distribution de la cible

<p align="center">
  <img src="outputs/figures/target_distribution.png" alt="Distribution de la cible" width="760" />
</p>

Cette figure montre la distribution de la variable cible et confirme la présence d’un déséquilibre de classes, ce qui justifie l’utilisation d’un critère de performance robuste comme le F1 macro.

### Matrice de corrélation

<p align="center">
  <img src="outputs/figures/correlation_matrix.png" alt="Matrice de corrélation" width="760" />
</p>

La matrice de corrélation permet d’observer les relations entre variables numériques et d’identifier les corrélations fortes ou les redondances potentielles.

### Âge et HbA1c selon le statut de complication

<p align="center">
  <img src="outputs/figures/age_hba1c_distribution.png" alt="Age et HbA1c selon le statut" width="760" />
</p>

Cette visualisation met en évidence la relation entre certains indicateurs cliniques et la survenue de complications, ce qui aide à interpréter la décision du modèle.

---

## Modélisation

Le projet compare plusieurs algorithmes de classification binaire :

- Logistic Regression
- Random Forest
- Gradient Boosting

Le modèle final est sélectionné selon une approche orientée performance sur données déséquilibrées, en privilégiant le F1 macro plutôt que la simple précision. Cela permet de mieux évaluer les performances sur les deux classes même lorsque la classe positive est minoritaire.

Quand `imbalanced-learn` est disponible, le pipeline utilise SMOTE pour rééquilibrer les classes sur le jeu d’entraînement. Si la dépendance n’est pas installée, un mécanisme de repli compatible est appliqué automatiquement.

---

## Résultats attendus

Les performances sont suivies dans le fichier `models/metrics.json` et peuvent être interprétées selon :

- Accuracy
- F1 macro
- matrice de confusion
- classification report

La sélection du meilleur modèle est faite automatiquement à la fin de l’entraînement.

---

## Bonnes pratiques de développement

- toujours travailler depuis la racine du projet ;
- utiliser un environnement virtuel dédié ;
- ne pas modifier les artefacts générés sans revalider le modèle ;
- garder les données brutes dans `data/raw` ;
- conserver les graphiques et les résultats dans `outputs/` et `models/` ;
- documenter les changements avant d’ajuster le pipeline.

---

## Contribution

Le projet est structuré pour être réutilisable et extensible. Les scripts principaux sont centralisés dans `src/`, les analyses exploratoires dans `notebooks/`, et les artefacts générés dans `models/` et `outputs/`.

---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).

---

## Contact

Pour toute question, amélioration ou collaboration, vous pouvez ouvrir une issue ou utiliser le dépôt GitHub associé au projet.
