"""
Entraînement et sélection du modèle de prédiction des complications à 6 mois.

Choix retenu ici : un seul round d'entraînement, avec SMOTE intégré dans
un Pipeline imbalanced-learn (comme dans la version finale du notebook,
qui est aussi celle correspondant aux artefacts modele_final.pkl /
feature_columns.pkl fournis). La comparaison baseline vs SMOTE est
conservée à titre pédagogique dans notebooks/03_modelisation.ipynb,
mais le code de production ne garde que la version retenue.

Bug corrigé : dans le notebook original, l'imputation des NaN restants
de X_test utilisait la médiane calculée sur X_test lui-même
(`X_test.fillna(X_test.median())`), ce qui fait fuiter de l'information
sur la distribution du jeu de test dans le pré-traitement. Ici,
l'imputation (comme le reste du pré-traitement) est apprise uniquement
sur X_train, via le SimpleImputer intégré au Pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_cleaning import TARGET_COL, clean_dataset

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover - fallback sans imbalanced-learn
    from sklearn.pipeline import Pipeline

    HAS_IMBLEARN = False

RANDOM_STATE = 42


def resolve_project_path(path: str) -> str:
    """Résout un chemin relatif depuis la racine du projet."""
    raw_path = Path(path)
    if raw_path.exists():
        return str(raw_path)

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / raw_path,
        project_root / Path(*[part for part in raw_path.parts if part not in (".", "..")]),
        Path(__file__).resolve().parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(raw_path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Sépare X/y et encode la variable catégorielle 'sexe'."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    X = pd.get_dummies(X, columns=["sexe"], drop_first=True)
    feature_columns = X.columns.tolist()
    return X, y, feature_columns


def make_pipelines() -> dict[str, Pipeline]:
    """Construit les 3 pipelines de modélisation exigés par le projet.

    Si imbalanced-learn est disponible, chaque pipeline gère le
    déséquilibre de classes via SMOTE (appliqué uniquement sur les
    plis d'entraînement, jamais sur le test — comportement natif des
    Pipelines imblearn). Sinon, on retombe sur class_weight="balanced"
    comme alternative fonctionnellement proche.
    """
    if HAS_IMBLEARN:
        resampler_step = [("smote", SMOTE(random_state=RANDOM_STATE))]
        log_reg = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        resampler_step = []
        log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
        rf = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        )
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)  # pas de class_weight en GB

    pipelines = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                *resampler_step,
                ("model", log_reg),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                *resampler_step,
                ("model", rf),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                *resampler_step,
                ("model", gb),
            ]
        ),
    }
    return pipelines


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
    }


def main(data_path: str, output_dir: str) -> None:
    data_path = resolve_project_path(data_path)
    output_dir = Path(resolve_project_path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_dataset(data_path)
    X, y, feature_columns = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = make_pipelines()
    results = []
    fitted = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        res = evaluate(name, pipeline, X_test, y_test)
        results.append(res)
        print(f"=== {name} ===")
        print(f"Accuracy: {res['accuracy']:.4f} | F1 macro: {res['f1_macro']:.4f}")

    best = max(results, key=lambda r: r["f1_macro"])
    best_name = best["name"]
    best_pipeline = fitted[best_name]
    print(f"\nMeilleur modèle (F1 macro) : {best_name}")

    joblib.dump(best_pipeline, output_dir / "modele_final.pkl")
    joblib.dump(feature_columns, output_dir / "feature_columns.pkl")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_name, "smote_used": HAS_IMBLEARN, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nArtefacts sauvegardés dans {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le modèle de prédiction des complications")
    parser.add_argument("--data", default="data/raw/projet3_sante_complications.csv")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    main(args.data, args.output_dir)
