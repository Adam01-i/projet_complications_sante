"""
Prédiction du risque de complication à 6 mois pour un nouveau patient.

Exemple :
    python predict.py --model models/modele_final.pkl \
                       --features models/feature_columns.pkl \
                       --input exemple_patient.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


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


def load_artifacts(model_path: str, features_path: str):
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    return model, feature_columns


def prepare_input(record: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Transforme un dict patient brut en DataFrame aligné sur les colonnes
    utilisées à l'entraînement (même encodage one-hot de 'sexe')."""
    df = pd.DataFrame([record])
    if "sexe" in df.columns:
        df = pd.get_dummies(df, columns=["sexe"], drop_first=True)
    # Ajoute les colonnes manquantes (ex: sexe_m si le patient est "f") à 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def predict(record: dict, model_path: str, features_path: str) -> dict:
    model, feature_columns = load_artifacts(model_path, features_path)
    X = prepare_input(record, feature_columns)
    proba = model.predict_proba(X)[0][1]
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "probabilite_complication": round(float(proba), 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédit le risque de complication à 6 mois")
    parser.add_argument("--model", default="models/modele_final.pkl")
    parser.add_argument("--features", default="models/feature_columns.pkl")
    parser.add_argument("--input", required=True, help="Fichier JSON décrivant un patient")
    args = parser.parse_args()

    model_path = resolve_project_path(args.model)
    features_path = resolve_project_path(args.features)
    input_path = resolve_project_path(args.input)

    with open(input_path, encoding="utf-8") as f:
        patient = json.load(f)

    result = predict(patient, model_path, features_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
