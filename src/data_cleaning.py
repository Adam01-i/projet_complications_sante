"""
Nettoyage du dataset "complications de santé à 6 mois".

Ce module consolide toutes les étapes de nettoyage explorées dans le
notebook original (projet3.ipynb) en fonctions réutilisables et testables.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "complication_6m"

# Colonnes numériques "sales" nécessitant un nettoyage générique
# (unités, virgules décimales, espaces)
GENERIC_NUMERIC_COLS = [
    "duree_hospitalisation_jours",
    "nb_hospitalisations_12m",
    "nb_medicaments",
    "adherence_traitement_pct",
    "activite_physique_min_semaine",
    "consommation_alcool_unites_semaine",
    "score_comorbidite",
    "score_stress",
]

TABAGISME_MAPPING = {"jamais": 0, "ancien": 1, "actif": 2, "inconnu": np.nan}


def resolve_project_path(path: str) -> str:
    """Résout un chemin de fichier depuis la racine du projet.

    Accepte les chemins relatifs exécutés depuis la racine du repo ou depuis
    le dossier src, et tolère aussi les chemins mal écrits avec des "../".
    """
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


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Charge le CSV brut et normalise immédiatement les noms de colonnes."""
    resolved_csv = resolve_project_path(csv_path)
    df = pd.read_csv(resolved_csv, on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def _clean_numeric_generic(x):
    """Convertit une valeur texte 'sale' (unités, virgule, espaces) en float."""
    if pd.isna(x):
        return np.nan
    x = str(x)
    x = re.sub(r"[a-zA-Z/]+", "", x)  # enlève unités type mg/L
    x = x.replace(",", ".").strip()
    try:
        return float(x)
    except ValueError:
        return np.nan


def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la colonne age : préfixes 'age:', suffixes 'ans', virgules."""
    df = df.copy()
    df["age"] = (
        df["age"]
        .astype(str)
        .str.replace("age:", "", regex=False)
        .str.replace("ans", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())
    df["age"] = df["age"].astype(int)
    return df


def clean_sexe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise la colonne sexe (m/f) à partir de variantes libres."""
    df = df.copy()
    df["sexe"] = df["sexe"].astype(str).str.lower().str.strip()
    df["sexe"] = df["sexe"].replace(
        {"femme": "f", "f": "f", "m": "m", "homme": "m", "h": "m"}
    )
    return df


def clean_imc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["imc"] = (
        df["imc"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.extract(r"([0-9]+\.?[0-9]*)")[0]
    )
    df["imc"] = pd.to_numeric(df["imc"], errors="coerce")
    return df


def clean_tension(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie tension systolique/diastolique et retire les valeurs
    physiologiquement incohérentes (systolique <= diastolique)."""
    df = df.copy()
    for col in ["tension_systolique", "tension_diastolique"]:
        df[col] = df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    incoherent = df["tension_systolique"] <= df["tension_diastolique"]
    df = df[~incoherent].copy()
    return df


def add_hypertension_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hypertension"] = (
        (df["tension_systolique"] >= 140) | (df["tension_diastolique"] >= 90)
    ).astype(int)
    return df


def clean_glycemie(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["glycemie_jeun"] = (
        df["glycemie_jeun"]
        .astype(str)
        .str.replace("mg/L", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["glycemie_jeun"] = pd.to_numeric(df["glycemie_jeun"], errors="coerce")
    return df


def clean_hba1c(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie hba1c. Contrairement au notebook original, les valeurs
    manquantes résiduelles après extraction numérique sont explicitement
    imputées ici (médiane) plutôt que laissées en NaN jusqu'au split."""
    df = df.copy()

    def _parse(x):
        if pd.isna(x):
            return np.nan
        x = str(x).replace(",", ".")
        match = re.findall(r"\d+\.?\d*", x)
        return float(match[0]) if match else np.nan

    df["hba1c"] = df["hba1c"].apply(_parse)
    df["hba1c"] = df["hba1c"].fillna(df["hba1c"].median())
    return df


def clean_lipid_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["cholesterol_ldl", "cholesterol_hdl", "triglycerides"]:
        df[col] = df[col].apply(_clean_numeric_generic)
    return df


def clean_generic_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in GENERIC_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_clean_numeric_generic)
    return df


def clean_tabagisme(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tabagisme_statut"] = df["tabagisme_statut"].str.lower().str.strip()
    df["tabagisme_statut"] = df["tabagisme_statut"].map(TABAGISME_MAPPING)
    df["tabagisme_statut"] = df["tabagisme_statut"].fillna(0)
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


def impute_remaining_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes résiduelles (médiane) pour les
    colonnes qui en comportaient encore dans le notebook original."""
    df = df.copy()
    for col in ["adherence_traitement_pct", "activite_physique_min_semaine"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def clean_dataset(csv_path: str) -> pd.DataFrame:
    """Pipeline complet de nettoyage, du CSV brut au DataFrame prêt pour le ML."""
    df = load_raw_data(csv_path)
    df = df.drop_duplicates()

    df = clean_age(df)
    df = clean_sexe(df)
    df = clean_imc(df)
    df = clean_tension(df)
    df = add_hypertension_flag(df)
    df = clean_glycemie(df)
    df = clean_hba1c(df)
    df = clean_lipid_panel(df)
    df = clean_generic_numeric_cols(df)
    df = clean_tabagisme(df)
    df = clean_target(df)
    df = impute_remaining_numeric(df)

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys

    src = resolve_project_path(sys.argv[1]) if len(sys.argv) > 1 else "data/raw/projet3_sante_complications.csv"
    out = resolve_project_path(sys.argv[2]) if len(sys.argv) > 2 else "data/processed/dataset_clean.csv"
    cleaned = clean_dataset(src)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"Dataset nettoyé : {cleaned.shape[0]} lignes, {cleaned.shape[1]} colonnes -> {out_path}")
