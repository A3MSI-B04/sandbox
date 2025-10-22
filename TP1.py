"""Outils de chargement, pré‑traitement, clustering et régression pour le TP immobilier.

But: ce fichier est une version homogénéisée et plus lisible du script original.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Constantes et configuration

DEFAULT_HEADERS: Tuple[str, ...] = (
    "Identifiant de document",
    "Reference document",
    "1 Articles CGI",
    "2 Articles CGI",
    "3 Articles CGI",
    "4 Articles CGI",
    "5 Articles CGI",
    "No disposition",
    "Date mutation",
    "Nature mutation",
    "Valeur fonciere",
    "No voie",
    "B/T/Q",
    "Type de voie",
    "Code voie",
    "Voie",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Prefixe de section",
    "Section",
    "No plan",
    "No Volume",
    "1er lot",
    "Surface Carrez du 1er lot",
    "2eme lot",
    "Surface Carrez du 2eme lot",
    "3eme lot",
    "Surface Carrez du 3eme lot",
    "4eme lot",
    "Surface Carrez du 4eme lot",
    "5eme lot",
    "Surface Carrez du 5eme lot",
    "Nombre de lots",
    "Code type local",
    "Type local",
    "Identifiant local",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Nature culture",
    "Nature culture speciale",
    "Surface terrain",
)

SELECTED_COLUMNS: Tuple[str, ...] = (
    "Date mutation",
    "Nature mutation",
    "Valeur fonciere",
    "No voie",
    "Type de voie",
    "Voie",
    "Code postal",
    "Commune",
    "1er lot",
    "Surface Carrez du 1er lot",
    "2eme lot",
    "Surface Carrez du 2eme lot",
    "3eme lot",
    "Surface Carrez du 3eme lot",
    "4eme lot",
    "Surface Carrez du 4eme lot",
    "5eme lot",
    "Surface Carrez du 5eme lot",
    "Nombre de lots",
    "Type local",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Surface terrain",
)

INPUT_DELIMITER = "|"
OUTPUT_DELIMITER = ";"
READ_ENCODINGS = ("utf-8", "latin-1")

# Colonnes numériques potentielles utilisées pour le clustering / régression
NUMERICAL_COLUMNS = [
    "Valeur fonciere",
    "Surface Carrez du 1er lot",
    "Surface Carrez du 2eme lot",
    "Surface Carrez du 3eme lot",
    "Surface Carrez du 4eme lot",
    "Surface Carrez du 5eme lot",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Surface terrain",
]

# I/O et utilitaires

def prompt_file_path() -> Path:
    """Invite l'utilisateur à fournir un chemin de fichier et vérifie son existence."""
    while True:
        raw = input(f"Chemin du fichier texte : ").strip()
        if not raw:
            print("Le chemin ne doit pas être vide.")
            continue
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        print(f"Fichier introuvable : {p}")


def prompt_postal_code() -> str:
    """Invite l'utilisateur pour un code postal (chiffres uniquement)."""
    while True:
        code = input("Code postal à filtrer : ").strip()
        if not code:
            print("Le code postal ne doit pas être vide.")
            continue
        if not code.isdigit():
            print("Merci de ne saisir que des chiffres.")
            continue
        return code


def _iterate_rows(path: Path, encoding: str) -> Iterable[List[str]]:
    """Générateur qui lit le fichier ligne à ligne et split par INPUT_DELIMITER."""
    with path.open("r", encoding=encoding, newline="") as fh:
        for raw in fh:
            yield [cell.strip() for cell in raw.rstrip("\r\n").split(INPUT_DELIMITER)]


def load_rows_by_postal_code(path: Path, postal_code: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Charge toutes les lignes et retourne l'en‑tête + les enregistrements filtrés par code postal."""
    last_err: Optional[UnicodeDecodeError] = None
    rows: Optional[List[List[str]]] = None
    for enc in READ_ENCODINGS:
        try:
            rows = list(_iterate_rows(path, encoding=enc))
            break
        except UnicodeDecodeError as exc:
            last_err = exc
    if rows is None:
        raise last_err or RuntimeError("Impossible de lire le fichier.")

    if not rows:
        return list(DEFAULT_HEADERS), []

    header, *data_rows = rows
    # Si l'entête du fichier diffère (casse) on force DEFAULT_HEADERS
    if {h.lower() for h in header} != {h.lower() for h in DEFAULT_HEADERS}:
        header = list(DEFAULT_HEADERS)
        data_rows = rows  # tout le fichier devient des lignes de données

    filtered: List[Dict[str, str]] = []
    seen: set = set()
    for values in data_rows:
        normalized = (values + [""] * (len(header) - len(values)))[: len(header)]
        key = tuple(normalized)
        if key in seen:
            continue
        seen.add(key)
        rec = dict(zip(header, normalized))
        if rec.get("Code postal", "") == postal_code:
            filtered.append(rec)
    return header, filtered


def trim_records_to_selection(records: List[Dict[str, str]], columns: List[str]) -> List[Dict[str, str]]:
    """Garde uniquement les colonnes demandées pour chaque enregistrement."""
    return [{col: rec.get(col, "") for col in columns} for rec in records]


def display_filtered_sample(records: List[Dict[str, str]], columns: List[str], limit: int = 5) -> None:
    """Affiche un petit échantillon des enregistrements filtrés."""
    if not records:
        print("Aucune ligne ne correspond au code postal demandé.")
        return
    print(f"{len(records)} lignes correspondent au code postal.")
    for i, rec in enumerate(records[:limit], start=1):
        print(f"--- Ligne {i} ---")
        for col in columns:
            v = rec.get(col, "")
            if v:
                print(f"{col}: {v}")


def prompt_output_path(source: Path, postal_code: str) -> Path:
    """Demande le chemin de sortie pour le CSV (propose une valeur par défaut)."""
    default = source.with_name(f"{source.stem}_filtered_{postal_code}.csv")
    raw = input(f"Chemin du fichier CSV de sortie [{default}]: ").strip()
    if not raw:
        return default
    out = Path(raw).expanduser()
    if out.is_dir():
        return out / default.name
    return out


def write_records_as_csv(path: Path, header: List[str], records: List[Dict[str, str]]) -> None:
    """Écrit les enregistrements en CSV. On inclut automatiquement toutes les clés rencontrées."""
    fieldnames = list(header)
    # Ajouter les clés supplémentaires trouvées dans records (mais laisser l'ordre du header en priorité)
    extra = sorted({k for r in records for k in r.keys()} - set(fieldnames))
    fieldnames.extend(extra)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=OUTPUT_DELIMITER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# Fusion dépendances / résidences

def regrouper_dependances(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ajoute la clé 'Dependance presente' (Oui/Non) si une dépendance partage la même Valeur fonciere+Voie."""
    key_attrs = ("Valeur fonciere", "Voie")
    resid_by_key: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    dep_by_key: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}

    for rec in records:
        key = tuple(rec.get(a, "").strip() for a in key_attrs)
        t = rec.get("Type local", "").lower()
        if "dépendance" in t or "dependance" in t:
            dep_by_key.setdefault(key, []).append(rec)
        else:
            resid_by_key.setdefault(key, []).append(rec)

    out: List[Dict[str, str]] = []
    for key, resid_list in resid_by_key.items():
        has_dep = key in dep_by_key
        for r in resid_list:
            rr = r.copy()
            rr["Dependance presente"] = "Oui" if has_dep else "Non"
            out.append(rr)
    # si seules dépendances existent (pas de résidence), on les garde aussi
    for key, dep_list in dep_by_key.items():
        if key not in resid_by_key:
            for r in dep_list:
                rr = r.copy()
                rr["Dependance presente"] = "Oui"
                out.append(rr)
    return out


# Parsing numérique et sélection des features

def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Convertit une Series en float : supprime espaces, remplace ',' par '.' puis to_numeric coercing."""
    s = series.astype(str).str.replace(r"\s+", "", regex=True).str.replace(",", ".", regex=False)
    s = s.replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def select_numerical_features(records: List[Dict[str, str]], columns: List[str]) -> pd.DataFrame:
    """Construit un DataFrame contenant uniquement les colonnes numériques utiles, converties en float."""
    df = pd.DataFrame(records)
    selected = [c for c in NUMERICAL_COLUMNS if c in columns]
    if not selected:
        return pd.DataFrame(columns=selected)
    def clean(col: pd.Series) -> pd.Series:
        return pd.to_numeric(
            col.astype(str).str.replace(r"\s+", "", regex=True).str.replace(",", ".", regex=False).replace({"nan": None, "None": None}),
            errors="coerce",
        )
    df_num = df[selected].apply(clean)
    medians = df_num.median()
    return df_num.fillna(medians).fillna(0.0)


# Clustering (KMeans) et choix automatique de k

def find_optimal_k(X: np.ndarray, max_k: int = 10, plot: bool = False) -> int:
    """Teste plusieurs k et renvoie le k choisi (maximisant la silhouette si possible)."""
    n_samples = X.shape[0]
    if n_samples < 2:
        return 1
    max_k = min(max_k, max(2, n_samples - 1))
    k_candidates = [k for k in range(2, max_k + 1) if k < n_samples]

    inertias = []
    silhouettes = []
    valid_k = []

    for k in k_candidates:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            if len(set(labels)) < 2:
                silhouettes.append(np.nan)
            else:
                try:
                    silhouettes.append(silhouette_score(X, labels))
                except Exception:
                    silhouettes.append(np.nan)
            valid_k.append(k)
        except Exception:
            continue

    if not valid_k:
        return 1

    if plot:
        ks = valid_k
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(ks, inertias, marker="o")
        plt.title("Inertie vs k")
        plt.xlabel("k")
        plt.ylabel("Inertie")
        plt.subplot(1, 2, 2)
        plt.plot(ks, silhouettes, marker="o", color="green")
        plt.title("Silhouette vs k")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.tight_layout()
        plt.show()

    sil_arr = np.array(silhouettes, dtype=float)
    if not np.all(np.isnan(sil_arr)):
        best_idx = int(np.nanargmax(sil_arr))
        return valid_k[best_idx]
    return valid_k[0]


def cluster_properties(records: List[Dict[str, str]], columns: List[str], max_k: int = 10, plot: bool = False):
    """Exécute la préparation, choisit k et retourne les clusters + objets utiles."""
    X_df = select_numerical_features(records, columns)
    if X_df.empty or X_df.shape[0] < 1:
        raise RuntimeError("Pas de données numériques valides pour le clustering.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    k = find_optimal_k(X_scaled, max_k=max_k, plot=plot)
    if k < 1:
        raise RuntimeError("k invalide calculé pour le clustering.")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, k, kmeans, scaler


# Analyse descriptive des clusters

def print_cluster_summary(records: List[Dict[str, str]], clusters: np.ndarray, columns: List[str]) -> None:
    """Affiche des statistiques simples (moyennes) pour chaque cluster."""
    df = pd.DataFrame(records).copy()
    df["Cluster"] = clusters
    # identifier colonnes numériques parmi celles demandées
    num_cols = [c for c in columns if c in df.columns and (pd.api.types.is_numeric_dtype(df[c]) or c.startswith("Surface") or c == "Valeur fonciere")]
    print("\n=== Analyse descriptive des clusters ===")
    for cid in sorted(set(clusters)):
        sub = df[df["Cluster"] == cid]
        if sub.empty:
            continue
        print(f"\n--- Cluster {cid} ({len(sub)} transactions) ---")
        means = sub[num_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
        print("Moyennes (top 5) :")
        print(means.head(5).to_string())


# Régression par cluster

def train_regression_by_cluster(records: List[Dict[str, str]], selected_columns: List[str]) -> Dict[int, LinearRegression]:
    """Entraîne un LinearRegression par cluster et sauvegarde chaque modèle."""
    df = pd.DataFrame(records)
    if "Cluster" not in df.columns:
        raise RuntimeError("Pas de colonne 'Cluster' dans les enregistrements.")
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(-1).astype(int)

    X_all = select_numerical_features(records, selected_columns)
    if "Valeur fonciere" not in df.columns:
        raise RuntimeError("Pas de colonne 'Valeur fonciere' pour la régression.")
    y_all = parse_numeric_series(df["Valeur fonciere"])

    models: Dict[int, LinearRegression] = {}
    for cid in sorted(df["Cluster"].unique()):
        if cid < 0:
            continue
        mask = df["Cluster"] == cid
        Xc = X_all[mask]
        yc = y_all[mask]
        if len(Xc) < 2:
            print(f"Cluster {cid} trop petit pour entraîner un modèle (n={len(Xc)})")
            continue
        model = LinearRegression()
        model.fit(Xc, yc)
        models[cid] = model
        joblib.dump(model, f"regression_model_cluster_{cid}.joblib")
        print(f"Modèle entraîné et sauvegardé pour Cluster {cid} (n={len(Xc)})")
    return models


def predict_and_evaluate(records: List[Dict[str, str]], models: Dict[int, LinearRegression], selected_columns: List[str]) -> None:
    """Prédit par modèle de cluster, affiche RMSE/R² par cluster et global."""
    df = pd.DataFrame(records)
    if "Cluster" not in df.columns:
        raise RuntimeError("Pas de colonne 'Cluster' dans les enregistrements.")
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(-1).astype(int)

    X_all = select_numerical_features(records, selected_columns)
    if "Valeur fonciere" not in df.columns:
        raise RuntimeError("Pas de colonne 'Valeur fonciere' pour l'évaluation.")
    y_true = parse_numeric_series(df["Valeur fonciere"])

    y_pred = np.zeros(len(df), dtype=float)
    print("\n=== Évaluation des modèles ===")
    for cid, model in models.items():
        mask = df["Cluster"] == cid
        if mask.sum() == 0:
            continue
        Xc = X_all[mask]
        preds = model.predict(Xc)
        y_pred[mask.to_numpy()] = preds
        rmse = float(np.sqrt(mean_squared_error(y_true[mask], preds)))
        r2 = float(r2_score(y_true[mask], preds))
        print(f"Cluster {cid}: RMSE={rmse:.2f}, R²={r2:.2f}")

    overall_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    overall_r2 = float(r2_score(y_true, y_pred))
    print(f"\nGlobal: RMSE={overall_rmse:.2f}, R²={overall_r2:.2f}")


# Prédiction pondérée optionnelle (partie 5)

def predict_weighted(records: List[Dict[str, str]], models: Dict[int, LinearRegression], kmeans: KMeans, scaler: StandardScaler, selected_columns: List[str]) -> None:
    """Prédit une valeur pondérée par la probabilité d'appartenance aux clusters."""
    if kmeans is None or scaler is None:
        raise ValueError("KMeans et StandardScaler nécessaires pour la pondération.")
    df = pd.DataFrame(records)
    X = select_numerical_features(records, selected_columns)
    X_scaled = scaler.transform(X)
    distances = kmeans.transform(X_scaled)  # shape (n_samples, n_clusters)
    inverse = 1.0 / (distances + 1e-10)
    probs = inverse / inverse.sum(axis=1, keepdims=True)

    final_preds = np.zeros(len(df), dtype=float)
    for i, row in enumerate(X_scaled):
        p = 0.0
        for cid, model in models.items():
            # si cid dépasse le nombre de colonnes probs, on ignore (sécurité)
            if cid >= probs.shape[1]:
                continue
            p += float(model.predict([row])[0]) * float(probs[i, cid])
        final_preds[i] = p

    for rec, v in zip(records, final_preds):
        rec["Estimation pondérée"] = f"{v:.2f}"
    print("\nPrédictions pondérées ajoutées aux enregistrements.")


# Main

def main() -> None:
    path = prompt_file_path()
    postal_code = prompt_postal_code()
    header, records = load_rows_by_postal_code(path, postal_code)

    selected_header = [c for c in SELECTED_COLUMNS if c in header]
    trimmed = trim_records_to_selection(records, selected_header)
    display_filtered_sample(trimmed, selected_header)

    if not records:
        return

    trimmed = trim_records_to_selection(records, selected_header)
    trimmed = regrouper_dependances(trimmed)

    print("\n[CLUSTERING NON SUPERVISÉ]")
    try:
        clusters, k, kmeans, scaler = cluster_properties(trimmed, selected_header, max_k=8, plot=True)
    except Exception as exc:
        print(f"Erreur clustering : {exc}")
        return

    for rec, cl in zip(trimmed, clusters):
        rec["Cluster"] = int(cl)

    print(f"{k} clusters détectés.")
    print_cluster_summary(trimmed, clusters, selected_header)

    print("\n[RÉGRESSION PAR CLASSE]")
    regression_models = train_regression_by_cluster(trimmed, selected_header)
    if not regression_models:
        print("Aucun modèle de régression n'a pu être entraîné.")
        return

    predict_and_evaluate(trimmed, regression_models, selected_header)

    # partie 5 optionnelle : prédiction pondérée
    if regression_models and kmeans is not None and scaler is not None:
        predict_weighted(trimmed, regression_models, kmeans, scaler, selected_header)
        if "Estimation pondérée" not in selected_header:
            selected_header.append("Estimation pondérée")

    output = prompt_output_path(path, postal_code)
    if "Cluster" not in selected_header:
        selected_header.append("Cluster")
    write_records_as_csv(output, selected_header, trimmed)
    print(f"CSV écrit (avec clusters & pondération) dans {output.resolve()}")


if __name__ == "__main__":
    main()

