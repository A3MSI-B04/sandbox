"""Outils de chargement et analyse pour le TP immobilier."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DEFAULT_HEADERS: tuple[str, ...] = (
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

SELECTED_COLUMNS: tuple[str, ...] = (
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

DELIMITER = "|"

def prompt_file_path() -> Path:
    """Demande le chemin du fichier et s'assure qu'il existe."""
    while True:
        raw_path = input("Chemin du fichier texte (pipe-separe) : ").strip()
        if not raw_path:
            print("Le chemin ne doit pas etre vide.")
            continue
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path
        print(f"Fichier introuvable : {path}")

def prompt_postal_code() -> str:
    """Demande un code postal et valide le format minimal."""
    while True:
        postal_code = input("Code postal a filtrer : ").strip()
        if not postal_code:
            print("Le code postal ne doit pas etre vide.")
            continue
        if not postal_code.isdigit():
            print("Merci de ne saisir que des chiffres.")
            continue
        return postal_code

def _iterate_rows(path: Path, encoding: str) -> Iterable[list[str]]:
    """Lit le fichier ligne par ligne et separe chaque colonne par DELIMITER."""
    with path.open("r", encoding=encoding, newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            yield [cell.strip() for cell in line.split(DELIMITER)]

def load_rows_by_postal_code(path: Path, postal_code: str) -> tuple[list[str], list[dict[str, str]]]:
    last_error: Optional[UnicodeDecodeError] = None
    rows: Optional[list[list[str]]] = None
    for encoding in ("utf-8", "latin-1"):
        try:
            rows_iter = _iterate_rows(path, encoding=encoding)
            rows = list(rows_iter)
            break
        except UnicodeDecodeError as err:
            last_error = err
    if rows is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Impossible de lire le fichier.")

    if not rows:
        return list(DEFAULT_HEADERS), []

    header, *data_rows = rows
    if {h.lower() for h in header} != {h.lower() for h in DEFAULT_HEADERS}:
        data_rows = rows
        header = list(DEFAULT_HEADERS)

    filtered: List[dict[str, str]] = []
    seen_rows: set[tuple[str, ...]] = set()
    for values in data_rows:
        normalized = (values + [""] * (len(header) - len(values)))[: len(header)]
        row_key = tuple(normalized)
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        record = dict(zip(header, normalized))
        if record.get("Code postal", "") == postal_code:
            filtered.append(record)
    return header, filtered

def trim_records_to_selection(
    records: list[dict[str, str]], columns: list[str]
) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    for record in records:
        trimmed.append({column: record.get(column, "") for column in columns})
    return trimmed

def display_filtered_sample(
    records: list[dict[str, str]], columns: list[str], limit: int = 5
) -> None:
    if not records:
        print("Aucune ligne ne correspond au code postal demande.")
        return

    print(f"{len(records)} lignes correspondent au code postal.")
    for idx, record in enumerate(records[:limit], start=1):
        print(f"--- Ligne {idx} ---")
        for key in columns:
            value = record.get(key, "")
            if not value:
                continue
            print(f"{key}: {value}")

def prompt_output_path(source: Path, postal_code: str) -> Path:
    default_name = f"{source.stem}_filtered_{postal_code}.csv"
    default_path = source.with_name(default_name)
    raw_output = input(
        f"Chemin du fichier CSV de sortie [{default_path}] : "
    ).strip()
    if not raw_output:
        return default_path
    output_path = Path(raw_output).expanduser()
    if output_path.is_dir():
        return output_path / default_name
    return output_path

def write_records_as_csv(
    path: Path, header: list[str], records: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter=";")
        writer.writeheader()
        writer.writerows(records)

# Fonctions numériques et clustering

def select_numerical_features(records, columns):
    df = pd.DataFrame(records)
    numerical_columns = [
        "Valeur fonciere",
        "Surface Carrez du 1er lot",
        "Surface Carrez du 2eme lot",
        "Surface Carrez du 3eme lot",
        "Surface Carrez du 4eme lot",
        "Surface Carrez du 5eme lot",
        "Surface reelle bati",
        "Nombre pieces principales",
        "Surface terrain"
    ]
    selected = [col for col in numerical_columns if col in columns]
    if not selected:
        return pd.DataFrame(columns=selected)

    def _clean(col_ser: pd.Series) -> pd.Series:
        return pd.to_numeric(
            col_ser.astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.replace(",", ".", regex=False)
            .replace({"nan": None, "None": None}),
            errors="coerce",
        )

    df_n = df[selected].apply(_clean)
    medians = df_n.median()
    df_n = df_n.fillna(medians).fillna(0)
    return df_n

# === Ajout : parsing robuste pour une série numérique ===
def parse_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace({"": None, "nan": None, "None": None}),
        errors="coerce",
    ).fillna(0)

def find_optimal_k(X, max_k=10, plot=False):
    n_samples = X.shape[0]
    if n_samples < 2:
        return 1
    max_k = min(max_k, max(2, n_samples - 1))
    k_range = range(2, max_k + 1)

    inertias = []
    silhouettes = []
    valid_k = []
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            if len(set(labels)) < 2:
                silhouettes.append(float("nan"))
            else:
                try:
                    silhouettes.append(silhouette_score(X, labels))
                except Exception:
                    silhouettes.append(float("nan"))
            valid_k.append(k)
        except Exception:
            continue

    if not valid_k:
        return 1

    if plot:
        import numpy as _np
        import matplotlib.pyplot as _plt
        ks = list(valid_k)
        _plt.figure(figsize=(10, 4))
        _plt.subplot(1, 2, 1)
        _plt.plot(ks, inertias, marker="o")
        _plt.title("Inertie (distorsion) vs k")
        _plt.xlabel("k")
        _plt.ylabel("Inertie")
        _plt.subplot(1, 2, 2)
        _plt.plot(ks, silhouettes, marker="o", color="green")
        _plt.title("Score de silhouette vs k")
        _plt.xlabel("k")
        _plt.ylabel("Silhouette")
        _plt.tight_layout()
        _plt.show()

    import numpy as np
    sil_array = np.array(silhouettes, dtype=float)
    if not np.all(np.isnan(sil_array)):
        best_idx = int(np.nanargmax(sil_array))
        return valid_k[best_idx]
    return valid_k[0]

def cluster_properties(records, columns, max_k=10, plot=False):
    X = select_numerical_features(records, columns)
    if X.empty or X.shape[0] < 1:
        raise RuntimeError("Pas de données numériques valides pour le clustering.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = find_optimal_k(X_scaled, max_k=max_k, plot=plot)
    if k < 1:
        raise RuntimeError("k invalide calculé pour le clustering.")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, k, kmeans, scaler

# Interpretation automatique des clusters

def print_cluster_summary(records, clusters, columns):
    df = pd.DataFrame(records)
    df["Cluster"] = clusters
    num_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) or c.startswith("Surface")]
    print("\n=== Analyse descriptive des clusters ===")
    for cid in sorted(set(clusters)):
        sub = df[df["Cluster"] == cid]
        if len(sub) == 0:
            continue
        print(f"\n--- Cluster {cid} ({len(sub)} transactions) ---")
        means = sub[num_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
        # Affiche les 5 variables les plus élevées pour ce cluster
        print("Moyennes (top 5 variables) :")
        print(means.head(5).to_string())
    print("\nVariables les plus discriminantes (analyse manuelle recommandée dans le rapport).")

# Régression par classe
def train_regression_by_cluster(records: list[dict[str, str]], selected_columns: list[str]) -> Dict[int, LinearRegression]:
    df = pd.DataFrame(records)
    if "Cluster" not in df.columns:
        raise RuntimeError("Pas de colonne Cluster dans les enregistrements.")
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(-1).astype(int)
    X_all = select_numerical_features(records, selected_columns)
    if "Valeur fonciere" not in df.columns:
        raise RuntimeError("Pas de colonne Valeur fonciere pour la régression.")
    y_all = parse_numeric_series(df["Valeur fonciere"])
    models = {}
    for cluster_id in sorted(df["Cluster"].unique()):
        if cluster_id < 0:
            continue
        idx = df["Cluster"] == cluster_id
        X_cluster = X_all[idx]
        y_cluster = y_all[idx]
        if len(X_cluster) < 2:
            print(f"Cluster {cluster_id} trop petit pour entrainer un modèle (n={len(X_cluster)})")
            continue
        model = LinearRegression()
        model.fit(X_cluster, y_cluster)
        models[cluster_id] = model
        print(f"Modèle entraîné pour Cluster {cluster_id} (n={len(X_cluster)})")
        # Sauvegarde modèle
        joblib.dump(model, f"regression_model_cluster_{cluster_id}.joblib")
    return models

# === Modifié : prédiction/évaluation — parsing robuste pour y_true ===
def predict_and_evaluate(records: list[dict[str, str]], models: Dict[int, LinearRegression], selected_columns: list[str]) -> None:
    df = pd.DataFrame(records)
    if "Cluster" not in df.columns:
        raise RuntimeError("Pas de colonne Cluster dans les enregistrements.")
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(-1).astype(int)
    X_all = select_numerical_features(records, selected_columns)
    if "Valeur fonciere" not in df.columns:
        raise RuntimeError("Pas de colonne Valeur fonciere pour l'évaluation.")
    y_true = parse_numeric_series(df["Valeur fonciere"])
    y_pred = np.zeros(len(df))
    print("\n=== Évaluation des modèles ===")
    for cluster_id, model in models.items():
        idx = df["Cluster"] == cluster_id
        if idx.sum() == 0:
            continue
        X_cluster = X_all[idx]
        preds = model.predict(X_cluster)
        y_pred[idx] = preds
        rmse = np.sqrt(mean_squared_error(y_true[idx], preds))
        r2 = r2_score(y_true[idx], preds)
        print(f"Cluster {cluster_id}: RMSE={rmse:.2f}, R²={r2:.2f}")
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)
    print(f"\nGlobal: RMSE={overall_rmse:.2f}, R²={overall_r2:.2f}")

# --- Optionnel : estimation pondérée (partie 5) ---

def predict_weighted(records, models, kmeans=None, scaler=None, selected_columns=None):
    """
    Donne une estimation pondérée en fonction de la probabilité d'appartenance à chaque classe (clusters KMeans).
    """
    if kmeans is None or scaler is None:
        raise ValueError("KMeans et StandardScaler nécessaires pour la pondération.")
    df = pd.DataFrame(records)
    X = select_numerical_features(records, selected_columns)
    X_scaled = scaler.transform(X)
    # Probabilités : à partir des distances inverses à chaque centroïde
    distances = kmeans.transform(X_scaled)
    inverse_dist = 1 / (distances + 1e-10)
    probs = inverse_dist / inverse_dist.sum(axis=1, keepdims=True)
    final_preds = np.zeros(len(df))
    for i, row in enumerate(X_scaled):
        pred = 0
        for cid, model in models.items():
            pred += model.predict([row])[0] * probs[i, cid]
        final_preds[i] = pred
    # Ajoute la prédiction pondérée dans les enregistrements
    for rec, v in zip(records, final_preds):
        rec["Estimation pondérée"] = f"{v:.2f}"
    print("\nPrédictions pondérées (estimation avancée) ajoutées à records.")


def main() -> None:
    path = prompt_file_path()
    postal_code = prompt_postal_code()
    header, records = load_rows_by_postal_code(path, postal_code)
    selected_header = [column for column in SELECTED_COLUMNS if column in header]
    trimmed_records = trim_records_to_selection(records, selected_header)
    display_filtered_sample(trimmed_records, selected_header)
    if not records:
        return

    print("\n[CLUSTERING NON SUPERVISÉ]")
    try:
        clusters, k, kmeans, scaler = cluster_properties(trimmed_records, selected_header, max_k=8, plot=True)
    except Exception as e:
        print(f"Erreur clustering : {e}")
        return
    for rec, clust in zip(trimmed_records, clusters):
        rec["Cluster"] = int(clust)
    print(f"{k} clusters détectés.")
    print_cluster_summary(trimmed_records, clusters, selected_header)          # <--- amélioration majeure !    

    print("\n[RÉGRESSION PAR CLASSE]")
    regression_models = train_regression_by_cluster(trimmed_records, selected_header)
    if not regression_models:
        print("Aucun modèle de régression n'a pu être entraîné.")
        return
    predict_and_evaluate(trimmed_records, regression_models, selected_header)
    
    # Ajout partie 5 optionnelle
    if regression_models and kmeans and scaler:
        predict_weighted(trimmed_records, regression_models, kmeans, scaler, selected_header)
        if "Estimation pondérée" not in selected_header:
            selected_header.append("Estimation pondérée")

    output_path = prompt_output_path(path, postal_code)
    if "Cluster" not in selected_header:
        selected_header.append("Cluster")
    write_records_as_csv(output_path, selected_header, trimmed_records)
    print(f"CSV écrit (avec clusters & pondération) dans {output_path.resolve()}")

if __name__ == "__main__":
    main()

