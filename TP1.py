"""Outils de chargement pour le TP immobilier."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
            # On retire uniquement les retours a la ligne pour conserver les colonnes vides.
            line = raw_line.rstrip("\r\n")
            yield [cell.strip() for cell in line.split(DELIMITER)]


def load_rows_by_postal_code(path: Path, postal_code: str) -> tuple[list[str], list[dict[str, str]]]:
    """
    Charge et filtre les lignes correspondant au code postal cible.

    La lecture tente d'abord UTF-8 puis bascule sur Latin-1 si necessaire.
    """
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
        # Le fichier ne contient probablement pas d'en-tete : on le traite comme tel.
        data_rows = rows
        header = list(DEFAULT_HEADERS)

    filtered: List[dict[str, str]] = []
    seen_rows: set[tuple[str, ...]] = set()
    for values in data_rows:
        # Normalise la longueur de chaque ligne et elimine les doublons complets.
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
    """Conserve uniquement les colonnes selectionnees dans chaque enregistrement."""
    trimmed: list[dict[str, str]] = []
    for record in records:
        trimmed.append({column: record.get(column, "") for column in columns})
    return trimmed


def display_filtered_sample(
    records: list[dict[str, str]], columns: list[str], limit: int = 5
) -> None:
    """Affiche un extrait des enregistrements filtres."""
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
    """Propose un chemin de sortie par defaut et accepte l'override utilisateur."""
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
    """Ecrit les enregistrements filtres dans un fichier CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter=";")
        writer.writeheader()
        writer.writerows(records)


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
        # Pas de colonnes numériques sélectionnées -> DataFrame vide
        return pd.DataFrame(columns=selected)

    # Nettoyage robuste : supprime espaces, remplace virgules par points, conversion numérique
    def _clean(col_ser: pd.Series) -> pd.Series:
        return pd.to_numeric(
            col_ser.astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.replace(",", ".", regex=False)
            .replace({"nan": None, "None": None}),
            errors="coerce",
        )

    df_n = df[selected].apply(_clean)

    # Remplacement des NaN par la médiane de la colonne; si toute la colonne est NaN -> 0
    medians = df_n.median()
    df_n = df_n.fillna(medians).fillna(0)
    return df_n

def find_optimal_k(X, max_k=10, plot=False):
    # Limiter k en fonction du nombre d'échantillons
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
            # silhouette_score nécessite au moins 2 labels distincts
            if len(set(labels)) < 2:
                silhouettes.append(float("nan"))
            else:
                try:
                    silhouettes.append(silhouette_score(X, labels))
                except Exception:
                    silhouettes.append(float("nan"))
            valid_k.append(k)
        except Exception:
            # ignorer ce k si erreur (ex: trop de clusters pour peu d'échantillons)
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

    # Choix du k : si silhouette dispo, on maximise, sinon on prend le plus petit k testé (2)
    import math
    sil_array = _np.array(silhouettes, dtype=float)
    if not _np.all(_np.isnan(sil_array)):
        best_idx = int(_np.nanargmax(sil_array))
        return valid_k[best_idx]
    # fallback
    return valid_k[0]

def cluster_properties(records, columns, max_k=10, plot=False):
    # Sélection et standardisation
    X = select_numerical_features(records, columns)
    if X.empty or X.shape[0] < 1:
        raise RuntimeError("Pas de données numériques valides pour le clustering.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Choix k optimal
    k = find_optimal_k(X_scaled, max_k=max_k, plot=plot)
    if k < 1:
        raise RuntimeError("k invalide calculé pour le clustering.")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    # Assignation aux data
    return clusters, k, kmeans, scaler

def main():
    path = prompt_file_path()
    postal_code = prompt_postal_code()
    header, records = load_rows_by_postal_code(path, postal_code)
    selected_header = [column for column in SELECTED_COLUMNS if column in header]
    trimmed_records = trim_records_to_selection(records, selected_header)
    display_filtered_sample(trimmed_records, selected_header)
    if not records:
        return
    
    # ====== CLUSTERING (Partie 3) =======
    print("\n[CLUSTERING NON SUPERVISÉ]")
    try:
        clusters, k, model, scaler = cluster_properties(trimmed_records, selected_header, max_k=8, plot=True)
    except Exception as e:
        print(f"Erreur clustering : {e}")
        return

    for rec, clust in zip(trimmed_records, clusters):
        rec["Cluster"] = str(clust)

    print(f"{k} clusters détectés.")
    # Affiche un aperçu des clusters trouvés dans les premiers exemples
    for idx, rec in enumerate(trimmed_records[:5]):
        print(f"Ligne {idx+1}: Cluster {rec['Cluster']}")
    
    # ====================================
    
    output_path = prompt_output_path(path, postal_code)
    # Ajoute 'Cluster' à l'entête si absent
    if "Cluster" not in selected_header:
        selected_header.append("Cluster")
    write_records_as_csv(output_path, selected_header, trimmed_records)
    print(f"CSV écrit (avec clusters) dans {output_path.resolve()}")

'''
def main() -> None:
    """Point d'entree simple pour tester la recuperation."""
    path = prompt_file_path()
    postal_code = prompt_postal_code()
    header, records = load_rows_by_postal_code(path, postal_code)
    selected_header = [column for column in SELECTED_COLUMNS if column in header]
    trimmed_records = trim_records_to_selection(records, selected_header)
    display_filtered_sample(trimmed_records, selected_header)
    if not records:
        return
    output_path = prompt_output_path(path, postal_code)
    write_records_as_csv(output_path, selected_header, trimmed_records)
    print(f"CSV ecrit dans {output_path.resolve()}")
'''

if __name__ == "__main__":
    main()

