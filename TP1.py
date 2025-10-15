"""Outils de chargement pour le TP immobilier."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

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
    """Lit le fichier ligne par ligne avec un encodage specifique."""
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            yield [cell.strip() for cell in row]


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
    for values in data_rows:
        # Complete les lignes trop courtes avec des champs vides.
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        record = dict(zip(header, values))
        if record.get("Code postal", "") == postal_code:
            filtered.append(record)
    return header, filtered


def display_filtered_sample(records: list[dict[str, str]], limit: int = 5) -> None:
    """Affiche un extrait des enregistrements filtres."""
    if not records:
        print("Aucune ligne ne correspond au code postal demande.")
        return

    print(f"{len(records)} lignes correspondent au code postal.")
    for idx, record in enumerate(records[:limit], start=1):
        print(f"--- Ligne {idx} ---")
        for key, value in record.items():
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


def write_records_as_csv(path: Path, header: list[str], records: list[dict[str, str]]) -> None:
    """Ecrit les enregistrements filtres dans un fichier CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Point d'entree simple pour tester la recuperation."""
    path = prompt_file_path()
    postal_code = prompt_postal_code()
    header, records = load_rows_by_postal_code(path, postal_code)
    display_filtered_sample(records)
    if not records:
        return
    output_path = prompt_output_path(path, postal_code)
    write_records_as_csv(output_path, header, records)
    print(f"CSV ecrit dans {output_path.resolve()}")


if __name__ == "__main__":
    main()
