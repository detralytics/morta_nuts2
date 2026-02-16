# =============================================================================
# Utils
# =============================================================================

import pandas as pd
from eurostatapiclient import EurostatAPIClient
import geopandas as gpd
import numpy as np
import os


def parse_age(label):
    if isinstance(label, str):
        if label.startswith("Y") and label[1:].isdigit():
            return int(label[1:])
        elif label == "Y_LT1":
            return 0
        elif label == "Y_GE85":
            return 85
    return None


def age_year_pivot_table(data_raw, region, gender, indicator):
    # Sub-dataframe
    tab = data_raw[
        (data_raw['geo']      == region) &
        (data_raw['sex']      == gender) &
        (data_raw['indic_de'] == indicator)
    ]
    tab = tab.reset_index(drop=True)
    # Pivot table
    tab = pd.pivot_table(
        tab, values='values', index=['age'],
        columns=['time'], aggfunc="sum", fill_value=1e-6, observed=True
    )
    # We sort by ascending ages
    tab   = tab.sort_values("age").reset_index(drop=True)
    ages  = tab.index.values.astype('int')
    years = tab.columns.values.astype('int')
    return tab, ages, years


# Régions à exclure par pays (outremers, territoires non continentaux, etc.)
EXCLUDE_REGIONS = {
    "FR": ["FRY1", "FRY2", "FRY3", "FRY4", "FRY5"],  # DOM-TOM français
    "PT": ["PT20", "PT30"],                             # Açores, Madère
    "ES": ["ES63", "ES64", "ES70"],                     # Ceuta, Melilla, Canaries
    "NO": ["NO0B"],                                     # Svalbard
    # Ajouter d'autres pays ici si besoin
}


def filter_shapefile(
    shapef,
    country="FR",
    nuts_level=2,
    exclude_outremer=True
):
    """
    Filtre un shapefile NUTS par pays et niveau NUTS.
    Fonctionne pour n'importe quel pays (FR, BE, DE, ES, IT...).

    Paramètres
    ----------
    shapef          : GeoDataFrame NUTS
    country         : code pays ISO 2 lettres (ex. "FR", "BE", "DE")
    nuts_level      : niveau NUTS (1, 2 ou 3)
    exclude_outremer: si True, exclut les régions non continentales
                      définies dans EXCLUDE_REGIONS pour ce pays.
                      Si le pays n'est pas dans EXCLUDE_REGIONS,
                      aucune région n'est exclue.
    """
    shapef = shapef.copy()

    shapef = shapef[
        (shapef["CNTR_CODE"] == country) &
        (shapef["LEVL_CODE"] == nuts_level)
    ]

    if exclude_outremer and country in EXCLUDE_REGIONS:
        shapef = shapef[
            ~shapef["NUTS_ID"].isin(EXCLUDE_REGIONS[country])
        ]

    return shapef.sort_values("NUTS_ID").reset_index(drop=True)


# =============================================================================
# Core Eurostat loader
# =============================================================================

def _load_eurostat_dataset(
    dataset,
    regions,
    filename,
    parse_age,
    data_path,
    country="FR",
    download=False,
    language="en"
):
    """
    Chargeur Eurostat générique avec auto-download et nommage par pays.

    Le fichier est sauvegardé sous  <data_path>/<country>_<filename>.csv
    ce qui évite qu'un téléchargement Belgique écrase le fichier France.

    Exemple : data/FR_mxt_raw.csv  /  data/BE_mxt_raw.csv
    """
    os.makedirs(data_path, exist_ok=True)

    # Nom de fichier unique par pays  →  FR_mxt_raw.csv, BE_mxt_raw.csv ...
    safe_filename = f"{country}_{filename}.csv"
    file_path     = os.path.join(data_path, safe_filename)

    client = EurostatAPIClient("1.0", "json", language)

    if download or not os.path.exists(file_path):
        frames = []
        for reg in regions:
            resp = client.get_dataset(dataset, params={"geo": reg})
            df   = resp.to_dataframe()
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)
        data.to_csv(file_path, index=False)
        print(f"[Eurostat] Téléchargé → {file_path}  ({len(data):,} lignes)")
    else:
        data = pd.read_csv(file_path)
        print(f"[Eurostat] Chargé depuis cache → {file_path}  ({len(data):,} lignes)")

    # ---------------- Nettoyage ----------------
    if "age" in data.columns:
        data = data[~data["age"].isin(["TOTAL", "UNK", "Y_OPEN"])]
        data["age"] = data["age"].map(parse_age)
        data = data.dropna(subset=["age"])
        data["age"] = data["age"].astype(int)

    if "time" in data.columns:
        data["time"] = data["time"].astype(int)

    return data.reset_index(drop=True)


# =============================================================================
# API publique — une fonction par dataset, country libre
# =============================================================================

def load_mxt_raw(
    shapef,
    country="FR",
    nuts_level=2,
    data_path="../data",
    download=False
):
    """
    Charge les taux de mortalité (demo_r_mlife) pour le pays choisi.

    Exemples
    --------
    load_mxt_raw(shapef, country="FR")   # France
    load_mxt_raw(shapef, country="BE")   # Belgique
    load_mxt_raw(shapef, country="DE")   # Allemagne
    """
    shapef_f = filter_shapefile(shapef, country, nuts_level)
    regions  = shapef_f["NUTS_ID"].tolist()

    return _load_eurostat_dataset(
        dataset   = "demo_r_mlife",
        regions   = regions,
        filename  = f"mxt_raw_{country}",
        parse_age = parse_age,
        data_path = data_path,
        country   = country,
        download  = download,
    )


def load_dxt_raw(
    shapef,
    country="FR",
    nuts_level=2,
    data_path="../data",
    download=False
):
    """
    Charge les décès (demo_r_magec) pour le pays choisi.
    """
    shapef_f = filter_shapefile(shapef, country, nuts_level)
    regions  = shapef_f["NUTS_ID"].tolist()

    return _load_eurostat_dataset(
        dataset   = "demo_r_magec",
        regions   = regions,
        filename  = f"Dxt_raw_{country}",
        parse_age = parse_age,
        data_path = data_path,
        country   = country,
        download  = download,
    )


def load_lxt_raw(
    shapef,
    country="FR",
    nuts_level=2,
    data_path="../data",
    download=False
):
    """
    Charge les populations (demo_r_d2jan) pour le pays choisi.
    """
    shapef_f = filter_shapefile(shapef, country, nuts_level)
    regions  = shapef_f["NUTS_ID"].tolist()

    return _load_eurostat_dataset(
        dataset   = "demo_r_d2jan",
        regions   = regions,
        filename  = f"Lxt_raw_{country}",
        parse_age = parse_age,
        data_path = data_path,
        country   = country,
        download  = download,
    )




