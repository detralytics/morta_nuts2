import pandas as pd
from eurostatapiclient import EurostatAPIClient
import geopandas as gpd
import numpy as np
import os



# =============================================================================
# Utils
# =============================================================================

def parse_age(label):
    if isinstance(label, str):
        if label.startswith("Y") and label[1:].isdigit():
            return int(label[1:])
        elif label == "Y_LT1":
            return 0
        elif label == "Y_GE85":
            return 85
    return None


def age_year_pivot_table(data_raw,region,gender,indicator):
    # Sub-dataframe
    tab  = data_raw[ (data_raw['geo']==region) & (data_raw['sex']==gender) & \
                    (data_raw['indic_de']==indicator)]
    tab  = tab.reset_index(drop=True)
    # Pivot table 
    tab  = pd.pivot_table(tab, values='values', index=['age'],
                 columns=['time'], aggfunc="sum", fill_value=1e-6,observed=True)
    # We sort by ascending ages
    tab  = tab.sort_values("age").reset_index(drop=True)
    # surface of log-mortality
    ages  = tab.index.values.astype('int')
    years = tab.columns.values.astype('int')
    return tab , ages , years 



def filter_shapefile(
    shapef,
    country="FR",
    nuts_level=2,
    exclude_outremer=True
):
    """
    Filter a NUTS shapefile by country and NUTS level.
    Default: France, NUTS2, without overseas regions.
    """
    shapef = shapef.copy()

    shapef = shapef[
        (shapef["CNTR_CODE"] == country) &
        (shapef["LEVL_CODE"] == nuts_level)
    ]

    if country == "FR" and exclude_outremer:
        shapef = shapef[
            ~shapef["NUTS_ID"].isin(["FRY1","FRY2","FRY3","FRY4","FRY5"])
        ]

    return shapef.sort_values("NUTS_ID").reset_index(drop=True)


# =============================================================================
# Core Eurostat loader
# =============================================================================

def _load_eurostat_dataset_fr(
    dataset,
    regions,
    filename,
    parse_age,
    data_path,
    download=False,
    language="en"
):
    """
    Generic Eurostat loader with auto-download and safe paths.
    """
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, filename)

    client = EurostatAPIClient("1.0", "json", language)

    if download or not os.path.exists(file_path):
        frames = []
        for reg in regions:
            resp = client.get_dataset(dataset, params={"geo": reg})
            df = resp.to_dataframe()
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)
        data.to_csv(file_path, index=False)
    else:
        data = pd.read_csv(file_path)

    # ---------------- Cleaning ----------------
    if "age" in data.columns:
        data = data[~data["age"].isin(["TOTAL", "UNK", "Y_OPEN"])]
        data["age"] = data["age"].map(parse_age)
        data = data.dropna(subset=["age"])
        data["age"] = data["age"].astype(int)

    if "time" in data.columns:
        data["time"] = data["time"].astype(int)

    return data.reset_index(drop=True)


# =============================================================================
# Public API — one function per dataset
# =============================================================================

def load_mxt_raw_fr(
    shapef,
    country="FR",
    nuts_level=2,
    data_path="../data",
    download=False
):
    shapef_f = filter_shapefile(shapef, country, nuts_level)
    regions = shapef_f["NUTS_ID"].tolist()

    return _load_eurostat_dataset_fr(
        dataset="demo_r_mlife",
        regions=regions,
        filename="mxt_raw.csv",
        parse_age=parse_age,
        data_path=data_path,
        download=download
    )


def load_Dxt_raw_fr(
    shapef,
    country="FR",
    nuts_level=2,
    data_path="../data",
    download=False
):
    shapef_f = filter_shapefile(shapef, country, nuts_level)
    regions = shapef_f["NUTS_ID"].tolist()

    return _load_eurostat_dataset_fr(
        dataset="demo_r_magec",
        regions=regions,
        filename="Dxt_raw.csv",
        parse_age=parse_age,
        data_path=data_path,
        download=download
    )


def load_Lxt_raw_fr(
    shapef,
    country="FR",
    nuts_level=2,
    data_path="../data",
    download=False
):
    shapef_f = filter_shapefile(shapef, country, nuts_level)
    regions = shapef_f["NUTS_ID"].tolist()

    return _load_eurostat_dataset_fr(
        dataset="demo_r_d2jan",
        regions=regions,
        filename="Lxt_raw.csv",
        parse_age=parse_age,
        data_path=data_path,
        download=download
    )
