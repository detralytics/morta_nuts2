import numpy as np
import pandas as pd
import geopandas as gpd
import os
from concurrent.futures import ThreadPoolExecutor


shapef = gpd.read_file("C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp")
regions   = shapef["NUTS_ID"].tolist()
France_outremer = list(['FRY1','FRY2','FRY3','FRY4','FRY5'])
regions = [item for item in regions if item not in France_outremer]
FRANCE_OUTREMER = list(['FRY1','FRY2','FRY3','FRY4','FRY5'])


# =============================================================================
# Fonction de Donatien mais pas optimisées : temps de d'exécution à 11 min utilisant ces fonctions 
# =============================================================================

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

def age_year_pivot_table_values(data_raw,region,gender):
    # Sub-dataframe
    tab  = data_raw[ (data_raw['geo']==region) & (data_raw['sex']==gender)] 
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




# =============================================================================
# Construction de la fonction sans correction de l'annomalie sur les ages et l'age maxi non identifié
# =============================================================================


"""
compute_brut_mortality_by_region — v3
======================================
Pré-pivot global adapté aux colonnes exactes :
  mxt_raw : geo / sex / indic_de / age / time / values
  Lxt_raw : geo / sex / age / time / values

Logique mathématique (Extg, Dxtg) strictement identique.
"""




# ---------------------------------------------------------------------------
# 1. PRÉ-PIVOT GLOBAL — remplace N appels à age_year_pivot_table
# ---------------------------------------------------------------------------
def _build_pivot_mu(mxt_raw, gender, common_ages):
    """
    Equivalent vectorisé de N appels à age_year_pivot_table(reg, gender, 'DEATHRATE') de la fonction de donatien.
    Retourne dict { region -> DataFrame(age x year) }, âges triés, valeurs float.
    fill_value=1e-6 identique à l'original.
    """
    sub = mxt_raw[
        (mxt_raw["sex"]      == gender)     &
        (mxt_raw["indic_de"] == "DEATHRATE") &
        (mxt_raw["age"].isin(common_ages))
    ]
    # Un seul pivot sur tout le jeu de données
    pivot = pd.pivot_table(
        sub,
        values="values",
        index=["geo", "age"],
        columns="time",
        aggfunc="sum",
        fill_value=1e-6,
        observed=True,
    )
    pivot.columns = pivot.columns.astype(int)
    pivot.index   = pivot.index.set_levels(
        pivot.index.levels[1].astype(int), level=1
    )
    # Sépare par région, trie les âges (sort_values("age") de l'original)
    return {
        reg: grp.droplevel(0).sort_index()
        for reg, grp in pivot.groupby(level="geo")
    }


def _build_pivot_L(Lxt_raw, gender, common_ages):
    """
    Equivalent vectorisé de N appels à age_year_pivot_table_values(reg, gender) de la fonction de donatien.
    """
    sub = Lxt_raw[
        (Lxt_raw["sex"] == gender) &
        (Lxt_raw["age"].isin(common_ages))
    ]
    pivot = pd.pivot_table(
        sub,
        values="values",
        index=["geo", "age"],
        columns="time",
        aggfunc="sum",
        fill_value=1e-6,
        observed=True,
    )
    pivot.columns = pivot.columns.astype(int)
    pivot.index   = pivot.index.set_levels(
        pivot.index.levels[1].astype(int), level=1
    )
    return {
        reg: grp.droplevel(0).sort_index()
        for reg, grp in pivot.groupby(level="geo")
    }


# ---------------------------------------------------------------------------
# 2. CALCUL PAR RÉGION — 100 % numpy
# ---------------------------------------------------------------------------
def _process_region_numpy(reg, mu_df, L_df):
    """
    mu_df, L_df : DataFrames (age x year) issus du pré-pivot.
    Retourne (reg, years, D_t, E_t) ou None.
    """
    # Alignement années
    common_years = mu_df.columns.intersection(L_df.columns)
    if len(common_years) == 0:
        return None

    # Alignement âges
    common_ages = mu_df.index.intersection(L_df.index)
    if len(common_ages) == 0:
        return None

    Mu = mu_df.loc[common_ages, common_years].values   # (A, T)
    L  = L_df.loc[common_ages, common_years].values    # (A, T)

    # Exposition — copie stricte de Extg
    E = np.empty_like(L)
    E[0, :]  = L[0, :]
    E[1:, :] = (L[1:, :] + L[:-1, :]) / 2
    E[-1, :] = L[-1, :]

    # Décès & agrégation âge
    D_t = np.einsum("at,at->t", Mu, E)   # = (Mu * E).sum(axis=0)
    E_t = E.sum(axis=0)

    return reg, common_years.values, D_t, E_t


# ---------------------------------------------------------------------------
# 3. FONCTION PRINCIPALE
# ---------------------------------------------------------------------------
def brut_mortality_by_region(
    mxt_raw,
    Lxt_raw,
    Dxt_raw,
    regions,
    gender="T",
    country="FR",
    n_jobs: int = -1,
):
    """
    Calcule par région et par année :
      - décès
      - exposition
      - taux de mortalité brute

    Méthodologie strictement identique au code 3D (Extg, Dxtg).
    """
    # =========================================================
    # 0. Filtrage régions
    # =========================================================
    if country == "FR":
        regions = [r for r in regions if r not in FRANCE_OUTREMER]

    # =========================================================
    # 1. HARMONISATION DES ÂGES (inchangée)
    # =========================================================
    mxt_g = mxt_raw[mxt_raw["sex"] == gender]
    Lxt_g = Lxt_raw[Lxt_raw["sex"] == gender]
    Dxt_g = Dxt_raw[Dxt_raw["sex"] == gender]

    common_ages = sorted(
        set(mxt_g["age"].unique())
        & set(Lxt_g["age"].unique())
        & set(Dxt_g["age"].unique())
    )
    if not common_ages:
        raise ValueError("Aucun âge commun après harmonisation")

    # =========================================================
    # 2. PRÉ-PIVOT GLOBAL (2 pivots en parallèle)
    # =========================================================
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_mu = pool.submit(_build_pivot_mu, mxt_raw, gender, common_ages)
        fut_L  = pool.submit(_build_pivot_L,  Lxt_raw, gender, common_ages)
        mu_by_region = fut_mu.result()
        L_by_region  = fut_L.result()

    # =========================================================
    # 3. BOUCLE RÉGIONALE NUMPY
    # =========================================================
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 1, len(regions))

    valid_regions = [r for r in regions if r in mu_by_region and r in L_by_region]

    def _worker(reg):
        return _process_region_numpy(reg, mu_by_region[reg], L_by_region[reg])

    raw_results = []
    if n_jobs == 1:
        for reg in valid_regions:
            r = _worker(reg)
            if r is not None:
                raw_results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            for r in pool.map(_worker, valid_regions):
                if r is not None:
                    raw_results.append(r)

    if not raw_results:
        raise ValueError("Aucune région exploitable après alignement âge/année")

    # =========================================================
    # 4. DATAFRAME FINAL EN UNE PASSE (concat numpy → 1 seul DataFrame)
    # =========================================================
    all_regions  = np.repeat([r[0] for r in raw_results], [len(r[1]) for r in raw_results])
    all_years    = np.concatenate([r[1] for r in raw_results])
    all_deaths   = np.concatenate([r[2] for r in raw_results])
    all_exposure = np.concatenate([r[3] for r in raw_results])

    return pd.DataFrame({
        "region":        all_regions,
        "year":          all_years,
        "deaths":        all_deaths,
        "exposure":      all_exposure,
        "mortality_rate": all_deaths / all_exposure,
    })


"""
compute_brut_mortality_by_region — avec correction d'annomilies sur les ages et ages max à 82 ans
======================================
Pré-pivot global adapté aux colonnes exactes :
  mxt_raw : geo / sex / indic_de / age / time / values
  Lxt_raw : geo / sex / age / time / values

Logique mathématique (Extg, Dxtg) strictement identique.
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# 1. PRÉ-PIVOT GLOBAL — remplace N appels à age_year_pivot_table
# ---------------------------------------------------------------------------
def _build_pivot_mu(mxt_raw, gender, common_ages):
    """
    Equivalent vectorisé de N appels à age_year_pivot_table(reg, gender, 'DEATHRATE').
    Retourne dict { region -> DataFrame(age x year) }, âges triés, valeurs float.
    fill_value=1e-6 identique à l'original.
    """
    sub = mxt_raw[
        (mxt_raw["sex"]      == gender)     &
        (mxt_raw["indic_de"] == "DEATHRATE") &
        (mxt_raw["age"].isin(common_ages))
    ]
    # Un seul pivot sur tout le jeu de données
    pivot = pd.pivot_table(
        sub,
        values="values",
        index=["geo", "age"],
        columns="time",
        aggfunc="sum",
        fill_value=1e-6,
        observed=True,
    )
    pivot.columns = pivot.columns.astype(int)
    pivot.index   = pivot.index.set_levels(
        pivot.index.levels[1].astype(int), level=1
    )
    # Sépare par région, trie les âges (sort_values("age") de l'original)
    return {
        reg: grp.droplevel(0).sort_index()
        for reg, grp in pivot.groupby(level="geo")
    }


def _build_pivot_L(Lxt_raw, gender, common_ages):
    """
    Equivalent vectorisé de N appels à age_year_pivot_table_values(reg, gender).
    """
    sub = Lxt_raw[
        (Lxt_raw["sex"] == gender) &
        (Lxt_raw["age"].isin(common_ages))
    ]
    pivot = pd.pivot_table(
        sub,
        values="values",
        index=["geo", "age"],
        columns="time",
        aggfunc="sum",
        fill_value=1e-6,
        observed=True,
    )
    pivot.columns = pivot.columns.astype(int)
    pivot.index   = pivot.index.set_levels(
        pivot.index.levels[1].astype(int), level=1
    )
    return {
        reg: grp.droplevel(0).sort_index()
        for reg, grp in pivot.groupby(level="geo")
    }


# Âges anomaliques à interpoler (mêmes valeurs que le code 3D)
_AGE_ADJ  = np.array([11, 22, 33, 44, 55, 66, 77])
# Plage d'âges conservée (0–82 inclus, identique à age_range du code 3D)
_AGE_MAX  = 82
_AGE_RANGE = np.arange(0, _AGE_MAX + 1)          # 0..82


# ---------------------------------------------------------------------------
# 2. CALCUL PAR RÉGION — 100 % numpy
# ---------------------------------------------------------------------------
def _process_region_numpy_corrige(reg, mu_df, L_df):
    """
    mu_df, L_df : DataFrames (age x year) issus du pré-pivot.
    Pipeline :
      1. alignement années / âges
      2. correction anomalies (âges 11,22,33,44,55,66,77) sur Mu
      3. troncature âge max = 82
      4. exposition (Extg)
      5. décès & agrégation
    Retourne (reg, years, D_t, E_t) ou None.
    """
    # ----------------------------------------------------------
    # 1. Alignement années
    # ----------------------------------------------------------
    common_years = mu_df.columns.intersection(L_df.columns)
    if len(common_years) == 0:
        return None

    # Alignement âges
    common_ages = mu_df.index.intersection(L_df.index)
    if len(common_ages) == 0:
        return None

    Mu = mu_df.loc[common_ages, common_years].values.copy()  # (A, T) — copie car on va modifier
    L  = L_df.loc[common_ages, common_years].values          # (A, T)
    age_arr = np.array(common_ages)                          # vecteur des âges alignés

    # ----------------------------------------------------------
    # 2. CORRECTION ANOMALIES SUR MU — miroir exact du code 3D
    #
    # PROBLÈME : les âges 11, 22, 33, 44, 55, 66, 77 (multiples de 11)
    #
    # SOLUTION : interpolation linéaire ponctuelle — on remplace le taux
    # aberrant par la moyenne de ses deux voisins directs (âge-1, âge+1),
    # sur toutes les années simultanément :
    #
    #   Mu[a, :] = (Mu[a+1, :] + Mu[a-1, :]) / 2
    #
    # Exemple pour a=22 :
    #   Mu[22, :] = (Mu[23, :] + Mu[21, :]) / 2   ← pour toutes les années
    #
    # NOTE : la correction porte uniquement sur Mu (taux).
    # Les décès D = Mu * E sont recalculés automatiquement à l'étape 5
    # avec le Mu corrigé — pas besoin de recalcul explicite ici.
    #
    # GARDE : on ne corrige que si les trois âges (a-1, a, a+1) sont
    # tous présents dans le vecteur age_arr de la région courante.
    # ----------------------------------------------------------
    for a in _AGE_ADJ:
        idx   = np.searchsorted(age_arr, a)
        idx_p = np.searchsorted(age_arr, a + 1)
        idx_m = np.searchsorted(age_arr, a - 1)
        # Vérification que les trois âges voisins sont bien disponibles
        if (idx   < len(age_arr) and age_arr[idx]   == a     and
            idx_p < len(age_arr) and age_arr[idx_p] == a + 1 and
            idx_m >= 0           and age_arr[idx_m] == a - 1):
            Mu[idx, :] = (Mu[idx_p, :] + Mu[idx_m, :]) / 2

    # ----------------------------------------------------------
    # 3. TRONCATURE ÂGE MAX = 82 — miroir de age_range = np.arange(0,83)
    # ----------------------------------------------------------
    mask    = age_arr <= _AGE_MAX
    age_arr = age_arr[mask]
    Mu      = Mu[mask, :]
    L       = L[mask, :]

    if len(age_arr) == 0:
        return None

    # ----------------------------------------------------------
    # 4. EXPOSITION — copie stricte de Extg
    # ----------------------------------------------------------
    E = np.empty_like(L)
    E[0, :]  = L[0, :]
    E[1:, :] = (L[1:, :] + L[:-1, :]) / 2
    E[-1, :] = L[-1, :]

    # ----------------------------------------------------------
    # 5. DÉCÈS & AGRÉGATION ÂGE
    # ----------------------------------------------------------
    D_t = np.einsum("at,at->t", Mu, E)   # = (Mu * E).sum(axis=0)
    E_t = E.sum(axis=0)

    return reg, common_years.values, D_t, E_t


# ---------------------------------------------------------------------------
# 3. FONCTION PRINCIPALE
# ---------------------------------------------------------------------------
def mortality_by_region(
    mxt_raw,
    Lxt_raw,
    Dxt_raw,
    regions,
    gender="T",
    country="FR",
    n_jobs: int = -1,
):
    """
    Calcule par région et par année :
      - décès
      - exposition
      - taux de mortalité brute

    Méthodologie strictement identique au code 3D (Extg, Dxtg).
    """
    # =========================================================
    # 0. Filtrage régions
    # =========================================================
    if country == "FR":
        regions = [r for r in regions if r not in FRANCE_OUTREMER]
    else:
        regions =regions

    # =========================================================
    # 1. HARMONISATION DES ÂGES (inchangée)
    # =========================================================
    mxt_g = mxt_raw[mxt_raw["sex"] == gender]
    Lxt_g = Lxt_raw[Lxt_raw["sex"] == gender]
    Dxt_g = Dxt_raw[Dxt_raw["sex"] == gender]

    common_ages = sorted(
        set(mxt_g["age"].unique())
        & set(Lxt_g["age"].unique())
        & set(Dxt_g["age"].unique())
    )
    if not common_ages:
        raise ValueError("Aucun âge commun après harmonisation")

    # =========================================================
    # 2. PRÉ-PIVOT GLOBAL (2 pivots en parallèle)
    # =========================================================
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_mu = pool.submit(_build_pivot_mu, mxt_raw, gender, common_ages)
        fut_L  = pool.submit(_build_pivot_L,  Lxt_raw, gender, common_ages)
        mu_by_region = fut_mu.result()
        L_by_region  = fut_L.result()

    # =========================================================
    # 3. BOUCLE RÉGIONALE NUMPY
    # =========================================================
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 1, len(regions))

    valid_regions = [r for r in regions if r in mu_by_region and r in L_by_region]

    def _worker(reg):
        return _process_region_numpy_corrige(reg, mu_by_region[reg], L_by_region[reg])

    raw_results = []
    if n_jobs == 1:
        for reg in valid_regions:
            r = _worker(reg)
            if r is not None:
                raw_results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            for r in pool.map(_worker, valid_regions):
                if r is not None:
                    raw_results.append(r)

    if not raw_results:
        raise ValueError("Aucune région exploitable après alignement âge/année")

    # =========================================================
    # 4. DATAFRAME FINAL EN UNE PASSE (concat numpy → 1 seul DataFrame)
    # =========================================================
    all_regions  = np.repeat([r[0] for r in raw_results], [len(r[1]) for r in raw_results])
    all_years    = np.concatenate([r[1] for r in raw_results])
    all_deaths   = np.concatenate([r[2] for r in raw_results])
    all_exposure = np.concatenate([r[3] for r in raw_results])

    return pd.DataFrame({
        "region":        all_regions,
        "year":          all_years,
        "deaths":        all_deaths,
        "exposure":      all_exposure,
        "mortality_rate": all_deaths / all_exposure,
    })


# ---------------------------------------------------------------------------
# 2. CALCUL PAR RÉGION — 100 % numpy (SANS AGRÉGATION PAR ÂGE)
# ---------------------------------------------------------------------------
def _process_region_numpy_corrige_sans_agg(reg, mu_df, L_df):
    """
    mu_df, L_df : DataFrames (age x year) issus du pré-pivot.
    Pipeline :
      1. alignement années / âges
      2. correction anomalies (âges 11,22,33,44,55,66,77) sur Mu
      3. troncature âge max = 82
      4. exposition (Extg)
      5. décès (sans agrégation)
    Retourne (reg, years, ages, D, E) ou None.
    """

    # ----------------------------------------------------------
    # 1. Alignement années
    # ----------------------------------------------------------
    common_years = mu_df.columns.intersection(L_df.columns)
    if len(common_years) == 0:
        return None

    common_ages = mu_df.index.intersection(L_df.index)
    if len(common_ages) == 0:
        return None

    Mu = mu_df.loc[common_ages, common_years].values.copy()
    L  = L_df.loc[common_ages, common_years].values
    age_arr = np.array(common_ages)

    # ----------------------------------------------------------
    # 2. CORRECTION ANOMALIES SUR MU
    # ----------------------------------------------------------
    for a in _AGE_ADJ:
        idx   = np.searchsorted(age_arr, a)
        idx_p = np.searchsorted(age_arr, a + 1)
        idx_m = np.searchsorted(age_arr, a - 1)

        if (idx   < len(age_arr) and age_arr[idx]   == a     and
            idx_p < len(age_arr) and age_arr[idx_p] == a + 1 and
            idx_m >= 0           and age_arr[idx_m] == a - 1):
            Mu[idx, :] = (Mu[idx_p, :] + Mu[idx_m, :]) / 2

    # ----------------------------------------------------------
    # 3. TRONCATURE ÂGE MAX = 82
    # ----------------------------------------------------------
    mask    = age_arr <= _AGE_MAX
    age_arr = age_arr[mask]
    Mu      = Mu[mask, :]
    L       = L[mask, :]

    if len(age_arr) == 0:
        return None

    # ----------------------------------------------------------
    # 4. EXPOSITION (Extg identique au code 3D)
    # ----------------------------------------------------------
    E = np.empty_like(L)
    E[0, :]  = L[0, :]
    E[1:, :] = (L[1:, :] + L[:-1, :]) / 2
    E[-1, :] = L[-1, :]

    # ----------------------------------------------------------
    # 5. DÉCÈS (SANS AGRÉGATION)
    # ----------------------------------------------------------
    D = Mu * E  # matrice (age x year)

    return reg, common_years.values, age_arr, D, E


# ---------------------------------------------------------------------------
# 3. FONCTION PRINCIPALE
# ---------------------------------------------------------------------------
def mortality_by_region_by_age(
    mxt_raw,
    Lxt_raw,
    Dxt_raw,
    regions,
    gender="T",
    country="FR",
    n_jobs: int = -1,
):
    """
    Calcule par région, par âge et par année :
      - décès
      - exposition
      - taux de mortalité brute

    Méthodologie strictement identique au code 3D (Extg, Dxtg),
    mais SANS agrégation par âge.
    """

    # =========================================================
    # 0. Filtrage régions
    # =========================================================
    if country == "FR":
        regions = [r for r in regions if r not in FRANCE_OUTREMER]

    # =========================================================
    # 1. HARMONISATION DES ÂGES
    # =========================================================
    mxt_g = mxt_raw[mxt_raw["sex"] == gender]
    Lxt_g = Lxt_raw[Lxt_raw["sex"] == gender]
    Dxt_g = Dxt_raw[Dxt_raw["sex"] == gender]

    common_ages = sorted(
        set(mxt_g["age"].unique())
        & set(Lxt_g["age"].unique())
        & set(Dxt_g["age"].unique())
    )

    if not common_ages:
        raise ValueError("Aucun âge commun après harmonisation")

    # =========================================================
    # 2. PRÉ-PIVOT GLOBAL
    # =========================================================
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_mu = pool.submit(_build_pivot_mu, mxt_raw, gender, common_ages)
        fut_L  = pool.submit(_build_pivot_L,  Lxt_raw, gender, common_ages)
        mu_by_region = fut_mu.result()
        L_by_region  = fut_L.result()

    # =========================================================
    # 3. BOUCLE RÉGIONALE NUMPY
    # =========================================================
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 1, len(regions))

    valid_regions = [r for r in regions if r in mu_by_region and r in L_by_region]

    def _worker(reg):
        return _process_region_numpy_corrige_sans_agg(reg, mu_by_region[reg], L_by_region[reg])

    raw_results = []

    if n_jobs == 1:
        for reg in valid_regions:
            r = _worker(reg)
            if r is not None:
                raw_results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            for r in pool.map(_worker, valid_regions):
                if r is not None:
                    raw_results.append(r)

    if not raw_results:
        raise ValueError("Aucune région exploitable après alignement âge/année")

    # =========================================================
    # 4. CONSTRUCTION DATAFRAME FINAL (RÉGION × ANNÉE × ÂGE)
    # =========================================================
    regions_list = []
    years_list = []
    ages_list = []
    deaths_list = []
    exposure_list = []

    for reg, years, ages, D, E in raw_results:
        A, T = D.shape

        regions_list.append(np.repeat(reg, A * T))
        years_list.append(np.tile(years, A))
        ages_list.append(np.repeat(ages, T))
        deaths_list.append(D.flatten())
        exposure_list.append(E.flatten())

    all_regions  = np.concatenate(regions_list)
    all_years    = np.concatenate(years_list)
    all_ages     = np.concatenate(ages_list)
    all_deaths   = np.concatenate(deaths_list)
    all_exposure = np.concatenate(exposure_list)

    return pd.DataFrame({
        "region":         all_regions,
        "year":           all_years,
        "age":            all_ages,
        "deaths":         all_deaths,
        "exposure":       all_exposure,
        "mortality_rate": all_deaths / all_exposure,
    })

