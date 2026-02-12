
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# plots/mortality_plots.py


# plots/mortality_plots.py

import numpy as np
import matplotlib.pyplot as plt


def plot_mortality_surface(
    mxt_raw=None,
    muxt=None,
    ages=None,
    years=None,
    *,
    region=None,
    gender="T",
    indicator="DEATHRATE",
    age_year_pivot_table=None,
    log_scale=True,
    cmap="viridis",
    figsize=(10, 6),
    show=True,
    ax=None
):
    """
    Plot a 3D mortality surface by age and year.

    The function automatically builds (muxt, ages, years) using
    age_year_pivot_table if they are not provided.

    Parameters
    ----------
    mxt_raw : DataFrame, optional
        Raw Eurostat mortality table
    muxt : array-like or DataFrame, optional
        Mortality rates (age x year)
    ages : array-like, optional
        Age vector
    years : array-like, optional
        Year vector
    region : str, optional
        Region code (required if mxt_raw is used)
    gender : str, default 'T'
    indicator : str, default 'DEATHRATE'
    age_year_pivot_table : callable, optional
        Function to build age-year pivot table
    log_scale : bool
        Plot log-mortality
    cmap : str
        Colormap
    figsize : tuple
    show : bool
        Call plt.show()
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """

    # ------------------------------------------------------------------
    # 1. Build muxt, ages, years ONLY if needed
    # ------------------------------------------------------------------
    if muxt is None or ages is None or years is None:
        if mxt_raw is None or age_year_pivot_table is None:
            raise ValueError(
                "Either provide (muxt, ages, years) OR "
                "(mxt_raw, region, age_year_pivot_table)"
            )

        muxt, ages, years = age_year_pivot_table(
            mxt_raw,
            region,
            gender,
            indicator
        )

    # ------------------------------------------------------------------
    # 2. Prepare grid
    # ------------------------------------------------------------------
    X, Y = np.meshgrid(years, ages)

    Z = muxt.values if hasattr(muxt, "values") else muxt
    if log_scale:
        Z = np.log(np.clip(Z, 1e-12, None))

    # ------------------------------------------------------------------
    # 3. Create plot
    # ------------------------------------------------------------------
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        edgecolor="none"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Age")
    ax.set_zlabel("Log-mortality rate" if log_scale else "Mortality rate")
    ax.set_title(f"Mortality surface – {region} ({gender})")

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=8)

    if show:
        plt.show()

    return ax


def plot_mortality_map(
    shapef,
    mxt_raw,
    year,
    age,
    gender="T",
    indicator="DEATHRATE",
    cmap="viridis",
    figsize=(10, 8)
):
    """
    Plot geographic map of mortality rates by region
    for a given age and year.
    """

    records = []

    for reg in shapef["NUTS_ID"]:
        muxt, ages, years = age_year_pivot_table(
            mxt_raw, reg, gender, indicator
        )

        if age in ages and year in years:
            i = np.where(ages == age)[0][0]
            j = np.where(years == year)[0][0]
            records.append((reg, muxt.iloc[i, j]))

    df_map = pd.DataFrame(records, columns=["NUTS_ID", "mu"])
    shapef_plot = shapef.merge(df_map, on="NUTS_ID", how="left")

    fig, ax = plt.subplots(figsize=figsize)
    shapef_plot.plot(
        column="mu",
        cmap=cmap,
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey"}
    )

    ax.set_title(f"Mortalité âge {age}, année {year}")
    ax.axis("off")
    plt.show()

