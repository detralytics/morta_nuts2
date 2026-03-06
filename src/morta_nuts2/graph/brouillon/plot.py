import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_models_by_region(
    x_values,
    curves_dict,
    year_to_plot,
    tv,
    regions,
    n_cols=3,
    yscale="log",
    xlabel="Age",
    ylabel="Mortality rate",
    title_prefix="Comparison"
):
    """
    Fonction générique pour tracer plusieurs modèles par région.
    
    Parameters
    ----------
    x_values : array (abscisse)
    curves_dict : dict {label: 3D array (age, year, region)}
    year_to_plot : année à tracer
    tv : vecteur des années
    regions : liste des régions
    n_cols : nombre de colonnes des subplots
    yscale : "log" ou "linear"
    """
    
    if year_to_plot not in tv:
        raise ValueError(f"Année {year_to_plot} non disponible.")
    
    t_index = np.where(tv == year_to_plot)[0][0]
    
    n_regions = len(regions)
    n_rows = int(np.ceil(n_regions / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = axes.flatten()
    
    for g, region_name in enumerate(regions):
        
        ax_plot = axes[g]
        
        for label, array3d in curves_dict.items():
            
            ax_plot.plot(
                x_values,
                array3d[:, t_index, g],
                label=label,linestyle=":" 
            )
        
        ax_plot.set_title(region_name)
        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel(ylabel)
        ax_plot.set_yscale(yscale)
        ax_plot.grid(True)
        ax_plot.legend(fontsize=8)
    
    # Supprimer axes inutilisés
    for i in range(n_regions, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"{title_prefix} - Year {year_to_plot}", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_map_compare_years(
    regions, data, tv_future,
    country_code, indicator_name,
    years=[2023, 2050],
    age=0, cmap="Blues", nuts_level=2,
):
    
    shapef = gpd.read_file("C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp")
    if data.ndim == 3:
        data = data[age, :, :]

    shp = shapef[
        (shapef["CNTR_CODE"] == country_code) &
        (shapef["LEVL_CODE"] == nuts_level)
    ].copy().to_crs(epsg=2154)

    all_vals = []
    dfs = []
    for yr in years:
        idx    = np.where(tv_future == yr)[0][0]
        values = data[idx, :]
        all_vals.append(values)
        df = shp.merge(
            gpd.GeoDataFrame({"NUTS_ID": regions, "value": values}),
            on="NUTS_ID", how="left"
        )
        dfs.append(df)

    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(years),
        figsize=(7 * len(years), 8),
        gridspec_kw={"bottom": 0.15}   # ✅ espace réservé en bas
    )

    for i, (df, yr) in enumerate(zip(dfs, years)):
        df.plot(
            column    = "value",
            cmap      = cmap,
            linewidth = 0.5,
            edgecolor = "black",
            ax        = axes[i],
            norm      = norm,
            legend    = False,
        )
        axes[i].set_title(str(yr), fontsize=13, fontweight="bold")
        axes[i].axis("off")

    # ✅ Colorbar dans axe dédié hors des cartes
    cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=indicator_name)

    plt.suptitle(f"{indicator_name} — {country_code}", fontsize=14, fontweight="bold", y=0.98)
    #plt.savefig("map.png", bbox_inches="tight", dpi=150)
    plt.show()



def plot_map_indicator(
    regions, data, tv_future,
    country_code, indicator_name,
    year=2023,
    age=0, cmap="Blues", nuts_level=2,
):
    shapef = gpd.read_file("C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp")
    
    if data.ndim == 3:
        data = data[age, :, :]   # (horizon, nb_regions)

    shp = shapef[
        (shapef["CNTR_CODE"] == country_code) &
        (shapef["LEVL_CODE"] == nuts_level)
    ].copy().to_crs(epsg=2154)

    idx    = np.where(tv_future == year)[0][0]
    values = data[idx, :]

    df = shp.merge(
        gpd.GeoDataFrame({"NUTS_ID": regions, "value": values}),
        on="NUTS_ID", how="left"
    )

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(8, 9), gridspec_kw={"bottom": 0.15})

    df.plot(
        column    = "value",
        cmap      = cmap,
        linewidth = 0.5,
        edgecolor = "black",
        ax        = ax,
        norm      = norm,
        legend    = False,
    )
    ax.set_title(str(year), fontsize=13, fontweight="bold")
    ax.axis("off")

    cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=indicator_name)

    plt.suptitle(f"{indicator_name} — {country_code}", fontsize=14, fontweight="bold", y=0.98)
    #plt.savefig("map.png", bbox_inches="tight", dpi=150)
    plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import geopandas as gpd
# import numpy as np

SHAPEF_PATH = "C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp"

# ═══════════════════════════════════════════════════════════════════
# FUNCTION 1 : Regional summary statistics
# ═══════════════════════════════════════════════════════════════════
def compute_regional_stats(ex, regions, tv_future, age=0):
    """
    ex         : (nb_ages, horizon, nb_regions)
    regions    : array of region codes
    tv_future  : array of projection years
    age        : target age (0 = e0, 65 = e65)

    Returns a dict of DataFrames :
      - 'by_year'   : aggregated stats per year   (horizon, stats)
      - 'by_region' : aggregated stats per region (nb_regions, stats)
      - 'full'      : full DataFrame              (horizon x nb_regions)
    """
    data = ex[age, :, :]   # (horizon, nb_regions)

    # ── Full DataFrame ─────────────────────────────────────────────
    df_full = pd.DataFrame(data, index=tv_future, columns=regions)
    df_full.index.name = "year"

    # ── Stats per year ─────────────────────────────────────────────
    df_by_year = pd.DataFrame({
        "mean"        : data.mean(axis=1),
        "std"         : data.std(axis=1),
        "min"         : data.min(axis=1),
        "max"         : data.max(axis=1),
        "range"       : data.max(axis=1) - data.min(axis=1),
        "cv"          : data.std(axis=1) / data.mean(axis=1) * 100,  # coefficient of variation
        "region_min"  : regions[data.argmin(axis=1)],
        "region_max"  : regions[data.argmax(axis=1)],
    }, index=tv_future)
    df_by_year.index.name = "year"

    # ── Stats per region ───────────────────────────────────────────
    df_by_region = pd.DataFrame({
        "mean"        : data.mean(axis=0),
        "std"         : data.std(axis=0),
        "min"         : data.min(axis=0),
        "max"         : data.max(axis=0),
        "avg_rank"    : pd.DataFrame(
                            np.argsort(np.argsort(-data, axis=1), axis=1),
                            columns=regions
                          ).mean(axis=0) + 1,
    }, index=regions)
    df_by_region.index.name = "region"
    df_by_region = df_by_region.sort_values("mean", ascending=False)

    return {
        "full"      : df_full,
        "by_year"   : df_by_year,
        "by_region" : df_by_region,
    }


# ═══════════════════════════════════════════════════════════════════
# FUNCTION 2 : Map of deviations from national average
# ═══════════════════════════════════════════════════════════════════
# def plot_map_deviation(
#     regions, data, tv_future,
#     country_code, indicator_name,
#     year=2023, age=0, nuts_level=2,
# ):
#     """
#     Choropleth map showing deviations from the national average.
#     Blue  = above average
#     Red   = below average
#     """
#     shapef = gpd.read_file(SHAPEF_PATH)

#     if data.ndim == 3:
#         data = data[age, :, :]   # (horizon, nb_regions)

#     # Filter shapefile by country and NUTS level, reproject to Lambert-93
#     shp = shapef[
#         (shapef["CNTR_CODE"] == country_code) &
#         (shapef["LEVL_CODE"] == nuts_level)
#     ].copy().to_crs(epsg=2154)

#     # Compute deviations from national mean for the selected year
#     idx        = np.where(tv_future == year)[0][0]
#     values     = data[idx, :]
#     deviations = values - values.mean()

#     # Merge deviations into the shapefile
#     df = shp.merge(
#         gpd.GeoDataFrame({"NUTS_ID": regions, "value": deviations}),
#         on="NUTS_ID", how="left"
#     )

#     # Symmetric colormap centered on 0
#     vabs = np.nanmax(np.abs(deviations))
#     norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

#     fig, ax = plt.subplots(1, 1, figsize=(8, 9), gridspec_kw={"bottom": 0.15})

#     df.plot(
#         column      = "value",
#         cmap        = "RdBu",
#         linewidth   = 0.5,
#         edgecolor   = "black",
#         ax          = ax,
#         norm        = norm,
#         legend      = False,
#         missing_kwds= {"color": "lightgrey"},
#     )
#     ax.set_title(str(year), fontsize=13, fontweight="bold")
#     ax.axis("off")

#     # Dedicated colorbar axis at the bottom
#     cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
#     sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
#                  label=f"Deviation from national average ({indicator_name})")

#     plt.suptitle(
#         f"Deviations from average — {indicator_name} — {country_code} ({year})",
#         fontsize=13, fontweight="bold", y=0.98
#     )
#     #plt.savefig("map_deviation.png", bbox_inches="tight", dpi=150)
#     plt.show()


# ═══════════════════════════════════════════════════════════════════
# FUNCTION 3 : Inter-regional dispersion over time
# ═══════════════════════════════════════════════════════════════════
def plot_dispersion_over_time(
    ex, regions, tv_future,
    indicator_name="Life expectancy",
    ages=[0, 65],
):
    """
    Plots the evolution of inter-regional dispersion over projection years.
    Shows standard deviation and range for each target age.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = plt.cm.tab10.colors

    for k, age in enumerate(ages):
        data   = ex[age, :, :]                          # (horizon, nb_regions)
        std    = data.std(axis=1)
        range_ = data.max(axis=1) - data.min(axis=1)
        label  = "e₀" if age == 0 else f"e{age}"

        axes[0].plot(tv_future, std,    color=colors[k], lw=2, label=label)
        axes[1].plot(tv_future, range_, color=colors[k], lw=2, label=label)

    # Axis labels and formatting
    for ax, title in zip(axes, ["Inter-regional std dev", "Range (max − min)"]):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Years of life")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Inter-regional convergence — {indicator_name}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    #plt.savefig("dispersion.png", bbox_inches="tight", dpi=150)
    plt.show()


    import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass, field


# ── Default color palette (cyclic if more series than colors) ────────────────
_DEFAULT_COLORS = [
    {"facecolor": "#1a1a2e", "edgecolor": "#e94560", "median": "#e94560"},  # red
    {"facecolor": "#fff8e1", "edgecolor": "#e67e22", "median": "#e67e22"},  # orange
    {"facecolor": "#e8f5e9", "edgecolor": "#27ae60", "median": "#27ae60"},  # green
    {"facecolor": "#f3e5f5", "edgecolor": "#8e44ad", "median": "#8e44ad"},  # purple
]

_REGIONAL_STYLE = {
    "facecolor": "#eaf2fb",
    "edgecolor": "#2c3e50",
    "median":    "#f39c12",
}


# ── Dataclass describing an extra series ─────────────────────────────────────
@dataclass
class ExtraSeries:
    """
    An extra series to overlay on top of the regional boxplots.

    Parameters
    ----------
    data     : ndarray (nb_xe, 1, nb_simul)
    label    : str   – display name (legend + tick label)
    position : "first" | "last"  – where to insert the boxplot
    style    : dict | None  – optional visual override
                 valid keys: facecolor, edgecolor, median, linewidth
    """
    data:     np.ndarray
    label:    str
    position: str  = "last"
    style:    dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_style(user_style: dict, default: dict, linewidth: float = 2.2) -> dict:
    merged = {**default, **user_style}
    merged.setdefault("linewidth", linewidth)
    return merged


def _boxplot_kwargs(style: dict, alpha: float = 0.88) -> dict:
    lw = style.get("linewidth", 1.8)
    return dict(
        patch_artist=True,
        boxprops=dict(facecolor=style["facecolor"], color=style["edgecolor"],
                      linewidth=lw, alpha=alpha),
        medianprops=dict(color=style["median"], linewidth=lw + 0.5),
        whiskerprops=dict(color=style["edgecolor"], linewidth=lw),
        capprops=dict(color=style["edgecolor"], linewidth=lw),
        flierprops=dict(marker="o", markerfacecolor=style["edgecolor"],
                        markeredgecolor=style["edgecolor"],
                        markersize=2.5, alpha=0.35, linestyle="none"),
        zorder=3,
    )


# ── Main function ────────────────────────────────────────────────────────────
def plot_annuity_boxplot(
    price_regional: np.ndarray,
    region_names:   list,
    extra_series:   list = None,
    xe_idx:         int   = 0,
    duration:       int   = 20,
    age:            int   = 60,
    figsize:        tuple = (18, 7),
    show:           bool  = True,
    ax              = None,
):
    """
    Plot regional boxplots + any number of extra series.

    Parameters
    ----------
    price_regional : ndarray (nb_xe, nb_reg, nb_simul)
    region_names   : list[str]
    extra_series   : list[ExtraSeries] | None
                     Pass [] or None to display regions only.
                     Examples:
                       [ExtraSeries(price_nat,   "National",  position="last")]
                       [ExtraSeries(price_nat,   "National",  position="last"),
                        ExtraSeries(price_bench, "Benchmark", position="first")]
    xe_idx         : age index
    duration / age : used in the chart title
    show           : calls plt.show() if True
    ax             : existing axes (creates a new figure if None)

    Returns
    -------
    fig, ax
    """
    extra_series = extra_series or []
    nb_reg = price_regional.shape[1]

    # Auto-assign colors to each extra series (cyclic)
    for k, es in enumerate(extra_series):
        default_col = _DEFAULT_COLORS[k % len(_DEFAULT_COLORS)]
        es.style = _resolve_style(es.style, default_col)

    # ── Ordered sequence: firsts | regions | lasts ──────────────────────────
    regional_entries = [
        (price_regional[xe_idx, g, :],
         _resolve_style({}, _REGIONAL_STYLE, linewidth=1.5),
         region_names[g], False)
        for g in range(nb_reg)
    ]
    firsts = [
        (es.data[xe_idx, 0, :], es.style, es.label, True)
        for es in extra_series if es.position == "first"
    ]
    lasts = [
        (es.data[xe_idx, 0, :], es.style, es.label, True)
        for es in extra_series if es.position == "last"
    ]
    series = firsts + regional_entries + lasts

    # ── Figure ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, (data, style, _, _) in enumerate(series, start=1):
        ax.boxplot(data, positions=[i], widths=0.55, **_boxplot_kwargs(style))

    # Visual separators between blocks
    n_first, n_last, n_total = len(firsts), len(lasts), len(series)
    if n_first:
        ax.axvline(n_first + 0.5, color="#bbb", linewidth=1.1, linestyle=":", zorder=1)
    if n_last:
        ax.axvline(n_total - n_last + 0.5, color="#bbb", linewidth=1.1, linestyle=":", zorder=1)

    # ── Axes & styling ────────────────────────────────────────────────────────
    tick_labels = [s[2] for s in series]
    ax.set_xticks(range(1, n_total + 1))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(0.3, n_total + 0.7)
    ax.set_title(f"Annuity price — {age} y.o. | duration {duration}y",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Region", fontsize=11, labelpad=8)
    ax.set_ylabel("Price (€)", fontsize=11, labelpad=8)
    ax.grid(True, axis="y", alpha=0.22, linestyle="--")
    ax.set_facecolor("#fafafa")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=_REGIONAL_STYLE["facecolor"],
                       edgecolor=_REGIONAL_STYLE["edgecolor"], label="Regions")
    ]
    for es in extra_series:
        legend_handles.append(
            mpatches.Patch(facecolor=es.style["facecolor"],
                           edgecolor=es.style["edgecolor"], label=es.label)
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    try:
        from IPython.display import display as ipy_display
        ipy_display(fig)
    except ImportError:
        if show:
            plt.show()
    plt.close(fig)
    return fig, ax


# ── Usage examples ───────────────────────────────────────────────────────────
# 0 série extra
# plot_annuity_boxplot(price_reg, names)

# # 1 série extra
# plot_annuity_boxplot(price_reg, names, extra_series=[
#     ExtraSeries(price_nat, "National", position="last"),
# ])

# # 2 séries extra (positions différentes)
# plot_annuity_boxplot(price_reg, names, extra_series=[
#     ExtraSeries(price_nat,   "National",  position="last"),
#     ExtraSeries(price_bench, "Benchmark", position="first"),
# ])

# # Style custom sur une série
# ExtraSeries(price_nat, "National", style={"edgecolor": "#00e5ff", "median": "#00e5ff"})