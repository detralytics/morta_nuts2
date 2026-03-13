

"""
Plotter
=====================================================

"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Default color palette for extra series (cycles if more series than colors) ──
_DEFAULT_COLORS = [
    {"facecolor": "#1a1a2e", "edgecolor": "#e94560", "median": "#e94560"},  # red
    {"facecolor": "#fff8e1", "edgecolor": "#e67e22", "median": "#e67e22"},  # orange
    {"facecolor": "#e8f5e9", "edgecolor": "#27ae60", "median": "#27ae60"},  # green
    {"facecolor": "#f3e5f5", "edgecolor": "#8e44ad", "median": "#8e44ad"},  # purple
]

# Default style applied to regional boxplots
_REGIONAL_STYLE = {
    "facecolor": "#eaf2fb",
    "edgecolor": "#2c3e50",
    "median":    "#f39c12",
}

from pathlib import Path

BASE_DIR = Path.cwd()
while not (BASE_DIR / "NUTS_files").exists():
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "NUTS_files" / "NUTS_RG_01M_2024_3035.shp"
# Path to the NUTS shapefile used for choropleth maps
SHAPEF_PATH = DATA_DIR


# ── Helper dataclass describing an extra series overlaid on regional boxplots ──
@dataclass
class ExtraSeries:
    """
    Describes an additional data series to overlay on top of the regional boxplots.

    Parameters
    ----------
    data     : ndarray of shape (nb_xe, 1, nb_simul)
    label    : str — display name used in the legend and tick labels
    position : "first" | "last" — where the boxplot is inserted relative to regions
    style    : dict | None — optional visual override
                 valid keys: facecolor, edgecolor, median, linewidth
    """
    data:     np.ndarray
    label:    str
    position: str  = "last"
    style:    dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
# CLASS 1 : Regional curve comparison
# ════════════════════════════════════════════════════════════════════
class RegionalCurvePlotter:
    """
    Plots multiple model curves for each region as a grid of subplots.

    Parameters
    ----------
    x_values     : array-like — x-axis values (e.g. age bins)
    curves_dict  : dict mapping label -> 3D array of shape (age, year, region)
    tv           : 1D array of available projection years
    regions      : list of region names
    n_cols       : number of columns in the subplot grid (default: 3)
    yscale       : y-axis scale, "log" or "linear" (default: "log")
    xlabel       : label for the x-axis (default: "Age")
    ylabel       : label for the y-axis (default: "Mortality rate")
    title_prefix : prefix string used in the figure title (default: "Comparison")
    """

    def __init__(
        self,
        x_values,
        curves_dict,
        tv,
        regions,
        n_cols=3,
        yscale="log",
        xlabel="Age",
        ylabel="Mortality rate",
        title_prefix="Comparison",
    ):
        self.x_values     = x_values
        self.curves_dict  = curves_dict
        self.tv           = tv
        self.regions      = regions
        self.n_cols       = n_cols
        self.yscale       = yscale
        self.xlabel       = xlabel
        self.ylabel       = ylabel
        self.title_prefix = title_prefix

    def plot(self, year_to_plot):
        """
        Render the subplot grid for the given projection year.

        Parameters
        ----------
        year_to_plot : int — the year to display (must exist in self.tv)
        """
        if year_to_plot not in self.tv:
            raise ValueError(f"Year {year_to_plot} is not available in tv.")

        # Find the time index corresponding to the requested year
        t_index = np.where(self.tv == year_to_plot)[0][0]

        n_regions = len(self.regions)
        n_rows    = int(np.ceil(n_regions / self.n_cols))

        fig, axes = plt.subplots(n_rows, self.n_cols, figsize=(15, 4 * n_rows))

        # Flatten axes array for uniform indexing regardless of grid shape
        if n_rows == 1:
            axes = np.array(axes).reshape(-1)
        else:
            axes = axes.flatten()

        for g, region_name in enumerate(self.regions):
            ax_plot = axes[g]

            # Plot each model curve for the current region
            for label, array3d in self.curves_dict.items():
                ax_plot.plot(
                    self.x_values,
                    array3d[:, t_index, g],
                    label=label,
                    linestyle=":",
                )

            ax_plot.set_title(region_name)
            ax_plot.set_xlabel(self.xlabel)
            ax_plot.set_ylabel(self.ylabel)
            ax_plot.set_yscale(self.yscale)
            ax_plot.grid(True)
            ax_plot.legend(fontsize=8)

        # Remove unused subplot slots when regions don't fill the grid
        for i in range(n_regions, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f"{self.title_prefix} - Year {year_to_plot}", fontsize=16)
        plt.tight_layout()
        plt.show()


# ════════════════════════════════════════════════════════════════════
# CLASS 2 : Choropleth map plotter
# ════════════════════════════════════════════════════════════════════
class MapPlotter:
    """
    Renders choropleth maps of a given indicator at the NUTS regional level.

    Parameters
    ----------
    regions        : array-like of NUTS region codes
    data           : ndarray — shape (nb_ages, horizon, nb_regions) or (horizon, nb_regions)
    tv_future      : 1D array of projection years
    country_code   : ISO country code used to filter the shapefile (e.g. "FR")
    indicator_name : string label for the colorbar and figure title
    shapef_path    : path to the NUTS shapefile (defaults to SHAPEF_PATH)
    cmap           : matplotlib colormap name (default: "Blues")
    nuts_level     : NUTS level to display (default: 2)
    age            : age index to slice when data is 3-dimensional (default: 0)
    """

    def __init__(
        self,
        regions,
        data,
        tv_future,
        country_code,
        indicator_name,
        shapef_path=SHAPEF_PATH,
        cmap="Blues",
        nuts_level=2,
        age=0,
    ):
        self.regions        = regions
        self.tv_future      = tv_future
        self.country_code   = country_code
        self.indicator_name = indicator_name
        self.shapef_path    = shapef_path
        self.cmap           = cmap
        self.nuts_level     = nuts_level
        self.age            = age  # stored for use in plot_static

        # Reduce to 2D (horizon, nb_regions) if a 3D array is provided
        self.data = data[age, :, :] if data.ndim == 3 else data

    def _load_shapefile(self):
        """Load the NUTS shapefile and filter to the target country and NUTS level."""
        shapef = gpd.read_file(self.shapef_path)
        shp = shapef[
            (shapef["CNTR_CODE"] == self.country_code) &
            (shapef["LEVL_CODE"] == self.nuts_level)
        ].copy().to_crs(epsg=2154)  # Reproject to Lambert-93 (France)
        return shp

    def _add_colorbar(self, fig, norm):
        """Append a horizontal colorbar in a dedicated axes below the map(s)."""
        cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=self.indicator_name)

    def plot_single_year(self, year=2023):
        """
        Draw a single choropleth map for the specified year.

        Parameters
        ----------
        year : int — projection year to display (default: 2023)
        """
        shp = self._load_shapefile()

        # Extract values for the target year
        idx    = np.where(self.tv_future == year)[0][0]
        values = self.data[idx, :]

        # Merge indicator values into the shapefile geometry
        df = shp.merge(
            gpd.GeoDataFrame({"NUTS_ID": self.regions, "value": values}),
            on="NUTS_ID", how="left",
        )

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(1, 1, figsize=(8, 9), gridspec_kw={"bottom": 0.15})

        df.plot(
            column="value", cmap=self.cmap, linewidth=0.5,
            edgecolor="black", ax=ax, norm=norm, legend=False,
        )
        ax.set_title(str(year), fontsize=13, fontweight="bold")
        ax.axis("off")

        self._add_colorbar(fig, norm)
        plt.suptitle(
            f"{self.indicator_name} — {self.country_code}",
            fontsize=14, fontweight="bold", y=0.98,
        )
        plt.show()

    def plot_compare_years(self, years=None):
        """
        Draw side-by-side choropleth maps for several years using a shared color scale.

        Parameters
        ----------
        years : list of int — years to compare (default: [2023, 2050])
        """
        if years is None:
            years = [2023, 2050]

        shp = self._load_shapefile()

        # Build a merged GeoDataFrame and collect values for each target year
        all_vals, dfs = [], []
        for yr in years:
            idx    = np.where(self.tv_future == yr)[0][0]
            values = self.data[idx, :]
            all_vals.append(values)
            df = shp.merge(
                gpd.GeoDataFrame({"NUTS_ID": self.regions, "value": values}),
                on="NUTS_ID", how="left",
            )
            dfs.append(df)

        # Use a common normalization range across all years for fair comparison
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(
            1, len(years),
            figsize=(7 * len(years), 8),
            gridspec_kw={"bottom": 0.15},  # Reserve space at the bottom for the colorbar
        )

        for i, (df, yr) in enumerate(zip(dfs, years)):
            df.plot(
                column="value", cmap=self.cmap, linewidth=0.5,
                edgecolor="black", ax=axes[i], norm=norm, legend=False,
            )
            axes[i].set_title(str(yr), fontsize=13, fontweight="bold")
            axes[i].axis("off")

        self._add_colorbar(fig, norm)
        plt.suptitle(
            f"{self.indicator_name} — {self.country_code}",
            fontsize=14, fontweight="bold", y=0.98,
        )
        plt.show()

    def plot_static(self, static_data, title_suffix=""):
        """
        Draw a choropleth map from data that has no time dimension.
        Accepts formats: (nb_regions,) or (nb_ages, nb_regions).

        Parameters
        ----------
        static_data  : ndarray — shape (nb_regions,) or (nb_ages, nb_regions)
                       When 2D, the age index used at __init__ time is applied.
        title_suffix : str — optional subtitle appended to the figure title
                       (e.g. "Mean error")
        """
        shp = self._load_shapefile()

        # Handle (nb_ages, nb_regions) → (nb_regions,)
        if static_data.ndim == 2:
            values = static_data[self.age, :]
        elif static_data.ndim == 1:
            values = static_data
        else:
            raise ValueError(
                f"static_data must be 1D (nb_regions,) or 2D (nb_ages, nb_regions), "
                f"got shape {static_data.shape}"
            )

        df = shp.merge(
            gpd.GeoDataFrame({"NUTS_ID": self.regions, "value": values}),
            on="NUTS_ID", how="left",
        )

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(1, 1, figsize=(8, 9), gridspec_kw={"bottom": 0.15})

        df.plot(
            column="value", cmap=self.cmap, linewidth=0.5,
            edgecolor="black", ax=ax, norm=norm, legend=False,
        )

        if title_suffix:
            ax.set_title(title_suffix, fontsize=13, fontweight="bold")
        ax.axis("off")

        self._add_colorbar(fig, norm)
        plt.suptitle(
            f"{self.indicator_name} — {self.country_code}",
            fontsize=14, fontweight="bold", y=0.98,
        )
        plt.show()

# ════════════════════════════════════════════════════════════════════
# CLASS 3 : Regional statistics
# ════════════════════════════════════════════════════════════════════
class RegionalStats:
    """
    Computes summary statistics for a regional indicator over a projection horizon.

    Parameters
    ----------
    ex        : ndarray of shape (nb_ages, horizon, nb_regions)
    regions   : array-like of region codes
    tv_future : 1D array of projection years
    age       : age index to extract (0 = e0, 65 = e65, etc.)
    """

    def __init__(self, ex, regions, tv_future, age=0):
        self.ex        = ex
        self.regions   = regions
        self.tv_future = tv_future
        self.age       = age

        # Slice the 3D array to (horizon, nb_regions) for the target age
        self.data = ex[age, :, :]

    def compute(self):
        """
        Run all aggregations and return a dict of DataFrames.

        Returns
        -------
        dict with keys:
          - "full"      : full (horizon x nb_regions) DataFrame
          - "by_year"   : aggregated statistics per projection year
          - "by_region" : aggregated statistics per region, sorted by mean descending
        """
        data    = self.data
        regions = self.regions
        tv      = self.tv_future

        # ── Full matrix ────────────────────────────────────────────
        df_full = pd.DataFrame(data, index=tv, columns=regions)
        df_full.index.name = "year"

        # ── Statistics per year ────────────────────────────────────
        df_by_year = pd.DataFrame({
            "mean"       : data.mean(axis=1),
            "std"        : data.std(axis=1),
            "min"        : data.min(axis=1),
            "max"        : data.max(axis=1),
            "range"      : data.max(axis=1) - data.min(axis=1),
            "cv"         : data.std(axis=1) / data.mean(axis=1) * 100,  # coefficient of variation (%)
            "region_min" : regions[data.argmin(axis=1)],
            "region_max" : regions[data.argmax(axis=1)],
        }, index=tv)
        df_by_year.index.name = "year"

        # ── Statistics per region ──────────────────────────────────
        df_by_region = pd.DataFrame({
            "mean"     : data.mean(axis=0),
            "std"      : data.std(axis=0),
            "min"      : data.min(axis=0),
            "max"      : data.max(axis=0),
            # Average rank across years (rank 1 = highest value each year)
            "avg_rank" : pd.DataFrame(
                             np.argsort(np.argsort(-data, axis=1), axis=1),
                             columns=regions,
                         ).mean(axis=0) + 1,
        }, index=regions)
        df_by_region.index.name = "region"
        df_by_region = df_by_region.sort_values("mean", ascending=False)

        return {
            "full"      : df_full,
            "by_year"   : df_by_year,
            "by_region" : df_by_region,
        }


# ════════════════════════════════════════════════════════════════════
# CLASS 4 : Inter-regional dispersion over time
# ════════════════════════════════════════════════════════════════════
class DispersionPlotter:
    """
    Visualises the evolution of inter-regional dispersion across projection years.

    Parameters
    ----------
    ex             : ndarray of shape (nb_ages, horizon, nb_regions)
    regions        : array-like of region codes (not directly used in the plot but
                     kept for consistency with other classes)
    tv_future      : 1D array of projection years
    indicator_name : string label used in the figure title (default: "Life expectancy")
    ages           : list of age indices to compare (default: [0, 65])
    """

    def __init__(
        self,
        ex,
        regions,
        tv_future,
        indicator_name="Life expectancy",
        ages=None,
    ):
        self.ex             = ex
        self.regions        = regions
        self.tv_future      = tv_future
        self.indicator_name = indicator_name
        self.ages           = ages if ages is not None else [0, 65]

    def plot(self):
        """
        Render two side-by-side panels:
          - left  : inter-regional standard deviation over time
          - right : inter-regional range (max − min) over time
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors    = plt.cm.tab10.colors

        for k, age in enumerate(self.ages):
            data   = self.ex[age, :, :]                   # (horizon, nb_regions)
            std    = data.std(axis=1)
            range_ = data.max(axis=1) - data.min(axis=1)
            label  = "e₀" if age == 0 else f"e{age}"

            axes[0].plot(self.tv_future, std,    color=colors[k], lw=2, label=label)
            axes[1].plot(self.tv_future, range_, color=colors[k], lw=2, label=label)

        # Apply shared formatting to both panels
        for ax, title in zip(axes, ["Inter-regional std dev", "Range (max − min)"]):
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Years of life")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Inter-regional convergence — {self.indicator_name}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()


# ════════════════════════════════════════════════════════════════════
# CLASS 5 : Annuity price boxplots
# ════════════════════════════════════════════════════════════════════

def _resolve_style(user_style: dict, default: dict, linewidth: float = 2.2) -> dict:
    """Merge user-provided style overrides with a default style dict."""
    merged = {**default, **user_style}
    merged.setdefault("linewidth", linewidth)
    return merged


def _boxplot_kwargs(style: dict, alpha: float = 0.88) -> dict:
    """Build a keyword-argument dict for ax.boxplot() from a style dict."""
    lw = style.get("linewidth", 1.8)
    return dict(
        patch_artist=True,
        boxprops    =dict(facecolor=style["facecolor"], color=style["edgecolor"],
                          linewidth=lw, alpha=alpha),
        medianprops =dict(color=style["median"], linewidth=lw + 0.5),
        whiskerprops=dict(color=style["edgecolor"], linewidth=lw),
        capprops    =dict(color=style["edgecolor"], linewidth=lw),
        flierprops  =dict(marker="o", markerfacecolor=style["edgecolor"],
                          markeredgecolor=style["edgecolor"],
                          markersize=2.5, alpha=0.35, linestyle="none"),
        zorder=3,
    )


class AnnuityBoxPlotter:
    """
    Draws regional boxplots of annuity prices, with optional extra series overlaid.

    Parameters
    ----------
    price_regional : ndarray of shape (nb_xe, nb_reg, nb_simul)
    region_names   : list of region label strings
    extra_series   : list[ExtraSeries] | None
                     Pass [] or None to display regional data only.
                     Examples:
                       [ExtraSeries(price_nat, "National", position="last")]
                       [ExtraSeries(price_nat,   "National",  position="last"),
                        ExtraSeries(price_bench, "Benchmark", position="first")]
    xe_idx         : index along the age dimension of price_regional (default: 0)
    duration       : annuity duration in years, used in the chart title (default: 20)
    age            : policyholder age, used in the chart title (default: 60)
    figsize        : figure size tuple (default: (18, 7))
    show           : whether to call plt.show() after rendering (default: True)
    """

    def __init__(
        self,
        price_regional: np.ndarray,
        region_names:   list,
        extra_series:   list = None,
        xe_idx:         int   = 0,
        duration:       int   = 20,
        age:            int   = 60,
        figsize:        tuple = (18, 7),
        show:           bool  = True,
    ):
        self.price_regional = price_regional
        self.region_names   = region_names
        self.extra_series   = extra_series or []
        self.xe_idx         = xe_idx
        self.duration       = duration
        self.age            = age
        self.figsize        = figsize
        self.show           = show

    def plot(self, ax=None):
        """
        Render the boxplot chart.

        Parameters
        ----------
        ax : matplotlib Axes | None
             Provide an existing axes to draw into; otherwise a new figure is created.

        Returns
        -------
        fig, ax : the matplotlib Figure and Axes objects
        """
        nb_reg = self.price_regional.shape[1]

        # Assign a color from the default palette to each extra series (cyclic)
        for k, es in enumerate(self.extra_series):
            default_col = _DEFAULT_COLORS[k % len(_DEFAULT_COLORS)]
            es.style = _resolve_style(es.style, default_col)

        # Build an ordered list: "first" extras → regional boxes → "last" extras
        regional_entries = [
            (self.price_regional[self.xe_idx, g, :],
             _resolve_style({}, _REGIONAL_STYLE, linewidth=1.5),
             self.region_names[g], False)
            for g in range(nb_reg)
        ]
        firsts = [
            (es.data[self.xe_idx, 0, :], es.style, es.label, True)
            for es in self.extra_series if es.position == "first"
        ]
        lasts = [
            (es.data[self.xe_idx, 0, :], es.style, es.label, True)
            for es in self.extra_series if es.position == "last"
        ]
        series = firsts + regional_entries + lasts

        # Create a new figure if no axes was provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        # Draw one boxplot per entry in the ordered series list
        for i, (data, style, _, _) in enumerate(series, start=1):
            ax.boxplot(data, positions=[i], widths=0.55, **_boxplot_kwargs(style))

        # Draw thin vertical separators between "first", regional, and "last" blocks
        n_first, n_last, n_total = len(firsts), len(lasts), len(series)
        if n_first:
            ax.axvline(n_first + 0.5, color="#bbb", linewidth=1.1, linestyle=":", zorder=1)
        if n_last:
            ax.axvline(n_total - n_last + 0.5, color="#bbb", linewidth=1.1, linestyle=":", zorder=1)

        # Axis labels and tick configuration
        tick_labels = [s[2] for s in series]
        ax.set_xticks(range(1, n_total + 1))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(0.3, n_total + 0.7)
        ax.set_title(
            f"Annuity price — {self.age} y.o. | duration {self.duration}y",
            fontsize=14, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Region", fontsize=11, labelpad=8)
        ax.set_ylabel("Price (€)", fontsize=11, labelpad=8)
        ax.grid(True, axis="y", alpha=0.22, linestyle="--")
        ax.set_facecolor("#fafafa")

        # Build the legend from patch handles
        legend_handles = [
            mpatches.Patch(
                facecolor=_REGIONAL_STYLE["facecolor"],
                edgecolor=_REGIONAL_STYLE["edgecolor"],
                label="Regions",
            )
        ]
        for es in self.extra_series:
            legend_handles.append(
                mpatches.Patch(
                    facecolor=es.style["facecolor"],
                    edgecolor=es.style["edgecolor"],
                    label=es.label,
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()

        # Display the figure (IPython-aware)
        try:
            from IPython.display import display as ipy_display
            ipy_display(fig)
        except ImportError:
            if self.show:
                plt.show()

        plt.close(fig)
        return fig, ax


# ════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════════════════════════════════

# ── RegionalCurvePlotter ────────────────────────────────────────────
# plotter = RegionalCurvePlotter(
#     x_values=ages,
#     curves_dict={"Model A": arr_a, "Model B": arr_b},
#     tv=tv_vector,
#     regions=region_list,
# )
# plotter.plot(year_to_plot=2040)

# ── MapPlotter ──────────────────────────────────────────────────────
# mp = MapPlotter(regions, data_3d, tv_future, "FR", "Life expectancy e0")
# mp.plot_single_year(year=2030)
# mp.plot_compare_years(years=[2023, 2050])

# ── RegionalStats ───────────────────────────────────────────────────
# stats = RegionalStats(ex, regions, tv_future, age=0)
# results = stats.compute()
# print(results["by_year"])

# ── DispersionPlotter ───────────────────────────────────────────────
# dp = DispersionPlotter(ex, regions, tv_future, indicator_name="Life expectancy")
# dp.plot()

# ── AnnuityBoxPlotter — no extra series ────────────────────────────
# AnnuityBoxPlotter(price_reg, names).plot()

# ── AnnuityBoxPlotter — one extra series ───────────────────────────
# AnnuityBoxPlotter(price_reg, names, extra_series=[
#     ExtraSeries(price_nat, "National", position="last"),
# ]).plot()

# ── AnnuityBoxPlotter — two extra series at different positions ─────
# AnnuityBoxPlotter(price_reg, names, extra_series=[
#     ExtraSeries(price_nat,   "National",  position="last"),
#     ExtraSeries(price_bench, "Benchmark", position="first"),
# ]).plot()

# ── AnnuityBoxPlotter — custom style on one series ──────────────────
# AnnuityBoxPlotter(price_reg, names, extra_series=[
#     ExtraSeries(price_nat, "National",
#                 style={"edgecolor": "#00e5ff", "median": "#00e5ff"}),
# ]).plot()
