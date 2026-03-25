"""
plot_class
==========

This module gathers all visualisation classes and utilities used for
demographic and actuarial analysis at the regional (NUTS) level.

It provides five main classes:

* :class:`RegionalCurvePlotter` — compares mortality curves across models
  and regions as a grid of subplots.
* :class:`MapPlotter` — generates NUTS-level choropleth maps from
  demographic indicators (life expectancy, model errors, etc.).
* :class:`RegionalStats` — computes descriptive statistics
  (mean, standard deviation, average rank) on a projected regional indicator.
* :class:`DispersionPlotter` — visualises the evolution of inter-regional
  dispersion (standard deviation and range) over the projection horizon.
* :class:`AnnuityBoxPlotter` — draws regional boxplots of annuity prices,
  with optional overlay of national or benchmark series.

The module also exposes the dataclass :class:`ExtraSeries` and two internal
helper functions for boxplot styling (:func:`_resolve_style`,
:func:`_boxplot_kwargs`).

**Dependencies**: ``geopandas``, ``matplotlib``, ``numpy``, ``pandas``,
``pathlib``, ``dataclasses``.

**NUTS shapefile**: ``SHAPEF_PATH`` is resolved automatically by walking up
the directory tree from the current working directory until a ``NUTS_files``
folder is found.

Quick-start example::

    # Compare mortality curves for the year 2040
    plotter = RegionalCurvePlotter(
        x_values=ages,
        curves_dict={"Model A": arr_a, "Model B": arr_b},
        tv=tv_vector,
        regions=region_list,
    )
    plotter.plot(year_to_plot=2040)
    plotter.save("output/curves_2040.png")

    # Choropleth map for 2030
    mp = MapPlotter(regions, data_3d, tv_future, "FR", "Life expectancy e0")
    mp.plot_single_year(year=2030)
    mp.save("output/map_2030.png")
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

# ── Global style : pure white background for all figures ────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
})


# ── Default color palette for extra series (cycles if more series than colors) ──
_DEFAULT_COLORS = [
    {"facecolor": "#1a1a2e", "edgecolor": "#e94560", "median": "#e94560"},  # red
    {"facecolor": "#fefefe", "edgecolor": "#e67e22", "median": "#e67e22"},  # orange
    {"facecolor": "#fefefe", "edgecolor": "#27ae60", "median": "#27ae60"},  # green
    {"facecolor": "#fefefe", "edgecolor": "#8e44ad", "median": "#8e44ad"},  # purple
]

# Default style applied to regional boxplots
_REGIONAL_STYLE = {
    "facecolor": "#fefefe",
    "edgecolor": "#2c3e50",
    "median":    "#f39c12",
}

# ── Resolve path to the NUTS shapefile ──────────────────────────────────────
BASE_DIR = Path.cwd()
while not (BASE_DIR / "NUTS_files").exists():
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "NUTS_files" / "NUTS_RG_01M_2024_3035.shp"
#: Path to the NUTS shapefile used for choropleth maps.
SHAPEF_PATH = DATA_DIR


# ════════════════════════════════════════════════════════════════════
# DATACLASS : ExtraSeries
# ════════════════════════════════════════════════════════════════════

@dataclass
class ExtraSeries:
    """
    Describes an additional data series to overlay on top of the regional
    boxplots in :class:`AnnuityBoxPlotter`.

    :param data: Numpy array of shape ``(nb_xe, 1, nb_simul)`` containing
        the simulations for this series.
    :type data: numpy.ndarray
    :param label: Display name used in the legend and x-axis tick labels.
    :type label: str
    :param position: Insertion position of the boxplot relative to regions.
        ``"first"`` places it before the regional boxes, ``"last"`` after.
    :type position: str
    :param style: Optional visual override dictionary. Valid keys:
        ``facecolor``, ``edgecolor``, ``median``, ``linewidth``.
        If empty or ``None``, a color from the default palette is applied.
    :type style: dict
    """
    data:     np.ndarray
    label:    str
    position: str  = "last"
    style:    dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
# INTERNAL HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def _resolve_style(user_style: dict, default: dict, linewidth: float = 2.2) -> dict:
    """
    Merge user-provided style overrides with a default style dictionary.

    Keys present in *user_style* override those in *default*.
    The ``linewidth`` key is added with value *linewidth* if it is not
    already present after merging.

    :param user_style: Overrides provided by the user (may be empty).
    :type user_style: dict
    :param default: Base style to use as a fallback.
    :type default: dict
    :param linewidth: Default line width if not specified elsewhere.
    :type linewidth: float
    :returns: Merged style dictionary ready for use.
    :rtype: dict
    """
    merged = {**default, **user_style}
    merged.setdefault("linewidth", linewidth)
    return merged


def _boxplot_kwargs(style: dict, alpha: float = 0.88) -> dict:
    """
    Build the keyword-argument dictionary to pass to ``ax.boxplot()``
    from a style dictionary.

    Sets the visual properties of the box, median line, whiskers, caps,
    and outlier markers.

    :param style: Style dictionary with keys ``facecolor``, ``edgecolor``,
        ``median``, and optionally ``linewidth``.
    :type style: dict
    :param alpha: Transparency applied to the box fill.
    :type alpha: float
    :returns: ``**kwargs`` dictionary ready to be unpacked into ``ax.boxplot()``.
    :rtype: dict
    """
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


# ════════════════════════════════════════════════════════════════════
# CLASS 1 : Regional curve comparison
# ════════════════════════════════════════════════════════════════════

class RegionalCurvePlotter:
    """
    Plots multiple model curves for each region as a grid of subplots,
    producing one figure per group of ``n_cols`` regions.

    Typically used to compare mortality rates or other demographic indicators
    from different models for a given projection year.

    :param x_values: Common x-axis values shared by all curves
        (e.g. age groups).
    :type x_values: array-like
    :param curves_dict: Dictionary ``{label: 3D_array}`` where each array
        has shape ``(nb_ages, nb_years, nb_regions)``.
    :type curves_dict: dict
    :param tv: 1D array of available projection years.
    :type tv: numpy.ndarray
    :param regions: List of region names.
    :type regions: list
    :param n_cols: Number of columns in the subplot grid (default: 3).
    :type n_cols: int
    :param yscale: Y-axis scale, ``"log"`` or ``"linear"`` (default: ``"log"``).
    :type yscale: str
    :param xlabel: X-axis label (default: ``"Age"``).
    :type xlabel: str
    :param ylabel: Y-axis label (default: ``"Mortality rate"``).
    :type ylabel: str
    :param title_prefix: Prefix string used in the figure title
        (default: ``"Comparison"``).
    :type title_prefix: str
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

    def _render_group(self, region_indices, t_index, year_to_plot, group_label):
        """
        Internal — renders one matplotlib figure for a subset of regions
        (one row of ``n_cols`` panels).

        :param region_indices: Indices of regions to display in this group,
            referring to positions in ``self.regions``.
        :type region_indices: list[int]
        :param t_index: Time index in the curve arrays corresponding to
            ``year_to_plot``.
        :type t_index: int
        :param year_to_plot: Year shown in the figure title.
        :type year_to_plot: int
        :param group_label: Group label, e.g. ``"1/2"`` or ``"2/2"``,
            appended to the overall title.
        :type group_label: str
        :returns: The generated matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        n      = len(region_indices)
        n_rows = int(np.ceil(n / self.n_cols))

        fig, axes = plt.subplots(
            n_rows, self.n_cols,
            figsize=(5 * self.n_cols, 4 * n_rows),
        )
        axes = np.array(axes).reshape(-1)   # always 1-D for uniform indexing

        for pos, g in enumerate(region_indices):
            ax = axes[pos]
            for label, array3d in self.curves_dict.items():
                ax.plot(
                    self.x_values,
                    array3d[:, t_index, g],
                    label=label,
                    linewidth=1.3,
                )

            ax.set_title(self.regions[g], fontsize=10, fontweight="bold")
            ax.set_xlabel(self.xlabel, fontsize=8)
            ax.set_ylabel(self.ylabel, fontsize=8)
            ax.set_yscale(self.yscale)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(fontsize=7, framealpha=0.85)

        # Remove unused subplot slots when regions don't fill the grid
        for i in range(n, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig

    def plot(self, year_to_plot):
        """
        Generate and display one figure per group of ``n_cols`` regions
        for the requested projection year.

        With ``n_cols=3`` (default) and 22 regions, this produces 8 figures
        of 3 panels each. Figures are also stored in ``self._figs`` for a
        subsequent call to :meth:`save`.

        :param year_to_plot: Year to display. Must exist in ``self.tv``.
        :type year_to_plot: int
        :raises ValueError: If ``year_to_plot`` is not found in ``self.tv``.
        """
        if year_to_plot not in self.tv:
            raise ValueError(f"Year {year_to_plot} is not available in tv.")

        t_index   = np.where(self.tv == year_to_plot)[0][0]
        n_regions = len(self.regions)
        n_figs    = int(np.ceil(n_regions / self.n_cols))

        self._figs = []
        for f in range(n_figs):
            start   = f * self.n_cols
            end     = min(start + self.n_cols, n_regions)
            indices = list(range(start, end))
            label   = f"{f + 1}/{n_figs}"
            fig     = self._render_group(indices, t_index, year_to_plot, label)
            self._figs.append(fig)

        self._fig  = self._figs[0]
        plt.show()

    def save(self, path, dpi=150, **kwargs):
        """
        Save all figures produced by :meth:`plot` to disk.

        *path* is used as a base: the suffix ``_part1``, ``_part2``, … is
        inserted before the file extension.

        Example: ``"output/curves_2040.png"`` produces
        ``"output/curves_2040_part1.png"`` and ``"output/curves_2040_part2.png"``.

        :param path: Base destination file path.
        :type path: str or pathlib.Path
        :param dpi: Resolution in dots per inch (default: 150).
        :type dpi: int
        :param kwargs: Additional arguments forwarded to ``matplotlib.savefig()``.
        :raises RuntimeError: If :meth:`plot` has not been called yet.
        """
        if not hasattr(self, "_figs"):
            raise RuntimeError("Call plot() before save().")
        p = Path(path)
        Path(p.parent).mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(self._figs, start=1):
            out = p.parent / f"{p.stem}_part{i}{p.suffix}"
            fig.savefig(out, dpi=dpi, bbox_inches="tight", **kwargs)
            print(f"✅ Saved → {out}")


# ════════════════════════════════════════════════════════════════════
# CLASS 2 : Choropleth map plotter
# ════════════════════════════════════════════════════════════════════

class MapPlotter:
    """
    Renders choropleth maps of a demographic indicator at the NUTS regional
    level from an Eurostat shapefile.

    Several display modes are available:

    * :meth:`plot_single_year` — one map for a given year.
    * :meth:`plot_compare_years` — side-by-side maps for multiple years
      sharing a common colour scale.
    * :meth:`plot_static` — map without a time dimension (aggregated data
      or mean errors).
    * :meth:`plot_compare_models` — side-by-side comparison of two models
      using a shared diverging colour scale.

    :param regions: NUTS region codes (e.g. ``["FR10", "FR21", ...]``).
    :type regions: array-like
    :param data: Data array of shape
        ``(nb_ages, horizon, nb_regions)`` or ``(horizon, nb_regions)``.
        If 3-D, the age slice *age* is extracted at initialisation.
    :type data: numpy.ndarray
    :param tv_future: 1D array of projection years.
    :type tv_future: numpy.ndarray
    :param country_code: ISO country code used to filter the shapefile
        (e.g. ``"FR"``).
    :type country_code: str
    :param indicator_name: Indicator label displayed in the colorbar and title.
    :type indicator_name: str
    :param shapef_path: Path to the NUTS shapefile (default: ``SHAPEF_PATH``).
    :type shapef_path: pathlib.Path
    :param cmap: Matplotlib colormap name (default: ``"Blues"``).
    :type cmap: str
    :param nuts_level: NUTS level to display (default: 2).
    :type nuts_level: int
    :param age: Age index to slice when *data* is 3-D (default: 0).
    :type age: int
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
        self.age            = age

        # Reduce to 2D (horizon, nb_regions) if a 3D array is provided
        self.data = data[age, :, :] if data.ndim == 3 else data

    def _load_shapefile(self):
        """
        Load the NUTS shapefile, filter to the target country and NUTS level,
        then reproject to Lambert-93 (EPSG:2154).

        :returns: Filtered and reprojected GeoDataFrame.
        :rtype: geopandas.GeoDataFrame
        """
        shapef = gpd.read_file(self.shapef_path)
        shp = shapef[
            (shapef["CNTR_CODE"] == self.country_code) &
            (shapef["LEVL_CODE"] == self.nuts_level)
        ].copy().to_crs(epsg=2154)
        return shp

    def _add_colorbar(self, fig, norm):
        """
        Append a horizontal colorbar in a dedicated axes at the bottom
        of the figure.

        :param fig: Matplotlib figure in which to insert the colorbar.
        :type fig: matplotlib.figure.Figure
        :param norm: Colour normalisation to associate with the colorbar.
        :type norm: matplotlib.colors.Normalize
        """
        cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=self.indicator_name)

    def plot_single_year(self, year=2023):
        """
        Draw a single choropleth map for the specified projection year.

        The figure is stored in ``self._fig`` for a subsequent call to
        :meth:`save`.

        :param year: Projection year to display (default: 2023).
        :type year: int
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
        self._fig = fig
        plt.show()

    def plot_compare_years(self, years=None):
        """
        Draw side-by-side choropleth maps for several years using a shared
        colour scale to allow direct comparison.

        The figure is stored in ``self._fig`` for a subsequent call to
        :meth:`save`.

        :param years: List of years to compare (default: ``[2023, 2050]``).
        :type years: list[int]
        """
        if years is None:
            years = [2023, 2050]

        shp = self._load_shapefile()

        # Build merged GeoDataFrames and collect values for each target year
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

        # Common normalisation range across all years for a fair comparison
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
        self._fig = fig
        plt.show()

    def plot_static(self, static_data, title_suffix=""):
        """
        Draw a choropleth map from data that has no time dimension
        (e.g. mean errors or aggregated indicators).

        Accepts shapes ``(nb_regions,)`` or ``(nb_ages, nb_regions)``.
        In the latter case, the age index defined at initialisation is used.

        :param static_data: Data to map, of shape
            ``(nb_regions,)`` or ``(nb_ages, nb_regions)``.
        :type static_data: numpy.ndarray
        :param title_suffix: Optional subtitle displayed on the map
            (e.g. ``"Mean error"``).
        :type title_suffix: str
        :raises ValueError: If *static_data* is neither 1-D nor 2-D.
        """
        shp = self._load_shapefile()

        # Select the age slice if the data array is 2-D
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
        self._fig = fig
        plt.show()

    def save(self, path, dpi=150, **kwargs):
        """
        Save the last rendered map to disk.

        :param path: Destination file path (e.g. ``"output/map_2030.png"``).
        :type path: str or pathlib.Path
        :param dpi: Resolution in dots per inch (default: 150).
        :type dpi: int
        :param kwargs: Additional arguments forwarded to ``matplotlib.savefig()``.
        :raises RuntimeError: If no plot method has been called yet.
        """
        if not hasattr(self, "_fig"):
            raise RuntimeError("Call a plot method before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"✅ Saved → {path}")

    def plot_compare_models(
        self,
        other_data,
        model_labels=("Model A", "Model B"),
        static=False,
        year=None,
        cmap_diverging="RdYlGn_r",
    ):
        """
        Display the values (errors or any indicator) of two models side by side
        on a map, with a shared diverging colour scale so differences are
        immediately visible.

        The normalisation is symmetric around zero for signed data (e.g. bias)
        and min/max for strictly positive data (e.g. RMSE, MAE).

        The figure is stored in ``self._fig`` for a subsequent call to
        :meth:`save`.

        :param other_data: Data for the second model, same shape as
            ``self.data`` (``(horizon, nb_regions)``) or ``(nb_regions,)``
            when *static=True*.
        :type other_data: numpy.ndarray
        :param model_labels: Titles for the left and right panels
            (default: ``("Model A", "Model B")``).
        :type model_labels: tuple[str, str]
        :param static: If ``True``, both datasets are treated as
            time-independent (default: ``False``).
        :type static: bool
        :param year: Projection year to display when ``static=False``.
            If ``None``, the first year in ``tv_future`` is used.
        :type year: int or None
        :param cmap_diverging: Colormap applied to both panels simultaneously.
            A diverging colormap (e.g. ``"RdYlGn_r"``, ``"coolwarm"``) is
            recommended for error maps (default: ``"RdYlGn_r"``).
        :type cmap_diverging: str

        .. note::
            The figure produced can be saved with :meth:`save`.

        Example::

            mp = MapPlotter(regions, errors_model_a, tv, "FR", "RMSE")
            mp.plot_compare_models(errors_model_b, model_labels=("LLP", "Lee-Carter"))
            mp.save("output/compare_models.png")
        """
        shp = self._load_shapefile()

        # ── Extract values for each model ────────────────────────────
        if static:
            vals_a = self.data[self.age, :] if self.data.ndim == 2 else self.data
            vals_b = other_data[self.age, :] if np.asarray(other_data).ndim == 2 else other_data
        else:
            if year is None:
                year = self.tv_future[0]
            idx    = np.where(self.tv_future == year)[0][0]
            vals_a = self.data[idx, :]
            vals_b = np.asarray(other_data)[idx, :]

        # ── Shared symmetric normalisation (best for error maps) ─────
        abs_max = max(np.nanmax(np.abs(vals_a)), np.nanmax(np.abs(vals_b)))
        vmin, vmax = -abs_max, abs_max
        # Fall back to min/max if all values are positive (e.g. RMSE, MAE)
        if np.nanmin([vals_a.min(), vals_b.min()]) >= 0:
            vmin = min(np.nanmin(vals_a), np.nanmin(vals_b))
            vmax = max(np.nanmax(vals_a), np.nanmax(vals_b))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # ── Build GeoDataFrames ──────────────────────────────────────
        def _merge(vals):
            return shp.merge(
                gpd.GeoDataFrame({"NUTS_ID": self.regions, "value": vals}),
                on="NUTS_ID", how="left",
            )

        df_a, df_b = _merge(vals_a), _merge(vals_b)

        # ── Plot both panels ─────────────────────────────────────────
        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 9),
            gridspec_kw={"bottom": 0.12},
        )

        for ax, df, label in zip(axes, [df_a, df_b], model_labels):
            df.plot(
                column="value", cmap=cmap_diverging, linewidth=0.5,
                edgecolor="black", ax=ax, norm=norm, legend=False,
            )
            ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
            ax.axis("off")

        # Shared colorbar for both panels
        cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
        sm = plt.cm.ScalarMappable(cmap=cmap_diverging, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=self.indicator_name)

        self._fig = fig
        plt.tight_layout()
        plt.show()


# ════════════════════════════════════════════════════════════════════
# CLASS 3 : Regional statistics
# ════════════════════════════════════════════════════════════════════

class RegionalStats:
    """
    Computes descriptive statistics on a projected regional indicator
    over the full projection horizon.

    Produces three tables: the complete ``(horizon × nb_regions)`` matrix,
    per-year aggregates, and per-region aggregates sorted by descending mean.

    :param ex: Data array of shape ``(nb_ages, horizon, nb_regions)``.
    :type ex: numpy.ndarray
    :param regions: Region codes or names.
    :type regions: array-like
    :param tv_future: 1D array of projection years.
    :type tv_future: numpy.ndarray
    :param age: Age index to extract (0 = e0, 65 = e65, …) (default: 0).
    :type age: int
    """

    def __init__(self, ex, regions, tv_future, age=0):
        self.ex        = ex
        self.regions   = regions
        self.tv_future = tv_future
        self.age       = age

        # Reduce 3D array to (horizon, nb_regions) for the target age
        self.data = ex[age, :, :]

    def compute(self):
        """
        Run all aggregations and return a dictionary of pandas DataFrames.

        Per-year statistics include mean, standard deviation, minimum,
        maximum, range, and coefficient of variation.
        Per-region statistics additionally include the average rank
        (rank 1 = highest value each year).

        :returns: Dictionary containing three DataFrames:

            * ``"full"``      — complete ``(horizon × nb_regions)`` matrix.
            * ``"by_year"``   — statistics aggregated per projection year.
            * ``"by_region"`` — statistics aggregated per region, sorted by
              descending mean.

        :rtype: dict[str, pandas.DataFrame]
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
    Visualises the evolution of inter-regional dispersion of an indicator
    over the full projection horizon.

    Two metrics are plotted side by side: the inter-regional standard
    deviation and the range (max − min), making it easy to assess whether
    regions are converging or diverging over time.

    :param ex: Data array of shape ``(nb_ages, horizon, nb_regions)``.
    :type ex: numpy.ndarray
    :param regions: Region codes or names (kept for consistency with other
        classes; not directly used in the plot).
    :type regions: array-like
    :param tv_future: 1D array of projection years.
    :type tv_future: numpy.ndarray
    :param indicator_name: Indicator label displayed in the figure title
        (default: ``"Life expectancy"``).
    :type indicator_name: str
    :param ages: List of age indices to compare (default: ``[0, 65]``).
    :type ages: list[int]
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
        Generate and display two side-by-side panels:

        * **Left panel**  — inter-regional standard deviation over time.
        * **Right panel** — inter-regional range (max − min) over time.

        One curve per age index in ``self.ages`` is drawn in each panel.
        The figure is stored in ``self._fig``.
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
        self._fig = fig
        plt.show()

    def save(self, path, dpi=150, **kwargs):
        """
        Save the last rendered dispersion figure to disk.

        :param path: Destination file path.
        :type path: str or pathlib.Path
        :param dpi: Resolution in dots per inch (default: 150).
        :type dpi: int
        :param kwargs: Additional arguments forwarded to ``matplotlib.savefig()``.
        :raises RuntimeError: If :meth:`plot` has not been called yet.
        """
        if not hasattr(self, "_fig"):
            raise RuntimeError("Call plot() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"✅ Saved → {path}")


# ════════════════════════════════════════════════════════════════════
# CLASS 5 : Annuity price boxplots
# ════════════════════════════════════════════════════════════════════

class AnnuityBoxPlotter:
    """
    Draws regional boxplots of annuity prices, with optional overlay of
    extra series (national price, benchmark, etc.) placed before or after
    the regional boxes.

    Extra series are defined via :class:`ExtraSeries` instances and can be
    positioned before (``"first"``) or after (``"last"``) the regional block.
    Thin vertical separators visually delimit the groups.

    :param price_regional: Regional price array of shape
        ``(nb_xe, nb_reg, nb_simul)``.
    :type price_regional: numpy.ndarray
    :param region_names: List of region label strings.
    :type region_names: list[str]
    :param extra_series: List of :class:`ExtraSeries` objects to overlay.
        Pass ``[]`` or ``None`` to display regional data only.
    :type extra_series: list[ExtraSeries] or None
    :param xe_idx: Index along the age dimension of *price_regional*
        (default: 0).
    :type xe_idx: int
    :param duration: Annuity duration in years, used in the chart title
        (default: 20).
    :type duration: int
    :param age: Policyholder age, used in the chart title (default: 60).
    :type age: int
    :param figsize: Figure size in inches (default: ``(18, 7)``).
    :type figsize: tuple
    :param show: If ``True``, calls ``plt.show()`` after rendering
        (default: ``True``).
    :type show: bool
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
        Render and display the boxplot chart.

        Display order: ``"first"`` extra series → regional boxes →
        ``"last"`` extra series. Thin vertical separators delimit the blocks
        when extra series are present.

        The figure is stored in ``self._fig`` for a subsequent call to
        :meth:`save`.

        :param ax: Existing matplotlib axes to draw into.
            If ``None``, a new figure is created.
        :type ax: matplotlib.axes.Axes or None
        :returns: The figure and axes used.
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        nb_reg = self.price_regional.shape[1]

        # Assign a color from the default palette to each extra series (cyclic)
        for k, es in enumerate(self.extra_series):
            default_col = _DEFAULT_COLORS[k % len(_DEFAULT_COLORS)]
            es.style = _resolve_style(es.style, default_col)

        # Build the ordered series list: "first" extras → regional boxes → "last" extras
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

        # Thin vertical separators between "first", regional, and "last" blocks
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

        # Build the legend from colored patch handles
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

        # Display the figure (IPython / Jupyter-aware)
        try:
            from IPython.display import display as ipy_display
            ipy_display(fig)
        except ImportError:
            if self.show:
                plt.show()

        self._fig = fig
        plt.close(fig)
        return fig, ax

    def save(self, path, dpi=150, **kwargs):
        """
        Save the last rendered boxplot figure to disk.

        :param path: Destination file path.
        :type path: str or pathlib.Path
        :param dpi: Resolution in dots per inch (default: 150).
        :type dpi: int
        :param kwargs: Additional arguments forwarded to ``matplotlib.savefig()``.
        :raises RuntimeError: If :meth:`plot` has not been called yet.
        """
        if not hasattr(self, "_fig"):
            raise RuntimeError("Call plot() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"✅ Saved → {path}")


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
# plotter.save("output/curves_2040.png")

# ── MapPlotter — single year ────────────────────────────────────────
# mp = MapPlotter(regions, data_3d, tv_future, "FR", "Life expectancy e0")
# mp.plot_single_year(year=2030)
# mp.save("output/map_2030.png")

# ── MapPlotter — compare years ──────────────────────────────────────
# mp.plot_compare_years(years=[2023, 2050])
# mp.save("output/map_compare_years.png")

# ── MapPlotter — compare two models side by side ────────────────────
# mp = MapPlotter(regions, errors_model_a, tv_future, "FR", "RMSE")
# mp.plot_compare_models(
#     other_data=errors_model_b,
#     model_labels=("LLP", "Lee-Carter"),
#     year=2040,
#     cmap_diverging="RdYlGn_r",
# )
# mp.save("output/compare_llp_vs_lc.png")
#
# For static error arrays (no time dimension):
# mp.plot_compare_models(errors_b, model_labels=("LLP", "LC"), static=True)
# mp.save("output/compare_static.png")

# ── RegionalStats ───────────────────────────────────────────────────
# stats = RegionalStats(ex, regions, tv_future, age=0)
# results = stats.compute()
# print(results["by_year"])

# ── DispersionPlotter ───────────────────────────────────────────────
# dp = DispersionPlotter(ex, regions, tv_future, indicator_name="Life expectancy")
# dp.plot()
# dp.save("output/dispersion.png")

# ── AnnuityBoxPlotter — no extra series ────────────────────────────
# bp = AnnuityBoxPlotter(price_reg, names)
# bp.plot()
# bp.save("output/annuity_boxes.png")

# ── AnnuityBoxPlotter — one extra series ───────────────────────────
# bp = AnnuityBoxPlotter(price_reg, names, extra_series=[
#     ExtraSeries(price_nat, "National", position="last"),
# ])
# bp.plot()
# bp.save("output/annuity_with_national.png")

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