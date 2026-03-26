"""
eurostat.py - Eurostat Data Manager
=====================================

This module provides a unified interface for fetching, caching, and processing
demographic data from the `Eurostat REST API <https://ec.europa.eu/eurostat/>`_,
coupled with NUTS geographic shapefiles for spatial operations.

**Main features:**

- Automatic loading of the NUTS shapefile (no manual handling required).
- Centralized configuration via :class:`EurostatConfig` (paths, datasets, excluded regions).
- Smart caching: data is downloaded once and stored as CSV files.
- Dynamic dataset registration: add new Eurostat dataset codes at runtime
  via :meth:`EurostatConfig.register_dataset` or
  :meth:`Eurostat_data.register_dataset`.
- Built-in data cleaning (age labels, year conversion).
- Backward-compatible wrapper functions for legacy code.

**Datasets available by default:**

=============  =====================  =================================
Key            Eurostat code          Description
=============  =====================  =================================
mortality      ``demo_r_mlife``       Life expectancy by NUTS region
deaths         ``demo_r_magec``       Deaths by age, sex and region
population     ``demo_r_d2jan``       Population on 1 January by region
=============  =====================  =================================

**Quick start example:**

.. code-block:: python

    from eurostat import Eurostat_data, EurostatConfig

    # Instantiate the manager (shapefile auto-loaded)
    manager = Eurostat_data()

    # Load French mortality data (downloaded then cached as CSV)
    mortality = manager.load("mortality", "FR")

    # Register a new dataset type at runtime
    manager.register_dataset("fertility", "demo_r_find3", filename="Fxt_raw")

    # Load the newly registered dataset
    fertility = manager.load("fertility", "DE")

**Dependencies:**

- ``pandas``
- ``geopandas``
- ``numpy``
- ``eurostatapiclient``

.. note::
    The NUTS shapefile must be present on disk. Configure its path via
    ``EurostatConfig.set_default_shapefile()`` or pass ``shapefile_path``
    to :class:`Eurostat_data`.
"""

import pandas as pd
from eurostatapiclient import EurostatAPIClient
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Literal, Union

# ---------------------------------------------------------------------------
# Base directory resolution
# ---------------------------------------------------------------------------
# Walk up from the current working directory until the NUTS_files folder is found.
BASE_DIR = Path.cwd()
while not (BASE_DIR / "NUTS_files").exists():
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "NUTS_files" / "NUTS_RG_01M_2024_3035.shp"


# =============================================================================
# EurostatConfig
# =============================================================================

class EurostatConfig:
    """Centralized configuration for the Eurostat manager.

    This class holds global defaults (paths, dataset codes, excluded regions)
    shared across all :class:`Eurostat_data` instances.  All attributes are
    class-level so they can be changed once and immediately reflected everywhere.

    Attributes
    ----------
    DEFAULT_SHAPEFILE_PATH : Path
        Default path to the NUTS shapefile used when no explicit path is given.
    DEFAULT_DATA_PATH : Path
        Default directory where downloaded CSV files are cached.
    DATASETS : dict[str, str]
        Mapping of human-readable dataset keys to Eurostat dataset codes.
        Use :meth:`register_dataset` to add new entries at runtime.
    EXCLUDE_REGIONS : dict[str, list[str]]
        Mapping of ISO country codes to lists of NUTS region IDs that should
        be excluded from queries (e.g. overseas territories).

    Examples
    --------
    Change the default shapefile path globally before creating any manager:

    .. code-block:: python

        EurostatConfig.set_default_shapefile("D:/data/NUTS_2024.shp")
        manager = Eurostat_data()  # will use the new path

    Register a new dataset so it becomes available to all managers:

    .. code-block:: python

        EurostatConfig.register_dataset(
            "fertility", "demo_r_find3", filename="Fxt_raw"
        )
    """

    #: Default path to the NUTS shapefile (resolved at import time).
    DEFAULT_SHAPEFILE_PATH = DATA_DIR

    #: Default directory for cached CSV data files.
    DEFAULT_DATA_PATH = Path("../data")

    #: Built-in dataset registry: ``{key: eurostat_code}``.
    DATASETS: Dict[str, str] = {
        "mortality": "demo_r_mlife",
        "deaths": "demo_r_magec",
        "population": "demo_r_d2jan",
    }

    #: Internal mapping of dataset keys to their cache filename stems.
    #: Used by :meth:`Eurostat_data.load` and :meth:`Eurostat_data.cache_info`.
    _FILENAME_MAP: Dict[str, str] = {
        "mortality": "mxt_raw",
        "deaths": "Dxt_raw",
        "population": "Lxt_raw",
    }

    #: Regions excluded by country (e.g. overseas territories).
    EXCLUDE_REGIONS: Dict[str, List[str]] = {
        "FR": ["FRY1", "FRY2", "FRY3", "FRY4", "FRY5"],  # DOM-TOM
        "PT": ["PT20", "PT30"],                            # Azores, Madeira
        "ES": ["ES63", "ES64", "ES70"],                    # Ceuta, Melilla, Canary Islands
        "NO": ["NO0B"],                                    # Svalbard
    }

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @classmethod
    def set_default_shapefile(cls, path: Union[str, Path]) -> None:
        """Change the default NUTS shapefile path globally.

        After calling this method, any new :class:`Eurostat_data` instance
        created without an explicit ``shapefile_path`` will use *path*.

        Parameters
        ----------
        path : str or Path
            Absolute or relative path to the NUTS ``.shp`` file.

        Examples
        --------
        .. code-block:: python

            EurostatConfig.set_default_shapefile("D:/data/NUTS_2024.shp")
        """
        cls.DEFAULT_SHAPEFILE_PATH = Path(path)

    @classmethod
    def set_default_data_path(cls, path: Union[str, Path]) -> None:
        """Change the default data cache directory globally.

        Parameters
        ----------
        path : str or Path
            Directory where downloaded CSV files will be stored.

        Examples
        --------
        .. code-block:: python

            EurostatConfig.set_default_data_path("D:/eurostat_cache")
        """
        cls.DEFAULT_DATA_PATH = Path(path)

    @classmethod
    def register_dataset(
        cls,
        key: str,
        eurostat_code: str,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a new Eurostat dataset type in the global configuration.

        Once registered, the dataset key becomes available in
        :meth:`Eurostat_data.load`, :meth:`Eurostat_data.cache_info`, and
        :meth:`Eurostat_data.clear_cache` for **all** manager instances.

        Parameters
        ----------
        key : str
            Short human-readable identifier for the dataset
            (e.g. ``"fertility"``).  Must be unique unless *overwrite* is
            ``True``.
        eurostat_code : str
            Official Eurostat dataset code as found in the Eurostat data
            browser (e.g. ``"demo_r_find3"``).
        filename : str, optional
            Stem used for the cached CSV file, i.e. the file will be saved
            as ``{country}_{filename}.csv``.  Defaults to ``"{key}_raw"``
            when not provided.
        overwrite : bool, optional
            If ``False`` (default) and *key* already exists, a
            :exc:`ValueError` is raised.  Set to ``True`` to silently
            replace the existing entry.

        Raises
        ------
        ValueError
            If *key* is already registered and *overwrite* is ``False``.

        Examples
        --------
        Register a fertility dataset, then load it for Germany:

        .. code-block:: python

            EurostatConfig.register_dataset(
                "fertility", "demo_r_find3", filename="Fxt_raw"
            )
            manager = Eurostat_data()
            fertility = manager.load("fertility", "DE")

        Overwrite an existing entry:

        .. code-block:: python

            EurostatConfig.register_dataset(
                "mortality", "demo_r_mlife2", overwrite=True
            )
        """
        if key in cls.DATASETS and not overwrite:
            raise ValueError(
                f"Dataset key '{key}' is already registered "
                f"(code: {cls.DATASETS[key]}). "
                f"Use overwrite=True to replace it."
            )

        cls.DATASETS[key] = eurostat_code
        cls._FILENAME_MAP[key] = filename if filename else f"{key}_raw"
        print(f"✅ [Config] Dataset registered: '{key}' → '{eurostat_code}' "
              f"(cache filename: '{cls._FILENAME_MAP[key]}')")


# =============================================================================
# Eurostat_data
# =============================================================================

class Eurostat_data:
    """Unified manager for all Eurostat scraping operations.

    This class is the main entry point.  It handles:

    - Automatic loading of the NUTS shapefile on first use (lazy loading).
    - Fetching data from the Eurostat API and caching results as CSV files.
    - Filtering NUTS regions by country, level, and exclusion lists.
    - Building age × year pivot tables for actuarial / demographic analysis.
    - Dynamic registration of new dataset types at the instance level.

    Parameters
    ----------
    shapefile : GeoDataFrame, optional
        An already-loaded NUTS shapefile.  Takes priority over
        *shapefile_path* when provided.
    shapefile_path : str or Path, optional
        Path to the NUTS ``.shp`` file.  Falls back to
        :attr:`EurostatConfig.DEFAULT_SHAPEFILE_PATH` when ``None``.
    data_path : str or Path, optional
        Directory for cached CSV files.  Falls back to
        :attr:`EurostatConfig.DEFAULT_DATA_PATH` when ``None``.
    language : str, optional
        Language code for Eurostat metadata labels (default ``"en"``).
    nuts_level : int, optional
        Default NUTS level used when not specified in method calls
        (default ``2``).
    auto_load_shapefile : bool, optional
        When ``True`` (default), the shapefile is loaded automatically the
        first time it is needed.  Set to ``False`` to skip shapefile loading
        entirely (useful when only non-spatial operations are needed).

    Examples
    --------
    Standard usage — shapefile loaded automatically:

    .. code-block:: python

        manager = Eurostat_data()
        mortality = manager.load("mortality", "FR")

    Provide a custom shapefile path:

    .. code-block:: python

        manager = Eurostat_data(shapefile_path="D:/custom/nuts.shp")

    Pass an already-loaded GeoDataFrame:

    .. code-block:: python

        import geopandas as gpd
        custom_shapef = gpd.read_file("path/to/shapefile.shp")
        manager = Eurostat_data(shapefile=custom_shapef)

    Register a new dataset type and load it:

    .. code-block:: python

        manager = Eurostat_data()
        manager.register_dataset("fertility", "demo_r_find3", filename="Fxt_raw")
        fertility = manager.load("fertility", "DE")
    """

    def __init__(
        self,
        shapefile: Optional[gpd.GeoDataFrame] = None,
        shapefile_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        language: str = "en",
        nuts_level: int = 2,
        auto_load_shapefile: bool = True,
    ):
        self._shapefile = shapefile
        self._shapefile_path = (
            Path(shapefile_path) if shapefile_path
            else EurostatConfig.DEFAULT_SHAPEFILE_PATH
        )
        self._auto_load = auto_load_shapefile
        self._shapefile_loaded = shapefile is not None

        self.data_path = Path(data_path) if data_path else EurostatConfig.DEFAULT_DATA_PATH
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.nuts_level = nuts_level

        # Eurostat REST API client
        self.client = EurostatAPIClient("1.0", "json", language)

        # In-memory cache for NUTS region lists (avoids repeated shapefile queries)
        self._regions_cache: Dict[Tuple[str, int, bool], List[str]] = {}

        print(
            "\n"
            "⚠️  [Eurostat] DATA QUALITY WARNING\n"
            "   Eurostat datasets may contain anomalies such as missing values,\n"
            "   suppressed cells, inconsistent time series, or regional boundary\n"
            "   changes. It is strongly recommended to perform thorough data\n"
            "   quality checks before proceeding with any analysis.\n"
        )

    # ------------------------------------------------------------------
    # Shapefile property and helpers
    # ------------------------------------------------------------------

    @property
    def shapefile(self) -> gpd.GeoDataFrame:
        """The NUTS shapefile as a GeoDataFrame, loaded lazily on first access.

        You never need to call this property directly; it is used internally
        by :meth:`get_regions` and :meth:`filter_shapefile`.

        Returns
        -------
        GeoDataFrame
            The loaded NUTS shapefile.

        Raises
        ------
        ValueError
            If the shapefile could not be loaded (e.g. file not found and
            auto-load is disabled).
        """
        if self._shapefile is None and self._auto_load:
            self._load_shapefile()

        if self._shapefile is None:
            raise ValueError(
                f"Shapefile not loaded. Path used: {self._shapefile_path}\n"
                f"Check that the file exists or set another path:\n"
                f"  EurostatConfig.set_default_shapefile('path/to/shapefile.shp')\n"
                f"  or manager = Eurostat_data(shapefile_path='path/to/shapefile.shp')"
            )

        return self._shapefile

    def _load_shapefile(self) -> None:
        """Load the NUTS shapefile from the configured path into memory.

        This private method is called automatically the first time
        :attr:`shapefile` is accessed.  It prints a warning if the file
        does not exist, rather than raising an exception, so that
        non-spatial operations can still proceed.
        """
        if not self._shapefile_path.exists():
            print(f"⚠️  Shapefile not found: {self._shapefile_path}")
            print(f"💡 Set the correct path with:")
            print(f"   EurostatConfig.set_default_shapefile('your/path.shp')")
            return

        print(f"📂 Loading NUTS shapefile... {self._shapefile_path.name}")
        self._shapefile = gpd.read_file(self._shapefile_path)
        self._shapefile_loaded = True
        print(f"✅ Shapefile loaded ({len(self._shapefile)} features)")

    def set_shapefile(
        self,
        shapefile: Optional[gpd.GeoDataFrame] = None,
        shapefile_path: Optional[Union[str, Path]] = None,
    ) -> "Eurostat_data":
        """Replace the current shapefile with a new one (chainable).

        Parameters
        ----------
        shapefile : GeoDataFrame, optional
            An already-loaded GeoDataFrame to use directly.
        shapefile_path : str or Path, optional
            Path to a new shapefile to load from disk.

        Returns
        -------
        Eurostat_data
            The same instance (allows method chaining).

        Examples
        --------
        .. code-block:: python

            manager.set_shapefile(shapefile_path="D:/new.shp").load("mortality", "FR")
        """
        if shapefile is not None:
            self._shapefile = shapefile
            self._shapefile_loaded = True
        elif shapefile_path is not None:
            self._shapefile_path = Path(shapefile_path)
            self._shapefile = None
            self._shapefile_loaded = False
            if self._auto_load:
                self._load_shapefile()

        # Invalidate the region cache because the shapefile changed
        self._regions_cache.clear()
        return self

    # ------------------------------------------------------------------
    # Dataset registration (instance level)
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        key: str,
        eurostat_code: str,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> "Eurostat_data":
        """Register a new Eurostat dataset type for this manager instance.

        This is a convenience wrapper around :meth:`EurostatConfig.register_dataset`
        that additionally returns ``self`` to allow method chaining.

        The registered dataset is immediately available via :meth:`load`,
        :meth:`cache_info`, and :meth:`clear_cache`.

        Parameters
        ----------
        key : str
            Short human-readable identifier (e.g. ``"fertility"``).
        eurostat_code : str
            Official Eurostat dataset code (e.g. ``"demo_r_find3"``).
        filename : str, optional
            Stem for the cache CSV file (defaults to ``"{key}_raw"``).
        overwrite : bool, optional
            Allow replacing an existing dataset entry (default ``False``).

        Returns
        -------
        Eurostat_data
            The same instance (allows method chaining).

        Raises
        ------
        ValueError
            If *key* already exists and *overwrite* is ``False``.

        Examples
        --------
        Register and immediately load in one chain:

        .. code-block:: python

            fertility = (
                Eurostat_data()
                .register_dataset("fertility", "demo_r_find3", filename="Fxt_raw")
                .load("fertility", "DE")
            )

        Register multiple datasets at once:

        .. code-block:: python

            manager = Eurostat_data()
            manager.register_dataset("fertility", "demo_r_find3")
            manager.register_dataset("marriages", "demo_r_marriages")
            print(manager.list_datasets())
        """
        EurostatConfig.register_dataset(key, eurostat_code, filename, overwrite)
        return self

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def load(
        self,
        dataset_type: str,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        download: bool = False,
        exclude_outremer: bool = True,
    ) -> pd.DataFrame:
        """Load an Eurostat dataset for a given country, with caching.

        On the first call, data is downloaded from the Eurostat API and saved
        to a local CSV file.  Subsequent calls with the same arguments read
        from the CSV unless *download* is ``True``.

        Parameters
        ----------
        dataset_type : str
            Key of the dataset to load.  Must be a key present in
            :attr:`EurostatConfig.DATASETS` (e.g. ``"mortality"``,
            ``"deaths"``, ``"population"``, or any key added via
            :meth:`register_dataset`).
        country : str, optional
            ISO 3166-1 alpha-2 country code (default ``"FR"``).
        nuts_level : int, optional
            NUTS hierarchical level (1, 2 or 3).  Uses the instance default
            when ``None``.
        download : bool, optional
            Force a fresh download even if a cached file already exists
            (default ``False``).
        exclude_outremer : bool, optional
            Exclude non-continental regions listed in
            :attr:`EurostatConfig.EXCLUDE_REGIONS` (default ``True``).

        Returns
        -------
        DataFrame
            Cleaned DataFrame ready for analysis.

        Raises
        ------
        KeyError
            If *dataset_type* is not found in :attr:`EurostatConfig.DATASETS`.

        Examples
        --------
        .. code-block:: python

            manager = Eurostat_data()
            mortality = manager.load("mortality", "FR")
            deaths    = manager.load("deaths", "BE", download=True)
        """
        nuts_level = nuts_level or self.nuts_level

        if dataset_type not in EurostatConfig.DATASETS:
            raise KeyError(
                f"Unknown dataset type: '{dataset_type}'. "
                f"Available: {list(EurostatConfig.DATASETS.keys())}. "
                f"Use register_dataset() to add new types."
            )

        # Retrieve region codes (results are cached in _regions_cache)
        regions = self.get_regions(country, nuts_level, exclude_outremer)

        dataset_code = EurostatConfig.DATASETS[dataset_type]
        filename = EurostatConfig._FILENAME_MAP[dataset_type]

        return self._load_and_cache(
            dataset_code=dataset_code,
            regions=regions,
            filename=filename,
            country=country,
            download=download,
        )

    def load_all(
        self,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Load all registered datasets for a given country.

        Iterates over every key in :attr:`EurostatConfig.DATASETS` and calls
        :meth:`load` for each one.

        Parameters
        ----------
        country : str, optional
            ISO 3166-1 alpha-2 country code (default ``"FR"``).
        nuts_level : int, optional
            NUTS level (uses instance default when ``None``).
        download : bool, optional
            Force fresh download for every dataset (default ``False``).

        Returns
        -------
        dict[str, DataFrame]
            Dictionary mapping each dataset key to its cleaned DataFrame.

        Examples
        --------
        .. code-block:: python

            manager = Eurostat_data()
            data = manager.load_all("FR")
            mortality  = data["mortality"]
            deaths     = data["deaths"]
            population = data["population"]
        """
        return {
            dataset: self.load(dataset, country, nuts_level, download)
            for dataset in EurostatConfig.DATASETS.keys()
        }

    def pivot_age_year(
        self,
        data: pd.DataFrame,
        region: str,
        gender: str,
        indicator: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Build an age × year cross-tabulation for a specific region.

        Filters *data* to the requested *region*, *gender*, and *indicator*,
        then produces a pivot table with ages as rows and years as columns.

        Parameters
        ----------
        data : DataFrame
            Source data returned by :meth:`load`.
        region : str
            NUTS region code (e.g. ``"FR10"``).
        gender : str
            Gender code as used by Eurostat: ``"M"`` (male), ``"F"`` (female),
            or ``"T"`` (total).
        indicator : str
            Eurostat indicator code (e.g. ``"LIFE_EXP"``).

        Returns
        -------
        pivot : DataFrame
            Age × year pivot table with numeric values.
        ages : ndarray
            Sorted integer array of age indices.
        years : ndarray
            Sorted integer array of year columns.

        Examples
        --------
        .. code-block:: python

            mortality = manager.load("mortality", "FR")
            pivot, ages, years = manager.pivot_age_year(
                mortality, "FR10", "M", "LIFE_EXP"
            )
        """
        subset = data[
            (data["geo"] == region)
            & (data["sex"] == gender)
            & (data["indic_de"] == indicator)
        ].reset_index(drop=True)

        pivot = pd.pivot_table(
            subset,
            values="values",
            index=["age"],
            columns=["time"],
            aggfunc="sum",
            fill_value=1e-6,
            observed=True,
        )

        pivot = pivot.sort_values("age").reset_index(drop=True)
        ages = pivot.index.values.astype("int")
        years = pivot.columns.values.astype("int")

        return pivot, ages, years

    def get_regions(
        self,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        exclude_outremer: bool = True,
    ) -> List[str]:
        """Return the list of NUTS region codes for a country (cached).

        Results are cached in ``_regions_cache`` so repeated calls with the
        same arguments do not re-query the shapefile.

        Parameters
        ----------
        country : str, optional
            ISO 3166-1 alpha-2 country code (default ``"FR"``).
        nuts_level : int, optional
            NUTS level (uses instance default when ``None``).
        exclude_outremer : bool, optional
            Exclude non-continental regions (default ``True``).

        Returns
        -------
        list[str]
            Sorted list of NUTS_ID codes.

        Examples
        --------
        .. code-block:: python

            regions_fr = manager.get_regions("FR", nuts_level=2)
            regions_be = manager.get_regions("BE")
        """
        nuts_level = nuts_level or self.nuts_level
        cache_key = (country, nuts_level, exclude_outremer)

        if cache_key not in self._regions_cache:
            filtered = self.filter_shapefile(country, nuts_level, exclude_outremer)
            self._regions_cache[cache_key] = filtered["NUTS_ID"].tolist()

        return self._regions_cache[cache_key]

    def filter_shapefile(
        self,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        exclude_outremer: bool = True,
    ) -> gpd.GeoDataFrame:
        """Filter the NUTS shapefile by country, NUTS level, and exclusions.

        Parameters
        ----------
        country : str, optional
            ISO 3166-1 alpha-2 country code (default ``"FR"``).
        nuts_level : int, optional
            NUTS level (uses instance default when ``None``).
        exclude_outremer : bool, optional
            If ``True``, removes regions listed in
            :attr:`EurostatConfig.EXCLUDE_REGIONS` for *country*
            (default ``True``).

        Returns
        -------
        GeoDataFrame
            Filtered and sorted GeoDataFrame with one row per NUTS region.
        """
        nuts_level = nuts_level or self.nuts_level

        filtered = self.shapefile[
            (self.shapefile["CNTR_CODE"] == country)
            & (self.shapefile["LEVL_CODE"] == nuts_level)
        ].copy()

        if exclude_outremer and country in EurostatConfig.EXCLUDE_REGIONS:
            filtered = filtered[
                ~filtered["NUTS_ID"].isin(EurostatConfig.EXCLUDE_REGIONS[country])
            ]

        return filtered.sort_values("NUTS_ID").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def parse_age(label) -> Optional[int]:
        """Parse an Eurostat age label string into an integer age.

        Eurostat uses string codes such as ``"Y25"`` or ``"Y_LT1"`` instead
        of plain integers.  This method converts them to numeric values
        suitable for analysis.

        Parameters
        ----------
        label : str or any
            Eurostat age label.

        Returns
        -------
        int or None
            Numeric age, or ``None`` if the label could not be parsed.

        Examples
        --------
        .. code-block:: python

            Eurostat_data.parse_age("Y25")    # → 25
            Eurostat_data.parse_age("Y_LT1")  # → 0
            Eurostat_data.parse_age("Y_GE85") # → 85
            Eurostat_data.parse_age("Y_GE95") # → 95
            Eurostat_data.parse_age("TOTAL")  # → None
        """
        if isinstance(label, str):
            if label.startswith("Y") and label[1:].isdigit():
                return int(label[1:])
            elif label == "Y_LT1":
                return 0
            elif label == "Y_GE85":
                return 85
            elif label == "Y_GE95":
                return 95
        return None

    def add_exclusion(self, country: str, regions: List[str]) -> "Eurostat_data":
        """Add NUTS region codes to the exclusion list for a country (chainable).

        Modifies :attr:`EurostatConfig.EXCLUDE_REGIONS` globally and clears
        the region cache so the change is reflected immediately.

        Parameters
        ----------
        country : str
            ISO 3166-1 alpha-2 country code.
        regions : list[str]
            NUTS_ID codes to exclude for *country*.

        Returns
        -------
        Eurostat_data
            The same instance (allows method chaining).

        Examples
        --------
        .. code-block:: python

            manager.add_exclusion("IT", ["ITG1", "ITG2"]).load("mortality", "IT")
        """
        if country not in EurostatConfig.EXCLUDE_REGIONS:
            EurostatConfig.EXCLUDE_REGIONS[country] = []
        EurostatConfig.EXCLUDE_REGIONS[country].extend(regions)
        self._regions_cache.clear()
        return self

    def list_datasets(self) -> List[str]:
        """Return the list of currently registered dataset keys.

        Returns
        -------
        list[str]
            All keys present in :attr:`EurostatConfig.DATASETS`.

        Examples
        --------
        .. code-block:: python

            manager.list_datasets()
            # → ['mortality', 'deaths', 'population']
        """
        return list(EurostatConfig.DATASETS.keys())

    def cache_info(
        self, country: Optional[str] = None
    ) -> Union[Dict[str, bool], Dict[str, Dict[str, bool]]]:
        """Check which datasets are available in the local cache.

        Parameters
        ----------
        country : str, optional
            ISO country code.  When provided, returns the cache status for
            that country only.  When ``None``, auto-detects all cached
            countries and returns a nested dict.

        Returns
        -------
        dict
            If *country* is given: ``{dataset_key: bool}`` mapping.

            If *country* is ``None``: ``{country: {dataset_key: bool}}``
            nested mapping.

        Examples
        --------
        .. code-block:: python

            manager.cache_info("FR")
            # → {'mortality': True, 'deaths': False, 'population': True}

            manager.cache_info()
            # → {'FR': {'mortality': True, ...}, 'BE': {...}}
        """
        filename_map = EurostatConfig._FILENAME_MAP

        if country:
            return {
                dataset: (self.data_path / f"{country}_{filename}.csv").exists()
                for dataset, filename in filename_map.items()
            }
        else:
            # Auto-detect countries from cached filenames
            countries: set = set()
            for file in self.data_path.glob("*_*.csv"):
                country_code = file.stem.split("_")[0]
                if len(country_code) == 2:  # ISO 2-letter code heuristic
                    countries.add(country_code)

            return {c: self.cache_info(c) for c in sorted(countries)}

    def clear_cache(
        self,
        country: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> int:
        """Delete cached CSV files from the data directory.

        Parameters
        ----------
        country : str, optional
            ISO country code.  When ``None``, all countries are targeted.
        dataset : str, optional
            Dataset key (e.g. ``"mortality"``).  When ``None``, all datasets
            for the selected country are deleted.

        Returns
        -------
        int
            Number of files deleted.

        Examples
        --------
        .. code-block:: python

            manager.clear_cache("FR")              # Delete all FR data
            manager.clear_cache("FR", "mortality") # Delete FR mortality only
            manager.clear_cache()                  # Wipe entire cache
        """
        count = 0

        if country and dataset:
            # Targeted single-file deletion
            filepath = self.data_path / f"{country}_{EurostatConfig._FILENAME_MAP[dataset]}.csv"
            if filepath.exists():
                filepath.unlink()
                count = 1
                print(f"🗑️  [Cache] Deleted: {filepath.name}")
        elif country:
            # Delete all cached files for one country
            for file in self.data_path.glob(f"{country}_*.csv"):
                file.unlink()
                count += 1
            print(f"🗑️  [Cache] Deleted: {count} files for {country}")
        else:
            # Full cache wipe
            for file in self.data_path.glob("*_*.csv"):
                file.unlink()
                count += 1
            if count > 0:
                print(f"🗑️  [Cache] Deleted: {count} files")

        return count

    def stats(self) -> Dict[str, object]:
        """Return statistics about the current manager state.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``shapefile_loaded`` (bool): whether the shapefile is in memory.
            - ``shapefile_path`` (str): configured shapefile path.
            - ``cached_files`` (int): number of CSV files in the cache.
            - ``cache_size_mb`` (float): total cache size in megabytes.
            - ``cached_countries`` (int): number of distinct countries cached.
            - ``regions_cache_entries`` (int): number of entries in the
              in-memory region cache.
            - ``data_path`` (str): path to the cache directory.

        Examples
        --------
        .. code-block:: python

            print(manager.stats())
        """
        cache_files = list(self.data_path.glob("*_*.csv"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "shapefile_loaded": self._shapefile_loaded,
            "shapefile_path": str(self._shapefile_path),
            "cached_files": len(cache_files),
            "cache_size_mb": total_size / (1024 * 1024),
            "cached_countries": len(
                set(
                    f.stem.split("_")[0]
                    for f in cache_files
                    if len(f.stem.split("_")[0]) == 2
                )
            ),
            "regions_cache_entries": len(self._regions_cache),
            "data_path": str(self.data_path),
        }

    def __repr__(self) -> str:
        """Return a human-readable summary of the manager."""
        s = self.stats()
        shapefile_status = (
            "✓ loaded"
            if s["shapefile_loaded"]
            else f"⏳ will load from {Path(s['shapefile_path']).name}"
        )
        return (
            f"EurostatManager(\n"
            f"  shapefile={shapefile_status},\n"
            f"  cache={s['cached_files']} files ({s['cache_size_mb']:.1f} MB),\n"
            f"  countries={s['cached_countries']},\n"
            f"  data='{s['data_path']}'\n"
            f")"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_cache(
        self,
        dataset_code: str,
        regions: List[str],
        filename: str,
        country: str,
        download: bool,
    ) -> pd.DataFrame:
        """Orchestrate download-or-read and automatic cleaning for a dataset.

        Checks whether a cached CSV exists for *country* and *filename*.
        Downloads from Eurostat if the file is missing or *download* is
        ``True``, saves the result, then returns a cleaned DataFrame.

        Parameters
        ----------
        dataset_code : str
            Eurostat dataset code (e.g. ``"demo_r_mlife"``).
        regions : list[str]
            NUTS region codes to query.
        filename : str
            Stem of the CSV cache file (without country prefix or extension).
        country : str
            ISO country code used to name the cache file.
        download : bool
            If ``True``, forces a fresh download even if a cache file exists.

        Returns
        -------
        DataFrame
            Cleaned DataFrame.
        """
        filepath = self.data_path / f"{country}_{filename}.csv"
        should_download = download or not filepath.exists()

        if should_download:
            data = self._download_dataset(dataset_code, regions)
            data.to_csv(filepath, index=False)
            print(f"⬇️  [Eurostat] Downloaded → {filepath.name}")
        else:
            data = pd.read_csv(filepath)
            print(f"💾 [Eurostat] Cache → {filepath.name}")

        return self._clean_data(data)

    def _download_dataset(
        self,
        dataset_code: str,
        regions: List[str],
    ) -> pd.DataFrame:
        """Download data for all specified NUTS regions via the Eurostat API.

        Iterates over *regions* one by one, calling the API for each, and
        concatenates the results into a single DataFrame.  Progress is printed
        to stdout every 5 regions.

        Parameters
        ----------
        dataset_code : str
            Eurostat dataset code.
        regions : list[str]
            List of NUTS_ID codes to query.

        Returns
        -------
        DataFrame
            Raw (uncleaned) concatenated data for all regions.
        """
        print(f"🌐 [Eurostat] Downloading... ({len(regions)} regions)")

        frames = []
        for i, region in enumerate(regions, 1):
            if i % 5 == 0:
                print(f"  → Progress: {i}/{len(regions)}")
            response = self.client.get_dataset(dataset_code, params={"geo": region})
            df = response.to_dataframe()
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean a raw Eurostat DataFrame.

        Performs the following transformations when the corresponding columns
        are present:

        - **age**: removes aggregate codes (``"TOTAL"``, ``"UNK"``,
          ``"Y_OPEN"``), converts labels to integers via :meth:`parse_age`,
          and drops rows where parsing fails.
        - **time**: converts year strings to integers.

        Parameters
        ----------
        data : DataFrame
            Raw DataFrame as returned by the Eurostat API.

        Returns
        -------
        DataFrame
            Cleaned DataFrame with reset integer index.
        """
        data = data.copy()

        if "age" in data.columns:
            data = data[~data["age"].isin(["TOTAL", "UNK", "Y_OPEN"])]
            data["age"] = data["age"].map(self.parse_age)
            data = data.dropna(subset=["age"])
            data["age"] = data["age"].astype(int)

        if "time" in data.columns:
            data["time"] = data["time"].astype(int)

        return data.reset_index(drop=True)


# =============================================================================
# Backward-compatible wrapper functions
# =============================================================================
# These thin wrappers preserve the old functional API for scripts written
# before the Eurostat_data class was introduced.

def load_mxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Load mortality rate data (backward-compatible wrapper).

    .. deprecated::
        Use :meth:`Eurostat_data.load` with ``dataset_type="mortality"`` instead.

    Parameters
    ----------
    shapef : GeoDataFrame, optional
        Pre-loaded NUTS shapefile.
    country : str, optional
        ISO country code (default ``"FR"``).
    nuts_level : int, optional
        NUTS level (default ``2``).
    data_path : str, optional
        Cache directory path (default ``"../data"``).
    download : bool, optional
        Force re-download (default ``False``).

    Returns
    -------
    DataFrame
        Cleaned mortality DataFrame.
    """
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("mortality", country, download=download)


def load_dxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Load death count data (backward-compatible wrapper).

    .. deprecated::
        Use :meth:`Eurostat_data.load` with ``dataset_type="deaths"`` instead.

    Parameters
    ----------
    shapef : GeoDataFrame, optional
        Pre-loaded NUTS shapefile.
    country : str, optional
        ISO country code (default ``"FR"``).
    nuts_level : int, optional
        NUTS level (default ``2``).
    data_path : str, optional
        Cache directory path (default ``"../data"``).
    download : bool, optional
        Force re-download (default ``False``).

    Returns
    -------
    DataFrame
        Cleaned deaths DataFrame.
    """
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("deaths", country, download=download)


def load_lxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Load population data (backward-compatible wrapper).

    .. deprecated::
        Use :meth:`Eurostat_data.load` with ``dataset_type="population"`` instead.

    Parameters
    ----------
    shapef : GeoDataFrame, optional
        Pre-loaded NUTS shapefile.
    country : str, optional
        ISO country code (default ``"FR"``).
    nuts_level : int, optional
        NUTS level (default ``2``).
    data_path : str, optional
        Cache directory path (default ``"../data"``).
    download : bool, optional
        Force re-download (default ``False``).

    Returns
    -------
    DataFrame
        Cleaned population DataFrame.
    """
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("population", country, download=download)


def age_year_pivot_table(data_raw, region, gender, indicator):
    """Create an age × year pivot table (backward-compatible wrapper).

    .. deprecated::
        Use :meth:`Eurostat_data.pivot_age_year` instead.

    Parameters
    ----------
    data_raw : DataFrame
        Source data returned by one of the ``load_*`` functions.
    region : str
        NUTS region code.
    gender : str
        Gender code (``"M"``, ``"F"``, ``"T"``).
    indicator : str
        Eurostat indicator code.

    Returns
    -------
    tuple
        ``(pivot_table, ages_array, years_array)``
    """
    # No shapefile needed for pivot operations
    manager = Eurostat_data(auto_load_shapefile=False)
    return manager.pivot_age_year(data_raw, region, gender, indicator)


def filter_shapefile(shapef=None, country="FR", nuts_level=2, exclude_outremer=True):
    """Filter the NUTS shapefile (backward-compatible wrapper).

    .. deprecated::
        Use :meth:`Eurostat_data.filter_shapefile` instead.

    Parameters
    ----------
    shapef : GeoDataFrame, optional
        Pre-loaded NUTS shapefile.
    country : str, optional
        ISO country code (default ``"FR"``).
    nuts_level : int, optional
        NUTS level (default ``2``).
    exclude_outremer : bool, optional
        Exclude non-continental regions (default ``True``).

    Returns
    -------
    GeoDataFrame
        Filtered NUTS GeoDataFrame.
    """
    manager = Eurostat_data(shapefile=shapef, nuts_level=nuts_level)
    return manager.filter_shapefile(country, exclude_outremer=exclude_outremer)


def parse_age(label):
    """Parse an Eurostat age label to an integer (backward-compatible wrapper).

    .. deprecated::
        Use :meth:`Eurostat_data.parse_age` instead.

    Parameters
    ----------
    label : str
        Eurostat age label (e.g. ``"Y25"``, ``"Y_LT1"``).

    Returns
    -------
    int or None
        Parsed integer age, or ``None`` if the label is unrecognised.
    """
    return Eurostat_data.parse_age(label)

