# =============================================================================
# Eurostat Data Manager - Smart with Shapefile Auto-Load
# =============================================================================

"""
Eurostat Data Manager - Smart with Shapefile Auto-Load
=====================================================

"""

import pandas as pd
from eurostatapiclient import EurostatAPIClient
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Literal, Union


class EurostatConfig:
    """Centralized configuration for paths and default parameters."""
    
    # Default path for NUTS shapefile
    DEFAULT_SHAPEFILE_PATH = Path("C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp")
    
    # Default data path
    DEFAULT_DATA_PATH = Path("../data")
    
    # Available datasets
    DATASETS = {
        "mortality": "demo_r_mlife",
        "deaths": "demo_r_magec",
        "population": "demo_r_d2jan"
    }
    
    # Regions to exclude by country
    EXCLUDE_REGIONS = {
        "FR": ["FRY1", "FRY2", "FRY3", "FRY4", "FRY5"],  # DOM-TOM
        "PT": ["PT20", "PT30"],                           # Azores, Madeira
        "ES": ["ES63", "ES64", "ES70"],                   # Ceuta, Melilla, Canary Islands
        "NO": ["NO0B"],                                   # Svalbard
    }
    
    @classmethod
    def set_default_shapefile(cls, path: Union[str, Path]) -> None:
        """
        Changes the default shapefile path globally.
        
        Examples
        --------
        >>> EurostatConfig.set_default_shapefile("D:/data/NUTS_2024.shp")
        """
        cls.DEFAULT_SHAPEFILE_PATH = Path(path)
    
    @classmethod
    def set_default_data_path(cls, path: Union[str, Path]) -> None:
        """
        Changes the default data path globally.
        
        Examples
        --------
        >>> EurostatConfig.set_default_data_path("D:/eurostat_cache")
        """
        cls.DEFAULT_DATA_PATH = Path(path)


class Eurostat_data:
    """
    Single manager for all Eurostat operations.
    
    The NUTS shapefile is loaded automatically from the default path.
    No need to handle 'shapef' in your code!
    
    Examples
    
    --------
    # Standard usage (shapefile auto-loaded)
    >>> manager = EurostatManager()
    >>> mortality = manager.load("mortality", "FR")
    
    # With a custom shapefile
    >>> manager = EurostatManager(shapefile_path="D:/custom/nuts.shp")
    
    # Or pass a GeoDataFrame directly
    >>> custom_shapef = gpd.read_file("path/to/shapefile.shp")
    >>> manager = EurostatManager(shapefile=custom_shapef)
    
    # Change the default for the whole application
    >>> EurostatConfig.set_default_shapefile("D:/data/NUTS_2024.shp")
    >>> manager = EurostatManager()  # Will use the new path
    """
    
    def __init__(
        self,
        shapefile: Optional[gpd.GeoDataFrame] = None,
        shapefile_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        language: str = "en",
        nuts_level: int = 2,
        auto_load_shapefile: bool = True
    ):
        """
        Initializes the Eurostat manager.
        
        Parameters
        ----------
        shapefile : GeoDataFrame, optional
            Already loaded NUTS shapefile (takes priority if provided)
        shapefile_path : str or Path, optional
            Custom shapefile path (uses default if None)
        data_path : str or Path, optional
            Data storage directory (uses default if None)
        language : str
            Language for Eurostat metadata
        nuts_level : int
            Default NUTS level (1, 2 or 3)
        auto_load_shapefile : bool
            If True, automatically loads the shapefile on first use
        """
        self._shapefile = shapefile
        self._shapefile_path = Path(shapefile_path) if shapefile_path else EurostatConfig.DEFAULT_SHAPEFILE_PATH
        self._auto_load = auto_load_shapefile
        self._shapefile_loaded = shapefile is not None
        
        self.data_path = Path(data_path) if data_path else EurostatConfig.DEFAULT_DATA_PATH
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.nuts_level = nuts_level
        
        # Eurostat API client
        self.client = EurostatAPIClient("1.0", "json", language)
        
        # Regions cache to avoid recomputing
        self._regions_cache: Dict[Tuple[str, int, bool], List[str]] = {}
    
    @property
    def shapefile(self) -> gpd.GeoDataFrame:
        """
        Access to the shapefile with auto-loading if necessary.
        
        The user never needs to call this property directly.
        """
        if self._shapefile is None and self._auto_load:
            self._load_shapefile()
        
        if self._shapefile is None:
            raise ValueError(
                f"Shapefile not loaded. Path used: {self._shapefile_path}\n"
                f"Check that the file exists or set another path:\n"
                f"  EurostatConfig.set_default_shapefile('path/to/shapefile.shp')\n"
                f"  or manager = EurostatManager(shapefile_path='path/to/shapefile.shp')"
            )
        
        return self._shapefile
    
    def _load_shapefile(self) -> None:
        """Loads the shapefile from the configured path."""
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
        shapefile_path: Optional[Union[str, Path]] = None
    ) -> 'Eurostat_data':
        """
        Sets a new shapefile (chainable).
        
        Parameters
        ----------
        shapefile : GeoDataFrame, optional
            Already loaded shapefile
        shapefile_path : str or Path, optional
            Path to a new shapefile to load
        
        Examples
        --------
        >>> manager.set_shapefile(shapefile_path="D:/new.shp").load("mortality", "FR")
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
        
        self._regions_cache.clear()
        return self
    
    # =========================================================================
    # Main methods - Simplified API
    # =========================================================================
    
    def load(
        self,
        dataset_type: Literal["mortality", "deaths", "population"],
        country: str = "FR",
        nuts_level: Optional[int] = None,
        download: bool = False,
        exclude_outremer: bool = True
    ) -> pd.DataFrame:
        """
        Loads an Eurostat dataset smartly.
        
        Parameters
        ----------
        dataset_type : str
            Dataset type: "mortality", "deaths" or "population"
        country : str
            ISO 2-letter country code
        nuts_level : int, optional
            NUTS level (uses default if None)
        download : bool
            Forces download even if cache exists
        exclude_outremer : bool
            Excludes non-continental regions
        
        Returns
        -------
        Cleaned DataFrame ready to use
        
        Examples
        --------
        >>> manager = EurostatManager()  # Shapefile auto-loaded!
        >>> mortality = manager.load("mortality", "FR")
        >>> deaths = manager.load("deaths", "BE", download=True)
        """
        nuts_level = nuts_level or self.nuts_level
        
        # Retrieve regions (with cache)
        regions = self.get_regions(country, nuts_level, exclude_outremer)
        
        # Dataset code
        dataset_code = EurostatConfig.DATASETS[dataset_type]
        
        # Filename with explicit type
        filename_map = {
            "mortality": "mxt_raw",
            "deaths": "Dxt_raw",
            "population": "Lxt_raw"
        }
        filename = filename_map[dataset_type]
        
        # Load with cache
        return self._load_and_cache(
            dataset_code=dataset_code,
            regions=regions,
            filename=filename,
            country=country,
            download=download
        )
    
    def load_all(
        self,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Loads all datasets for a country.
        
        Returns
        -------
        Dict with keys: "mortality", "deaths", "population"
        
        Examples
        --------
        >>> manager = EurostatManager()
        >>> data = manager.load_all("FR")
        >>> mortality = data["mortality"]
        >>> deaths = data["deaths"]
        >>> population = data["population"]
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
        indicator: str
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Creates an age × year cross-table for a given region.
        
        Parameters
        ----------
        data : DataFrame
            Source data (from load())
        region : str
            NUTS region code
        gender : str
            Gender ("M", "F", "T")
        indicator : str
            Eurostat indicator code
        
        Returns
        -------
        tuple
            (pivot_table, ages_array, years_array)
        
        Examples
        --------
        >>> pivot, ages, years = manager.pivot_age_year(
        ...     mortality, "FR10", "M", "LIFE_EXP"
        ... )
        """
        # Filtering
        subset = data[
            (data['geo'] == region) &
            (data['sex'] == gender) &
            (data['indic_de'] == indicator)
        ].reset_index(drop=True)
        
        # Pivot
        pivot = pd.pivot_table(
            subset,
            values='values',
            index=['age'],
            columns=['time'],
            aggfunc="sum",
            fill_value=1e-6,
            observed=True
        )
        
        # Sort by ascending age
        pivot = pivot.sort_values("age").reset_index(drop=True)
        ages = pivot.index.values.astype('int')
        years = pivot.columns.values.astype('int')
        
        return pivot, ages, years
    
    def get_regions(
        self,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        exclude_outremer: bool = True
    ) -> List[str]:
        """
        Retrieves the list of NUTS codes for a country (with cache).
        
        Parameters
        ----------
        country : str
            ISO 2-letter country code
        nuts_level : int, optional
            NUTS level (uses default if None)
        exclude_outremer : bool
            Excludes non-continental regions
        
        Returns
        -------
        List of NUTS codes
        
        Examples
        --------
        >>> regions_fr = manager.get_regions("FR", nuts_level=2)
        >>> regions_be = manager.get_regions("BE")
        """
        nuts_level = nuts_level or self.nuts_level
        
        # Cache key
        cache_key = (country, nuts_level, exclude_outremer)
        
        # Check cache
        if cache_key not in self._regions_cache:
            filtered = self.filter_shapefile(country, nuts_level, exclude_outremer)
            self._regions_cache[cache_key] = filtered["NUTS_ID"].tolist()
        
        return self._regions_cache[cache_key]
    
    def filter_shapefile(
        self,
        country: str = "FR",
        nuts_level: Optional[int] = None,
        exclude_outremer: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Filters the NUTS shapefile by country and level.
        
        Parameters
        ----------
        country : str
            ISO 2-letter country code
        nuts_level : int, optional
            NUTS level (uses default if None)
        exclude_outremer : bool
            Excludes non-continental regions
        
        Returns
        -------
        Filtered and sorted GeoDataFrame
        """
        nuts_level = nuts_level or self.nuts_level
        
        filtered = self.shapefile[
            (self.shapefile["CNTR_CODE"] == country) &
            (self.shapefile["LEVL_CODE"] == nuts_level)
        ].copy()
        
        if exclude_outremer and country in EurostatConfig.EXCLUDE_REGIONS:
            filtered = filtered[
                ~filtered["NUTS_ID"].isin(EurostatConfig.EXCLUDE_REGIONS[country])
            ]
        
        return filtered.sort_values("NUTS_ID").reset_index(drop=True)
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    @staticmethod
    def parse_age(label) -> Optional[int]:
        """
        Parses an Eurostat age label into a numeric value.
        
        Examples
        --------
        >>> EurostatManager.parse_age("Y25")    # 25
        >>> EurostatManager.parse_age("Y_LT1")  # 0
        >>> EurostatManager.parse_age("Y_GE85") # 85
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
    
    def add_exclusion(self, country: str, regions: List[str]) -> 'Eurostat_data':
        """
        Adds regions to exclude for a country (chainable).
        
        Examples
        --------
        >>> manager.add_exclusion("IT", ["ITG1", "ITG2"]).load("mortality", "IT")
        """
        if country not in EurostatConfig.EXCLUDE_REGIONS:
            EurostatConfig.EXCLUDE_REGIONS[country] = []
        EurostatConfig.EXCLUDE_REGIONS[country].extend(regions)
        self._regions_cache.clear()
        return self
    
    def list_datasets(self) -> List[str]:
        """Lists the available datasets."""
        return list(EurostatConfig.DATASETS.keys())
    
    def cache_info(self, country: Optional[str] = None) -> Union[Dict[str, bool], Dict[str, Dict[str, bool]]]:
        """
        Checks which datasets are cached.
        
        Parameters
        ----------
        country : str, optional
            If None, checks all detected countries
        
        Returns
        -------
        Dict with status of each dataset
        
        Examples
        --------
        >>> manager.cache_info("FR")
        {'mortality': True, 'deaths': False, 'population': True}
        
        >>> manager.cache_info()  # All countries
        {'FR': {'mortality': True, 'deaths': False, ...}, 'BE': {...}}
        """
        filename_map = {
            "mortality": "mxt_raw",
            "deaths": "Dxt_raw",
            "population": "Lxt_raw"
        }
        
        if country:
            return {
                dataset: (self.data_path / f"{country}_{filename}.csv").exists()
                for dataset, filename in filename_map.items()
            }
        else:
            # Auto-detect cached countries
            countries = set()
            for file in self.data_path.glob("*_*.csv"):
                country_code = file.stem.split("_")[0]
                if len(country_code) == 2:  # ISO code
                    countries.add(country_code)
            
            return {
                c: self.cache_info(c) for c in sorted(countries)
            }
    
    def clear_cache(self, country: Optional[str] = None, dataset: Optional[str] = None) -> int:
        """
        Deletes cached files.
        
        Parameters
        ----------
        country : str, optional
            Country code (if None, all countries)
        dataset : str, optional
            Dataset type (if None, all datasets)
        
        Returns
        -------
        Number of deleted files
        
        Examples
        --------
        >>> manager.clear_cache("FR")  # Deletes everything for France
        >>> manager.clear_cache("FR", "mortality")  # Deletes only FR mortality
        >>> manager.clear_cache()  # Clears entire cache
        """
        count = 0
        
        if country and dataset:
            # Targeted deletion
            filename_map = {"mortality": "mxt_raw", "deaths": "Dxt_raw", "population": "Lxt_raw"}
            filepath = self.data_path / f"{country}_{filename_map[dataset]}.csv"
            if filepath.exists():
                filepath.unlink()
                count = 1
                print(f"🗑️  [Cache] Deleted: {filepath.name}")
        elif country:
            # Delete a country
            for file in self.data_path.glob(f"{country}_*.csv"):
                file.unlink()
                count += 1
            print(f"🗑️  [Cache] Deleted: {count} files for {country}")
        else:
            # Full deletion
            for file in self.data_path.glob("*_*.csv"):
                file.unlink()
                count += 1
            if count > 0:
                print(f"🗑️  [Cache] Deleted: {count} files")
        
        return count
    
    def stats(self) -> Dict[str, any]:
        """
        Statistics about the manager.
        
        Returns
        -------
        Dict with various stats
        """
        cache_files = list(self.data_path.glob("*_*.csv"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "shapefile_loaded": self._shapefile_loaded,
            "shapefile_path": str(self._shapefile_path),
            "cached_files": len(cache_files),
            "cache_size_mb": total_size / (1024 * 1024),
            "cached_countries": len(set(f.stem.split("_")[0] for f in cache_files if len(f.stem.split("_")[0]) == 2)),
            "regions_cache_entries": len(self._regions_cache),
            "data_path": str(self.data_path),
        }
    
    def __repr__(self) -> str:
        """Human-readable representation of the manager."""
        stats = self.stats()
        shapefile_status = "✓ loaded" if stats['shapefile_loaded'] else f"⏳ will load from {Path(stats['shapefile_path']).name}"
        
        return (
            f"EurostatManager(\n"
            f"  shapefile={shapefile_status},\n"
            f"  cache={stats['cached_files']} files ({stats['cache_size_mb']:.1f} MB),\n"
            f"  countries={stats['cached_countries']},\n"
            f"  data='{stats['data_path']}'\n"
            f")"
        )
    
    # =========================================================================
    # Private methods (internal)
    # =========================================================================
    
    def _load_and_cache(
        self,
        dataset_code: str,
        regions: List[str],
        filename: str,
        country: str,
        download: bool
    ) -> pd.DataFrame:
        """Loads a dataset with cache management."""
        filepath = self.data_path / f"{country}_{filename}.csv"
        
        # Download decision
        should_download = download or not filepath.exists()
        
        if should_download:
            data = self._download_dataset(dataset_code, regions)
            data.to_csv(filepath, index=False)
            print(f"⬇️  [Eurostat] Downloaded → {filepath.name}")
        else:
            data = pd.read_csv(filepath)
            print(f"💾 [Eurostat] Cache → {filepath.name}")
        
        # Automatic cleaning
        return self._clean_data(data)
    
    def _download_dataset(
        self,
        dataset_code: str,
        regions: List[str]
    ) -> pd.DataFrame:
        """Downloads data for all regions."""
        print(f"🌐 [Eurostat] Downloading... ({len(regions)} regions)")
        
        frames = []
        for i, region in enumerate(regions, 1):
            if i % 5 == 0:  # Feedback every 5 regions
                print(f"  → Progress: {i}/{len(regions)}")
            
            response = self.client.get_dataset(dataset_code, params={"geo": region})
            df = response.to_dataframe()
            frames.append(df)
        
        return pd.concat(frames, ignore_index=True)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans data (invalid ages, type conversion)."""
        data = data.copy()
        
        # Age cleaning
        if "age" in data.columns:
            data = data[~data["age"].isin(["TOTAL", "UNK", "Y_OPEN"])]
            data["age"] = data["age"].map(self.parse_age)
            data = data.dropna(subset=["age"])
            data["age"] = data["age"].astype(int)
        
        # Year conversion
        if "time" in data.columns:
            data["time"] = data["time"].astype(int)
        
        return data.reset_index(drop=True)


# =============================================================================
# Compatibility functions (simple wrappers)
# =============================================================================

def load_mxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Compatibility: loads mortality rates."""
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("mortality", country, download=download)


def load_dxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Compatibility: loads deaths."""
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("deaths", country, download=download)


def load_lxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Compatibility: loads populations."""
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("population", country, download=download)


def age_year_pivot_table(data_raw, region, gender, indicator):
    """Compatibility: creates an age × year pivot table."""
    manager = Eurostat_data(auto_load_shapefile=False)  # No shapefile needed for pivot
    return manager.pivot_age_year(data_raw, region, gender, indicator)


def filter_shapefile(shapef=None, country="FR", nuts_level=2, exclude_outremer=True):
    """Compatibility: filters the shapefile."""
    manager = Eurostat_data(shapefile=shapef, nuts_level=nuts_level)
    return manager.filter_shapefile(country, exclude_outremer=exclude_outremer)


def parse_age(label):
    """Compatibility: parses an age label."""
    return Eurostat_data.parse_age(label)




# =============================================================================
# EurostatChecker - Coherence checks for Eurostat demographic data
# =============================================================================

"""
EurostatChecker
===============
Validates coherence of Eurostat datasets (mortality, deaths, population).

Usage
-----
>>> checker = EurostatChecker(mortality=mxt, deaths=dxt, population=lxt)
>>> report = checker.run_all()
>>> report.summary()

Or run individual checks:
>>> checker.check_gender_consistency(mxt, indicator="LIFE_EXP")
"""

# =============================================================================
# EurostatChecker - Vérifications de cohérence des données Eurostat
# =============================================================================
#
# Indicateurs supportés (demo_r_mlife) :
#   DEATHRATE   : taux de mortalité pour 1000    → bornes [0, 1000], M > F
#   LIFEXP      : espérance de vie en années     → bornes [0, 120],  F > M
#   NUMBERDYING : nombre de décès (table de vie) → bornes [0, +∞],   M > F
#   PROBDEATH   : probabilité de décès           → bornes [0, 1],    M > F
#   PROBSURV    : probabilité de survie          → bornes [0, 1],    F > M
#   PYLIVED     : années vécues                  → bornes [0, +∞],   neutre
#   SURVIVORS   : survivants (table de vie)      → bornes [0, +∞],   neutre
#   TOTPYLIVED  : total années vécues            → bornes [0, +∞],   neutre
#
# Utilisation rapide :
#   from check_error import check_eurostat
#   report = check_eurostat(mortality=mxt, deaths=dxt, population=lxt)
#   report.summary()
#   issues = report.get_issues()
#   issues[0].details.head()
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Literal


# =============================================================================
# Structures de résultats
# =============================================================================

@dataclass
class CheckResult:
    """Résultat d'un check individuel."""
    check_name: str
    status: Literal["OK", "WARNING", "ERROR"]
    message: str
    details: Optional[pd.DataFrame] = None
    n_issues: int = 0

    def __repr__(self):
        icon = {"OK": "✅", "WARNING": "⚠️ ", "ERROR": "❌"}[self.status]
        return f"{icon} [{self.check_name}] {self.message}"


@dataclass
class CheckReport:
    """Rapport agrégé de tous les checks."""
    results: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult):
        self.results.append(result)

    def summary(self):
        """Affiche un résumé lisible."""
        ok   = sum(1 for r in self.results if r.status == "OK")
        warn = sum(1 for r in self.results if r.status == "WARNING")
        err  = sum(1 for r in self.results if r.status == "ERROR")
        print(f"\n{'='*65}")
        print(f"  EUROSTAT COHERENCE REPORT")
        print(f"  ✅ {ok} OK  |  ⚠️  {warn} WARNING  |  ❌ {err} ERROR")
        print(f"{'='*65}")
        for r in self.results:
            print(f"  {r}")
            if r.n_issues > 0:
                print(f"     → {r.n_issues} lignes problématiques")
        print(f"{'='*65}\n")

    def get_issues(self) -> List[CheckResult]:
        """Retourne uniquement les WARNING et ERROR."""
        return [r for r in self.results if r.status != "OK"]

    @property
    def has_errors(self) -> bool:
        return any(r.status == "ERROR" for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.status == "WARNING" for r in self.results)


# =============================================================================
# Classe principale
# =============================================================================

class EurostatChecker:
    """
    Vérifications de cohérence pour les datasets Eurostat démographiques.

    Paramètres
    ----------
    mortality   : DataFrame issu de manager.load("mortality")
    deaths      : DataFrame issu de manager.load("deaths")
    population  : DataFrame issu de manager.load("population")
    tolerance   : tolérance relative pour les checks d'égalité (défaut 5%)
    spike_threshold : ratio max autorisé d'une année sur l'autre (défaut 2x)

    Colonnes attendues : geo, sex, age, time, values
    Valeurs de sex     : "M" (hommes), "F" (femmes), "T" (total)
    """

    # -------------------------------------------------------------------------
    # Configuration par indicateur
    # higher_sex : sexe biologiquement attendu AU-DESSUS du total
    #   "M" -> hommes > total > femmes  (mortalité)
    #   "F" -> femmes > total > hommes  (espérance de vie, survie)
    #   None -> pas de contrainte directionnelle (indicateurs additifs)
    # -------------------------------------------------------------------------
    INDICATOR_CONFIG = {
        "DEATHRATE":   {"lower": 0,   "upper": 1000, "higher_sex": "M"},
        "LIFEXP":      {"lower": 0,   "upper": 120,  "higher_sex": "F"},
        "NUMBERDYING": {"lower": 0,   "upper": None, "higher_sex": "M"},
        "PROBDEATH":   {"lower": 0,   "upper": 1,    "higher_sex": "M"},
        "PROBSURV":    {"lower": 0,   "upper": 1,    "higher_sex": "F"},
        "PYLIVED":     {"lower": 0,   "upper": None, "higher_sex": None},
        "SURVIVORS":   {"lower": 0,   "upper": None, "higher_sex": None},
        "TOTPYLIVED":  {"lower": 0,   "upper": None, "higher_sex": None},
    }

    def __init__(
        self,
        mortality:        Optional[pd.DataFrame] = None,
        deaths:           Optional[pd.DataFrame] = None,
        population:       Optional[pd.DataFrame] = None,
        tolerance:        float = 0.05,
        spike_threshold:  float = 2.0,
    ):
        self.mortality       = mortality
        self.deaths          = deaths
        self.population      = population
        self.tolerance       = tolerance
        self.spike_threshold = spike_threshold

    # =========================================================================
    # Lancement de tous les checks
    # =========================================================================

    def run_all(self) -> CheckReport:
        """Lance tous les checks applicables et retourne un CheckReport."""
        report = CheckReport()

        if self.mortality is not None:
            report.add(self.check_gender_consistency(self.mortality))
            report.add(self.check_deathrate_gender_pattern(self.mortality))
            report.add(self.check_value_bounds_by_indicator(self.mortality))
            report.add(self.check_age_monotonicity(self.mortality))
            report.add(self.check_temporal_spikes(self.mortality, "mortality"))
            report.add(self.check_completeness(self.mortality, "mortality"))

        if self.deaths is not None:
            report.add(self.check_gender_additivity(self.deaths, "deaths"))
            report.add(self.check_value_bounds(self.deaths, "deaths", lower=0))
            report.add(self.check_temporal_spikes(self.deaths, "deaths"))
            report.add(self.check_completeness(self.deaths, "deaths"))

        if self.population is not None:
            report.add(self.check_gender_additivity(self.population, "population"))
            report.add(self.check_value_bounds(self.population, "population", lower=0))
            report.add(self.check_completeness(self.population, "population"))

        if self.deaths is not None and self.population is not None:
            report.add(self.check_implied_mortality_rate())

        return report

    # =========================================================================
    # Check 1 — Direction biologique par indicateur
    # =========================================================================

    def check_gender_consistency(self, df: pd.DataFrame) -> CheckResult:
        """
        Vérifie que la direction homme/femme est biologiquement cohérente.

        Pour DEATHRATE, PROBDEATH, NUMBERDYING -> M > T > F attendu.
        Pour LIFEXP, PROBSURV                  -> F > T > M attendu.

        Flagge chaque ligne qui viole cette règle à plus de `tolerance` près.
        """
        check_name = "gender_consistency [mortality]"

        if not {"geo", "sex", "time", "values"}.issubset(df.columns):
            return CheckResult(check_name, "WARNING", "Colonnes requises manquantes")
        if "indic_de" not in df.columns:
            return CheckResult(check_name, "WARNING", "Colonne indic_de absente — check ignoré")

        group_cols = ["geo", "time", "age"] if "age" in df.columns else ["geo", "time"]
        issues = []

        for indic, config in self.INDICATOR_CONFIG.items():
            higher_sex = config.get("higher_sex")
            if higher_sex is None:
                continue

            sub = df[df["indic_de"] == indic]
            if sub.empty:
                continue

            try:
                piv = sub.pivot_table(
                    index=group_cols, columns="sex", values="values", aggfunc="mean"
                ).reset_index()
            except Exception:
                continue

            if "T" not in piv.columns:
                continue

            lower_sex = "M" if higher_sex == "F" else "F"

            # Sexe inférieur ne doit pas dépasser T
            if lower_sex in piv.columns:
                s   = piv[["geo", "time", lower_sex, "T"]].dropna()
                bad = s[s[lower_sex] > s["T"] * (1 + self.tolerance)].copy()
                if not bad.empty:
                    bad["indic_de"]  = indic
                    bad["violation"] = f"{lower_sex} > T  (attendu : {higher_sex} > T > {lower_sex})"
                    bad["excess_%"]  = ((bad[lower_sex] - bad["T"]) / bad["T"] * 100).round(2)
                    issues.append(bad[["geo", "time", "indic_de", "violation", lower_sex, "T", "excess_%"]])

            # Sexe supérieur ne doit pas être sous T
            if higher_sex in piv.columns:
                s   = piv[["geo", "time", higher_sex, "T"]].dropna()
                bad = s[s["T"] > s[higher_sex] * (1 + self.tolerance)].copy()
                if not bad.empty:
                    bad["indic_de"]  = indic
                    bad["violation"] = f"T > {higher_sex}  (attendu : {higher_sex} > T)"
                    bad["deficit_%"] = ((bad["T"] - bad[higher_sex]) / bad["T"] * 100).round(2)
                    issues.append(bad[["geo", "time", "indic_de", "violation", higher_sex, "T", "deficit_%"]])

        if not issues:
            return CheckResult(check_name, "OK",
                "Direction genre conforme à la biologie pour tous les indicateurs")

        all_issues = pd.concat(issues, ignore_index=True)
        return CheckResult(check_name, "ERROR",
            f"{len(all_issues)} violations de direction genre (M>T>F ou F>T>M selon indicateur)",
            details=all_issues, n_issues=len(all_issues))

    # =========================================================================
    # Check 2 — Pattern global F > M sur DEATHRATE
    # =========================================================================

    def check_deathrate_gender_pattern(
        self,
        df: pd.DataFrame,
        min_anomaly_rate: float = 0.10,
    ) -> CheckResult:
        """
        Sur DEATHRATE uniquement : mesure la fréquence du pattern F > M.

        Biologiquement M > F est attendu à quasi tous les âges et régions.
        Si F > M dépasse `min_anomaly_rate` (10% par défaut) des lignes,
        le dataset est considéré globalement suspect.

        Le détail par région indique si c'est concentré ou généralisé.
        """
        check_name = "deathrate_gender_pattern [mortality]"

        if "indic_de" not in df.columns:
            return CheckResult(check_name, "WARNING", "Colonne indic_de absente — check ignoré")

        sub = df[df["indic_de"] == "DEATHRATE"]
        if sub.empty:
            return CheckResult(check_name, "WARNING", "Indicateur DEATHRATE absent")

        group_cols = ["geo", "time", "age"] if "age" in df.columns else ["geo", "time"]

        try:
            piv = sub.pivot_table(
                index=group_cols, columns="sex", values="values", aggfunc="mean"
            ).reset_index().dropna(subset=["M", "F"])
        except Exception as e:
            return CheckResult(check_name, "WARNING", f"Impossible de pivoter : {e}")

        if piv.empty:
            return CheckResult(check_name, "WARNING", "Pas assez de données M/F")

        total_rows   = len(piv)
        f_above_m    = int((piv["F"] > piv["M"]).sum())
        f_above_t    = int((piv["F"] > piv["T"]).sum()) if "T" in piv.columns else None
        anomaly_rate = f_above_m / total_rows

        # Détail par région
        piv["F_gt_M"] = piv["F"] > piv["M"]
        by_geo = (
            piv.groupby("geo")["F_gt_M"]
            .agg(n_anomalies="sum", n_total="count")
        )
        by_geo["rate_%"] = (by_geo["n_anomalies"] / by_geo["n_total"] * 100).round(1)
        by_geo = by_geo[by_geo["n_anomalies"] > 0].sort_values("rate_%", ascending=False)

        msg_parts = [f"F > M dans {f_above_m}/{total_rows} cas ({anomaly_rate*100:.1f}%)"]
        if f_above_t is not None:
            msg_parts.append(f"F > Total dans {f_above_t} cas")

        if anomaly_rate > min_anomaly_rate:
            msg_parts.append(f"dépasse le seuil de {min_anomaly_rate*100:.0f}% → données suspectes")
            status = "ERROR"
        else:
            status = "OK"

        return CheckResult(check_name, status, " | ".join(msg_parts),
            details=by_geo.reset_index(), n_issues=f_above_m)

    # =========================================================================
    # Check 3 — Bornes par indicateur
    # =========================================================================

    def check_value_bounds_by_indicator(self, df: pd.DataFrame) -> CheckResult:
        """
        Vérifie que chaque valeur reste dans les bornes physiquement plausibles
        définies par indicateur dans INDICATOR_CONFIG.

        Exemples :
          PROBDEATH / PROBSURV -> [0, 1]
          LIFEXP               -> [0, 120]
          DEATHRATE            -> [0, 1000]
        """
        check_name = "value_bounds_by_indicator [mortality]"

        if "indic_de" not in df.columns:
            return self.check_value_bounds(df, "mortality")

        issues = []
        for indic, config in self.INDICATOR_CONFIG.items():
            sub = df[df["indic_de"] == indic]
            if sub.empty:
                continue
            lower, upper = config["lower"], config["upper"]
            mask = pd.Series(False, index=sub.index)
            if lower is not None: mask |= sub["values"] < lower
            if upper is not None: mask |= sub["values"] > upper
            bad = sub[mask].copy()
            if not bad.empty:
                bad["expected_bounds"] = f"[{lower}, {upper if upper else '+inf'}]"
                issues.append(bad)

        if not issues:
            return CheckResult(check_name, "OK",
                "Toutes les valeurs dans les bornes par indicateur")

        all_issues = pd.concat(issues, ignore_index=True)
        return CheckResult(check_name, "ERROR",
            f"{len(all_issues)} valeurs hors bornes (bornes spécifiques par indicateur)",
            details=all_issues, n_issues=len(all_issues))

    # =========================================================================
    # Check 4 — Additivité M + F ≈ T  (décès et population)
    # =========================================================================

    def check_gender_additivity(self, df: pd.DataFrame, dataset_label: str = "data") -> CheckResult:
        """
        Vérifie que Hommes + Femmes ≈ Total à `tolerance` près.
        S'applique aux décès et à la population (indicateurs de comptage).
        Une déviation > 5% signale un problème d'extraction ou d'agrégation.
        """
        check_name = f"gender_additivity [{dataset_label}]"

        if not {"geo", "sex", "time", "values"}.issubset(df.columns):
            return CheckResult(check_name, "WARNING", "Colonnes requises manquantes")

        group_cols = ["geo", "time", "age"] if "age" in df.columns else ["geo", "time"]

        try:
            piv = df.pivot_table(
                index=group_cols, columns="sex", values="values", aggfunc="sum"
            ).reset_index()
        except Exception as e:
            return CheckResult(check_name, "WARNING", f"Impossible de pivoter : {e}")

        if not {"T", "M", "F"}.issubset(piv.columns):
            return CheckResult(check_name, "WARNING", "Colonnes T, M, F manquantes")

        sub = piv[["geo", "time", "M", "F", "T"]].dropna()
        sub = sub[sub["T"] > 0].copy()
        sub["M_plus_F"] = sub["M"] + sub["F"]
        sub["diff_%"]   = ((sub["M_plus_F"] - sub["T"]) / sub["T"] * 100).abs().round(2)
        bad = sub[sub["diff_%"] > self.tolerance * 100]

        if bad.empty:
            return CheckResult(check_name, "OK",
                f"M + F ≈ T dans la tolérance de {self.tolerance*100:.0f}%")

        return CheckResult(check_name, "ERROR",
            f"{len(bad)} lignes ou M + F s'ecarte de T de >{self.tolerance*100:.0f}%",
            details=bad.sort_values("diff_%", ascending=False), n_issues=len(bad))

    # =========================================================================
    # Check 5 — Monotonicité par âge
    # =========================================================================

    def check_age_monotonicity(
        self,
        df: pd.DataFrame,
        dataset_label: str = "mortality",
        min_age: int = 30,
    ) -> CheckResult:
        """
        Sur DEATHRATE, PROBDEATH, NUMBERDYING uniquement.
        Vérifie que la mortalité croît avec l'âge après `min_age` ans.
        Une chute > 50% entre deux âges consécutifs est flaggée.

        Note : le creux de mortalité enfant (0-30 ans) est exclu volontairement
        car il est biologiquement normal (minimum vers 8-10 ans).
        """
        check_name = f"age_monotonicity [{dataset_label}]"

        if not {"geo", "sex", "time", "age", "values"}.issubset(df.columns):
            return CheckResult(check_name, "WARNING",
                "Colonnes geo, sex, time, age, values requises")

        monotone_indicators = {"DEATHRATE", "PROBDEATH", "NUMBERDYING"}
        working = df.copy()
        if "indic_de" in working.columns:
            working = working[working["indic_de"].isin(monotone_indicators)]
            if working.empty:
                return CheckResult(check_name, "OK", "Aucun indicateur monotone à vérifier")

        group_cols = ["geo", "sex", "time"]
        if "indic_de" in working.columns:
            group_cols.append("indic_de")

        issues = []
        for key, grp in working[working["age"] >= min_age].groupby(group_cols):
            grp_s = grp.sort_values("age")
            vals  = grp_s["values"].values
            ages  = grp_s["age"].values
            for i in range(len(vals) - 1):
                if vals[i] > 0 and vals[i+1] < vals[i] * 0.5:
                    row = dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
                    row.update({
                        "age_from":   int(ages[i]),
                        "age_to":     int(ages[i+1]),
                        "value_from": round(float(vals[i]),   4),
                        "value_to":   round(float(vals[i+1]), 4),
                        "drop_%":     round((vals[i] - vals[i+1]) / vals[i] * 100, 1),
                    })
                    issues.append(row)

        if not issues:
            return CheckResult(check_name, "OK",
                f"Pas d'inversion d'age significative au-dessus de {min_age} ans")

        df_issues = pd.DataFrame(issues)
        return CheckResult(check_name, "WARNING",
            f"{len(df_issues)} inversions d'age (chute >50%) au-dessus de {min_age} ans",
            details=df_issues.sort_values("drop_%", ascending=False),
            n_issues=len(df_issues))

    # =========================================================================
    # Check 6 — Sauts temporels
    # =========================================================================

    def check_temporal_spikes(self, df: pd.DataFrame, dataset_label: str = "data") -> CheckResult:
        """
        Flagge les variations année-sur-année dont le ratio dépasse `spike_threshold`.
        Détecte les erreurs de saisie, les changements de méthodologie non signalés,
        ou les valeurs manquantes remplies par zéro.
        """
        check_name = f"temporal_spikes [{dataset_label}]"

        if not {"geo", "sex", "time", "values"}.issubset(df.columns):
            return CheckResult(check_name, "WARNING",
                "Colonnes geo, sex, time, values requises")

        group_cols = ["geo", "sex"]
        if "age"      in df.columns: group_cols.append("age")
        if "indic_de" in df.columns: group_cols.append("indic_de")

        issues = []
        for key, grp in df.groupby(group_cols):
            grp_s = grp.sort_values("time")
            grp_s = grp_s[grp_s["values"] > 0]
            if len(grp_s) < 2:
                continue
            vals  = grp_s["values"].values
            times = grp_s["time"].values
            for i in range(len(vals) - 1):
                ratio = vals[i+1] / vals[i]
                if ratio > self.spike_threshold or ratio < 1 / self.spike_threshold:
                    row = dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
                    row.update({
                        "year_from":  int(times[i]),
                        "year_to":    int(times[i+1]),
                        "value_from": round(float(vals[i]),   4),
                        "value_to":   round(float(vals[i+1]), 4),
                        "ratio":      round(float(ratio),     3),
                    })
                    issues.append(row)

        if not issues:
            return CheckResult(check_name, "OK",
                f"Pas de saut temporel au-dessus de {self.spike_threshold}x")

        df_issues = pd.DataFrame(issues)
        return CheckResult(check_name, "WARNING",
            f"{len(df_issues)} sauts annee-sur-annee suspects (ratio > {self.spike_threshold}x)",
            details=df_issues.sort_values("ratio", ascending=False),
            n_issues=len(df_issues))

    # =========================================================================
    # Check 7 — Complétude
    # =========================================================================

    def check_completeness(self, df: pd.DataFrame, dataset_label: str = "data") -> CheckResult:
        """
        Vérifie deux choses :
          1. Absence de valeurs nulles dans la colonne `values`.
          2. Absence de trous dans la grille geo x temps : une région présente
             sur 2010-2020 ne doit pas disparaître silencieusement en 2015.
        """
        check_name = f"completeness [{dataset_label}]"

        if not {"geo", "time"}.issubset(df.columns):
            return CheckResult(check_name, "WARNING", "Colonnes geo et time requises")

        n_null = int(df["values"].isna().sum()) if "values" in df.columns else 0

        actual_combos = df[["geo", "time"]].drop_duplicates()
        full_grid = pd.MultiIndex.from_product(
            [df["geo"].unique(), df["time"].unique()], names=["geo", "time"]
        ).to_frame(index=False)
        missing = (
            full_grid
            .merge(actual_combos, on=["geo", "time"], how="left", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns="_merge")
        )

        issues = []
        if n_null        > 0: issues.append(f"{n_null} valeurs nulles")
        if not missing.empty: issues.append(f"{len(missing)} combinaisons geo x temps manquantes")

        if not issues:
            return CheckResult(check_name, "OK", "Pas de probleme de completude")

        return CheckResult(check_name, "WARNING", " | ".join(issues),
            details=missing if not missing.empty else None,
            n_issues=len(missing) + n_null)

    # =========================================================================
    # Check 8 — Bornes génériques (décès / population)
    # =========================================================================

    def check_value_bounds(
        self,
        df: pd.DataFrame,
        dataset_label: str = "data",
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> CheckResult:
        """
        Check générique : vérifie que les valeurs restent dans [lower, upper].
        Pour les décès et la population, seule la borne inférieure >= 0 s'applique.
        """
        check_name = f"value_bounds [{dataset_label}]"

        if "values" not in df.columns:
            return CheckResult(check_name, "WARNING", "Colonne 'values' manquante")

        if lower is None and upper is None:
            lower, upper = (0.0, 1.0) if dataset_label == "mortality" else (0.0, None)

        mask = pd.Series(False, index=df.index)
        if lower is not None: mask |= df["values"] < lower
        if upper is not None: mask |= df["values"] > upper
        bad = df[mask]

        bound_str = f"[{lower}, {upper if upper else '+inf'}]"
        if bad.empty:
            return CheckResult(check_name, "OK",
                f"Toutes les valeurs dans les bornes {bound_str}")

        return CheckResult(check_name, "ERROR",
            f"{len(bad)} valeurs hors bornes {bound_str}",
            details=bad, n_issues=len(bad))

    # =========================================================================
    # Check 9 — Taux implicite D/L vs DEATHRATE observé (cross-dataset)
    # =========================================================================

    def check_implied_mortality_rate(self, max_relative_error: float = 0.10) -> CheckResult:
        """
        Calcule le taux implicite mx = Deces / Population et le compare
        au DEATHRATE observé dans le dataset mortalité.

        Un écart > 10% signale une incohérence entre datasets : soit les données
        ne correspondent pas aux mêmes années/régions, soit il y a un problème
        d'extraction. Nécessite les 3 datasets.
        """
        check_name = "implied_mortality_rate [cross-dataset]"

        required = {"geo", "sex", "age", "time", "values"}
        for label, ds in [("mortality", self.mortality),
                           ("deaths",    self.deaths),
                           ("population", self.population)]:
            if ds is None or not required.issubset(ds.columns):
                return CheckResult(check_name, "WARNING",
                    f"Données ou colonnes manquantes dans {label}")

        if "indic_de" in self.mortality.columns:
            mxt_base = self.mortality[
                (self.mortality["sex"] == "T") &
                (self.mortality["indic_de"] == "DEATHRATE")
            ]
        else:
            mxt_base = self.mortality[self.mortality["sex"] == "T"]

        if mxt_base.empty:
            return CheckResult(check_name, "WARNING",
                "Indicateur DEATHRATE absent — cross-check ignoré")

        mxt = mxt_base[["geo","age","time","values"]].rename(columns={"values": "mx_obs"})
        dxt = (self.deaths[self.deaths["sex"] == "T"]
               [["geo","age","time","values"]].rename(columns={"values": "Dxt"}))
        lxt = (self.population[self.population["sex"] == "T"]
               [["geo","age","time","values"]].rename(columns={"values": "Lxt"}))

        merged = mxt.merge(dxt, on=["geo","age","time"]).merge(lxt, on=["geo","age","time"])
        merged = merged[(merged["Lxt"] > 0) & (merged["mx_obs"] > 0)].copy()
        merged["mx_implied"] = merged["Dxt"] / merged["Lxt"]
        merged["rel_error"]  = ((merged["mx_implied"] - merged["mx_obs"]) / merged["mx_obs"]).abs()

        bad = merged[merged["rel_error"] > max_relative_error]
        if bad.empty:
            return CheckResult(check_name, "OK",
                f"D/L ≈ DEATHRATE observé dans la tolérance de {max_relative_error*100:.0f}%")

        return CheckResult(check_name, "WARNING",
            f"{len(bad)} lignes ou D/L s'ecarte de >{max_relative_error*100:.0f}% du DEATHRATE",
            details=(bad[["geo","age","time","mx_obs","mx_implied","rel_error"]]
                     .sort_values("rel_error", ascending=False)),
            n_issues=len(bad))


# =============================================================================
# Fonction utilitaire
# =============================================================================

def check_eurostat(
    mortality:    Optional[pd.DataFrame] = None,
    deaths:       Optional[pd.DataFrame] = None,
    population:   Optional[pd.DataFrame] = None,
    print_report: bool = True,
    **kwargs
) -> CheckReport:
    """
    Lance tous les checks de cohérence Eurostat en une ligne.

    Paramètres
    ----------
    mortality, deaths, population : DataFrames issus de manager.load()
    print_report : bool — affiche le résumé si True
    **kwargs     : tolerance=0.05, spike_threshold=2.0

    Retourne
    --------
    CheckReport

    Exemples
    --------
    # Lancement standard
    report = check_eurostat(mortality=mxt, deaths=dxt, population=lxt)

    # Voir les problèmes détaillés
    for issue in report.get_issues():
        print(issue)
        if issue.details is not None:
            display(issue.details.head(10))

    # Changer les seuils
    report = check_eurostat(
        mortality=mxt, deaths=dxt, population=lxt,
        tolerance=0.02,       # tolérance 2% au lieu de 5%
        spike_threshold=3.0   # sauts autorisés jusqu'à 3x au lieu de 2x
    )

    # Lancer un seul check
    checker = EurostatChecker(mortality=mxt)
    result  = checker.check_deathrate_gender_pattern(mxt)
    print(result)
    display(result.details)
    """
    checker = EurostatChecker(
        mortality=mortality, deaths=deaths, population=population, **kwargs
    )
    report = checker.run_all()
    if print_report:
        report.summary()
    return report
