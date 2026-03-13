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

from pathlib import Path

BASE_DIR = Path.cwd()
while not (BASE_DIR / "NUTS_files").exists():
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "NUTS_files" / "NUTS_RG_01M_2024_3035.shp"

class EurostatConfig:
    """Centralized configuration for paths and default parameters."""
    
    # Default path for NUTS shapefile
    #DEFAULT_SHAPEFILE_PATH = Path("C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp")
    DEFAULT_SHAPEFILE_PATH = DATA_DIR #or DATA_DIR because it is Path object
    
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


