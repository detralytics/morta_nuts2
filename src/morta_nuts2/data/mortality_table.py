"""
MortalityCalculator - Optimized and Innovative Version
=====================================================

Main improvements ::

- Smart cache to avoid recomputations
- Automatic data validation
- Detailed operation logging
- Support for different output formats
- Integrated visualization methods
- Advanced error handling
- Configurable processing pipeline
- Multi-format export (CSV, Parquet, Excel)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from typing import Optional, Union, List, Dict, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import hashlib
import pickle


from pathlib import Path

# Cherche la racine du projet en remontant jusqu'à trouver NUTS_files/
BASE_DIR = Path.cwd()
while not (BASE_DIR / "NUTS_files").exists():
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "NUTS_files" / "NUTS_RG_01M_2024_3035.shp"
# Path to the NUTS shapefile used for choropleth maps
#SHAPEF_PATH = DATA_DIR



# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class MortalityConfig:
    """Configuration for mortality calculations."""
    age_max: int = 82
    age_anomalies: np.ndarray = field(default_factory=lambda: np.array([11, 22, 33, 44, 55, 66, 77]))
    fill_value: float = 1e-6
    n_jobs: int = -1
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    validate_data: bool = True
    correct_anomalies: bool = True
    
    def __post_init__(self):
        self.age_range = np.arange(0, self.age_max + 1)
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.getcwd(), '.mortality_cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class DataValidator:
    """Data validator with detailed reports."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: List[str],
                          name: str = "DataFrame") -> Dict[str, any]:
        """Validates a DataFrame and returns a report."""
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            report['valid'] = False
            report['errors'].append(f"Missing columns in {name}: {missing_cols}")
        
        # Basic statistics
        report['stats']['n_rows'] = len(df)
        report['stats']['n_cols'] = len(df.columns)
        report['stats']['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
        
        # Check missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            report['warnings'].append(f"Missing values detected: {null_counts[null_counts > 0].to_dict()}")
        
        # Check duplicates
        if df.duplicated().any():
            n_duplicates = df.duplicated().sum()
            report['warnings'].append(f"{n_duplicates} duplicate rows detected")
        
        return report
    
    @staticmethod
    def validate_mortality_data(mxt_raw: pd.DataFrame, 
                               Lxt_raw: pd.DataFrame, 
                               Dxt_raw: pd.DataFrame) -> Dict[str, any]:
        """Specific validation for mortality data."""
        report = {'overall_valid': True, 'datasets': {}}
        
        # Validate each dataset
        report['datasets']['mxt'] = DataValidator.validate_dataframe(
            mxt_raw, ['geo', 'sex', 'indic_de', 'age', 'time', 'values'], 'mxt_raw'
        )
        report['datasets']['Lxt'] = DataValidator.validate_dataframe(
            Lxt_raw, ['geo', 'sex', 'age', 'time', 'values'], 'Lxt_raw'
        )
        report['datasets']['Dxt'] = DataValidator.validate_dataframe(
            Dxt_raw, ['geo', 'sex', 'age', 'time', 'values'], 'Dxt_raw'
        )
        
        # Check consistency between datasets
        for key, val in report['datasets'].items():
            if not val['valid']:
                report['overall_valid'] = False
        
        return report


class CacheManager:
    """Cache manager to optimize repeated calculations."""
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generates a unique cache key."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[any]:
        """Retrieves a value from the cache."""
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Error reading from cache: {e}")
        return None
    
    def set(self, key: str, value: any):
        """Stores a value in the cache."""
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logging.warning(f"Error writing to cache: {e}")
    
    def clear(self):
        """Clears the cache."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.info(f"{func.__name__} executed in {duration:.2f} seconds")
        return result
    return wrapper


class MortalityCalculator:
    """
    Optimized mortality calculator with advanced features.
    
    Innovative features:
    - Smart cache to avoid recomputations
    - Automatic data validation
    - Multi-format export support
    - Configurable pipeline
    - Detailed logging
    - Robust error handling
    """
    
    # Default path for the NUTS shapefile
    #DEFAULT_SHAPEFILE = "C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp"
    DEFAULT_SHAPEFILE=DATA_DIR
    FRANCE_OUTREMER = ['FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5']
    
    def __init__(self, 
                 shapefile_path: Optional[str] = None,
                 config: Optional[MortalityConfig] = None,
                 auto_load_regions: bool = True):
        """
        Initializes the mortality calculator.
        
        Parameters
        ----------
        shapefile_path : str, optional
            Path to the NUTS shapefile.
            If None, uses the default path.
        config : MortalityConfig, optional
            Custom configuration
        auto_load_regions : bool, default=True
            Automatically loads regions from the shapefile
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or MortalityConfig()
        
        # Cache manager
        self.cache_manager = CacheManager(
            self.config.cache_dir, 
            self.config.enable_cache
        )
        
        # Data validator
        self.validator = DataValidator()
        
        # Regions
        self.regions = []
        self.shapefile_path = shapefile_path or self.DEFAULT_SHAPEFILE
        
        if auto_load_regions:
            self._load_regions()
        
        # Execution statistics
        self.execution_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0
        }
        
        self.logger.info(f"MortalityCalculator initialized with {len(self.regions)} regions")
    
    def _load_regions(self):
        """Loads regions from the shapefile."""
        try:
            if os.path.exists(self.shapefile_path):
                self.logger.info(f"Loading shapefile: {self.shapefile_path}")
                shapef = gpd.read_file(self.shapefile_path)
                self.regions = shapef["NUTS_ID"].tolist()
                self.regions = [r for r in self.regions if r not in self.FRANCE_OUTREMER]
                self.logger.info(f"{len(self.regions)} regions loaded")
            else:
                self.logger.warning(f"Shapefile not found: {self.shapefile_path}")
        except Exception as e:
            self.logger.error(f"Error loading shapefile: {e}")
    
    def set_shapefile(self, shapefile_path: str):
        """
        Changes the shapefile and reloads regions.
        
        Parameters
        ----------
        shapefile_path : str
            New path to the shapefile
        """
        self.shapefile_path = shapefile_path
        self._load_regions()
    
    def get_regions(self, country: str = "FR") -> List[str]:
        """
        Gets the list of regions filtered by country.
        
        Parameters
        ----------
        country : str, default="FR"
            Country code (FR to filter overseas territories)
        
        Returns
        -------
        List[str]
            List of region codes
        """
        if country == "FR":
            return [r for r in self.regions if r not in self.FRANCE_OUTREMER]
        return self.regions
    
    @timing_decorator
    def validate_input_data(self, 
                           mxt_raw: pd.DataFrame,
                           Lxt_raw: pd.DataFrame,
                           Dxt_raw: pd.DataFrame) -> Dict[str, any]:
        """
        Validates the input data.
        
        Returns
        -------
        Dict
            Validation report
        """
        if not self.config.validate_data:
            return {'overall_valid': True, 'message': 'Validation disabled'}
        
        self.logger.info("Validating input data...")
        report = self.validator.validate_mortality_data(mxt_raw, Lxt_raw, Dxt_raw)
        
        if not report['overall_valid']:
            self.logger.error("Validation failed!")
            for dataset, data in report['datasets'].items():
                if data['errors']:
                    for error in data['errors']:
                        self.logger.error(f"  {dataset}: {error}")
        else:
            self.logger.info("Validation successful")
        
        return report
    
    @staticmethod
    def _build_pivot_mu(mxt_raw: pd.DataFrame, 
                       gender: str, 
                       common_ages: List[int],
                       fill_value: float = 1e-6) -> Dict[str, pd.DataFrame]:
        """
        Builds pivots for mortality rates.
        Optimized with full vectorization.
        """
        sub = mxt_raw[
            (mxt_raw["sex"] == gender) &
            (mxt_raw["indic_de"] == "DEATHRATE") &
            (mxt_raw["age"].isin(common_ages))
        ].copy()
        
        if len(sub) == 0:
            return {}
        
        pivot = pd.pivot_table(
            sub,
            values="values",
            index=["geo", "age"],
            columns="time",
            aggfunc="sum",
            fill_value=fill_value,
            observed=True,
        )
        
        pivot.columns = pivot.columns.astype(int)
        pivot.index = pivot.index.set_levels(
            pivot.index.levels[1].astype(int), level=1
        )
        
        return {
            reg: grp.droplevel(0).sort_index()
            for reg, grp in pivot.groupby(level="geo")
        }
    
    @staticmethod
    def _build_pivot_L(Lxt_raw: pd.DataFrame, 
                      gender: str, 
                      common_ages: List[int],
                      fill_value: float = 1e-6) -> Dict[str, pd.DataFrame]:
        """
        Builds pivots for exposure.
        Optimized with full vectorization.
        """
        sub = Lxt_raw[
            (Lxt_raw["sex"] == gender) &
            (Lxt_raw["age"].isin(common_ages))
        ].copy()
        
        if len(sub) == 0:
            return {}
        
        pivot = pd.pivot_table(
            sub,
            values="values",
            index=["geo", "age"],
            columns="time",
            aggfunc="sum",
            fill_value=fill_value,
            observed=True,
        )
        
        pivot.columns = pivot.columns.astype(int)
        pivot.index = pivot.index.set_levels(
            pivot.index.levels[1].astype(int), level=1
        )
        
        return {
            reg: grp.droplevel(0).sort_index()
            for reg, grp in pivot.groupby(level="geo")
        }
    
    def _correct_age_anomalies(self, Mu: np.ndarray, age_arr: np.ndarray) -> np.ndarray:
        """
        Corrects age anomalies by interpolation.
        
        Optimized with numpy vectorization.
        """
        if not self.config.correct_anomalies:
            return Mu
        
        Mu_corrected = Mu.copy()
        
        for a in self.config.age_anomalies:
            idx = np.searchsorted(age_arr, a)
            idx_p = np.searchsorted(age_arr, a + 1)
            idx_m = np.searchsorted(age_arr, a - 1)
            
            if (idx < len(age_arr) and age_arr[idx] == a and
                idx_p < len(age_arr) and age_arr[idx_p] == a + 1 and
                idx_m >= 0 and age_arr[idx_m] == a - 1):
                Mu_corrected[idx, :] = (Mu[idx_p, :] + Mu[idx_m, :]) / 2
        
        return Mu_corrected
    
    def _compute_exposure(self, L: np.ndarray) -> np.ndarray:
        """
        Computes exposure (Extg method).
        
        Optimized vectorized formula.
        """
        E = np.empty_like(L)
        E[0, :] = L[0, :]
        E[1:, :] = (L[1:, :] + L[:-1, :]) / 2
        E[-1, :] = L[-1, :]
        return E
    
    def _process_region_optimized(self, 
                                  reg: str, 
                                  mu_df: pd.DataFrame, 
                                  L_df: pd.DataFrame,
                                  aggregate_age: bool = True) -> Optional[Tuple]:
        """
        Processes a region with an optimized pipeline.
        
        Parameters
        ----------
        reg : str
            Region code
        mu_df : pd.DataFrame
            Mortality rates DataFrame
        L_df : pd.DataFrame
            Exposure DataFrame
        aggregate_age : bool
            If True, aggregates by age
        
        Returns
        -------
        Tuple or None
            Calculation results
        """
        try:
            # Year alignment
            common_years = mu_df.columns.intersection(L_df.columns)
            if len(common_years) == 0:
                return None
            
            # Age alignment
            common_ages = mu_df.index.intersection(L_df.index)
            if len(common_ages) == 0:
                return None
            
            # Matrix extraction
            Mu = mu_df.loc[common_ages, common_years].values.copy()
            L = L_df.loc[common_ages, common_years].values
            age_arr = np.array(common_ages)
            
            # Anomaly correction
            Mu = self._correct_age_anomalies(Mu, age_arr)
            
            # Max age truncation
            mask = age_arr <= self.config.age_max
            age_arr = age_arr[mask]
            Mu = Mu[mask, :]
            L = L[mask, :]
            
            if len(age_arr) == 0:
                return None
            
            # Exposure calculation
            E = self._compute_exposure(L)
            
            # Deaths calculation
            D = Mu * E
            
            if aggregate_age:
                # Aggregate by age
                D_t = D.sum(axis=0)
                E_t = E.sum(axis=0)
                return reg, common_years.values, D_t, E_t
            else:
                # Without aggregation
                return reg, common_years.values, age_arr, D, E
        
        except Exception as e:
            self.logger.error(f"Error processing region {reg}: {e}")
            return None
    
    @timing_decorator
    def _parallel_processing(self,
                            regions: List[str],
                            mu_by_region: Dict[str, pd.DataFrame],
                            L_by_region: Dict[str, pd.DataFrame],
                            aggregate_age: bool = True) -> List[Tuple]:
        """
        Parallel processing of regions with optimal thread management.
        """
        n_jobs = self.config.n_jobs
        if n_jobs == -1:
            n_jobs = min(os.cpu_count() or 1, len(regions))
        
        valid_regions = [r for r in regions if r in mu_by_region and r in L_by_region]
        
        self.logger.info(f"Processing {len(valid_regions)} regions with {n_jobs} threads")
        
        raw_results = []
        
        if n_jobs == 1:
            # Sequential mode
            for reg in valid_regions:
                result = self._process_region_optimized(
                    reg, mu_by_region[reg], L_by_region[reg], aggregate_age
                )
                if result is not None:
                    raw_results.append(result)
        else:
            # Parallel mode with progress tracking
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._process_region_optimized,
                        reg,
                        mu_by_region[reg],
                        L_by_region[reg],
                        aggregate_age
                    ): reg for reg in valid_regions
                }
                
                completed = 0
                total = len(futures)
                
                for future in as_completed(futures):
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        self.logger.info(f"Progress: {completed}/{total} regions processed")
                    
                    result = future.result()
                    if result is not None:
                        raw_results.append(result)
        
        return raw_results
    
    def _build_final_dataframe(self, 
                               raw_results: List[Tuple],
                               aggregate_age: bool = True) -> pd.DataFrame:
        """
        Builds the final DataFrame in an optimized way.
        """
        if aggregate_age:
            all_regions = np.repeat(
                [r[0] for r in raw_results], 
                [len(r[1]) for r in raw_results]
            )
            all_years = np.concatenate([r[1] for r in raw_results])
            all_deaths = np.concatenate([r[2] for r in raw_results])
            all_exposure = np.concatenate([r[3] for r in raw_results])
            
            return pd.DataFrame({
                "region": all_regions,
                "year": all_years,
                "deaths": all_deaths,
                "exposure": all_exposure,
                "mortality_rate": all_deaths / all_exposure,
            })
        else:
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
            
            all_regions = np.concatenate(regions_list)
            all_years = np.concatenate(years_list)
            all_ages = np.concatenate(ages_list)
            all_deaths = np.concatenate(deaths_list)
            all_exposure = np.concatenate(exposure_list)
            
            return pd.DataFrame({
                "region": all_regions,
                "year": all_years,
                "age": all_ages,
                "deaths": all_deaths,
                "exposure": all_exposure,
                "mortality_rate": all_deaths / all_exposure,
            })
    
    @timing_decorator
    def calculate_mortality(self,
                           mxt_raw: pd.DataFrame,
                           Lxt_raw: pd.DataFrame,
                           Dxt_raw: pd.DataFrame,
                           regions: Optional[List[str]] = None,
                           gender: str = "T",
                           country: str = "FR",
                           aggregate_age: bool = False,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Unified main method to calculate mortality.
        
        Parameters
        ----------
        mxt_raw : pd.DataFrame
            Mortality rate data
        Lxt_raw : pd.DataFrame
            Exposure data
        Dxt_raw : pd.DataFrame
            Deaths data
        regions : List[str], optional
            List of regions to process. If None, uses self.regions
        gender : str, default="T"
            Gender ("T", "M", "F")
        country : str, default="FR"
            Country code
        aggregate_age : bool, default=True
            If True, aggregates results by age
        use_cache : bool, default=True
            Uses cache if available
        
        Returns
        -------
        pd.DataFrame
            Mortality calculation results
        """
        start_time = datetime.now()
        
        # Validation
        if self.config.validate_data:
            validation_report = self.validate_input_data(mxt_raw, Lxt_raw, Dxt_raw)
            if not validation_report['overall_valid']:
                raise ValueError("Data validation failed")
        
        # Determine regions
        if regions is None:
            regions = self.get_regions(country)
        elif country == "FR":
            regions = [r for r in regions if r not in self.FRANCE_OUTREMER]
        
        # Check cache
        cache_key = None
        if use_cache and self.config.enable_cache:
            cache_key = self.cache_manager._get_cache_key(
                len(mxt_raw), len(Lxt_raw), len(Dxt_raw),
                tuple(regions), gender, country, aggregate_age
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("Result retrieved from cache")
                self.execution_stats['cache_hits'] += 1
                return cached_result
            self.execution_stats['cache_misses'] += 1
        
        # Age harmonization
        mxt_g = mxt_raw[mxt_raw["sex"] == gender]
        Lxt_g = Lxt_raw[Lxt_raw["sex"] == gender]
        Dxt_g = Dxt_raw[Dxt_raw["sex"] == gender]
        
        common_ages = sorted(
            set(mxt_g["age"].unique()) &
            set(Lxt_g["age"].unique()) &
            set(Dxt_g["age"].unique())
        )
        
        if not common_ages:
            raise ValueError("No common ages after harmonization")
        
        self.logger.info(f"Harmonized ages: {len(common_ages)} common ages")
        
        # Global parallel pre-pivot
        self.logger.info("Building pivots...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_mu = pool.submit(
                self._build_pivot_mu, mxt_raw, gender, common_ages, self.config.fill_value
            )
            fut_L = pool.submit(
                self._build_pivot_L, Lxt_raw, gender, common_ages, self.config.fill_value
            )
            mu_by_region = fut_mu.result()
            L_by_region = fut_L.result()
        
        # Parallel region processing
        raw_results = self._parallel_processing(
            regions, mu_by_region, L_by_region, aggregate_age
        )
        
        if not raw_results:
            raise ValueError("No usable region after age/year alignment")
        
        # Build final DataFrame
        self.logger.info("Building final DataFrame...")
        result = self._build_final_dataframe(raw_results, aggregate_age)
        
        # Cache storage
        if use_cache and cache_key:
            self.cache_manager.set(cache_key, result)
        
        # Statistics
        duration = (datetime.now() - start_time).total_seconds()
        self.execution_stats['total_calculations'] += 1
        self.execution_stats['total_processing_time'] += duration
        
        self.logger.info(f"Calculation complete: {len(result)} rows generated")
        
        return result
    
    def mortality_by_region(self, *args, **kwargs) -> pd.DataFrame:
        """
        Calculates mortality aggregated by region and year.
        Alias for calculate_mortality with aggregate_age=True.
        """
        kwargs['aggregate_age'] = True
        return self.calculate_mortality(*args, **kwargs)
    
    def mortality_by_region_by_age(self, *args, **kwargs) -> pd.DataFrame:
        """
        Calculates mortality by region, year and age.
        Alias for calculate_mortality with aggregate_age=False.
        """
        kwargs['aggregate_age'] = False
        return self.calculate_mortality(*args, **kwargs)
    
    def export_results(self, 
                      df: pd.DataFrame,
                      output_path: str,
                      format: Literal['csv', 'parquet', 'excel', 'feather'] = 'csv',
                      **kwargs):
        """
        Exports results in different formats.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to export
        output_path : str
            Output path
        format : str
            Export format ('csv', 'parquet', 'excel', 'feather')
        **kwargs
            Additional arguments for the export function
        """
        self.logger.info(f"Exporting results in {format} format...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False, **kwargs)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False, **kwargs)
        elif format == 'excel':
            df.to_excel(output_path, index=False, **kwargs)
        elif format == 'feather':
            df.to_feather(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Export successful: {output_path}")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Returns execution statistics.
        
        Returns
        -------
        Dict
            Complete statistics
        """
        stats = self.execution_stats.copy()
        
        if stats['total_calculations'] > 0:
            stats['avg_processing_time'] = (
                stats['total_processing_time'] / stats['total_calculations']
            )
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
                if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
            )
        
        return stats
    
    def clear_cache(self):
        """Clears the cache."""
        self.cache_manager.clear()
        self.logger.info("Cache cleared")
    
    def summary(self):
        """Displays a summary of the configuration and statistics."""
        print("="*60)
        print("MortalityCalculator - Summary")
        print("="*60)
        print(f"Shapefile: {self.shapefile_path}")
        print(f"Regions loaded: {len(self.regions)}")
        print(f"Maximum age: {self.config.age_max}")
        print(f"Anomaly correction: {self.config.correct_anomalies}")
        print(f"Cache enabled: {self.config.enable_cache}")
        print(f"Threads: {self.config.n_jobs if self.config.n_jobs != -1 else 'auto'}")
        print("\nExecution statistics:")
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*60)


# Compatibility functions with the old API
def mortality_by_region(*args, **kwargs):
    """Compatibility function - creates an instance and calculates."""
    warnings.warn(
        "Use of deprecated standalone function. "
        "Prefer using MortalityCalculator.calculate_mortality()",
        DeprecationWarning
    )
    calculator = MortalityCalculator()
    return calculator.mortality_by_region(*args, **kwargs)


def mortality_by_region_by_age(*args, **kwargs):
    """Compatibility function - creates an instance and calculates."""
    warnings.warn(
        "Use of deprecated standalone function. "
        "Prefer using MortalityCalculator.calculate_mortality()",
        DeprecationWarning
    )
    calculator = MortalityCalculator()
    return calculator.mortality_by_region_by_age(*args, **kwargs)



def build_input_from_dataframe(df):
    """
    Transforms a long DataFrame into 3D matrices compatible with LCp_fit.
    
    Parameters
    ----------
    df : DataFrame with columns
         ['region', 'year', 'age', 'deaths', 'exposure','mortality_rate']
    
    Returns
    --------
    Muxtg: (nb_ages, nb_years, nb_regions)
    Dxtg : (nb_ages, nb_years, nb_regions)
    Extg : (nb_ages, nb_years, nb_regions)
    xv   : sorted age vector
    tv   : sorted year vector
    regions : list of regions
    """
    
    # Sort for safety
    df = df.sort_values(["age", "year", "region"]).copy()
    
    xv = np.sort(df["age"].unique())
    tv = np.sort(df["year"].unique())
    regions = np.sort(df["region"].unique())
    
    nb_ages = len(xv)
    nb_years = len(tv)
    nb_regions = len(regions)
    
    # Index mapping
    age_idx = {a:i for i,a in enumerate(xv)}
    year_idx = {y:i for i,y in enumerate(tv)}
    reg_idx = {r:i for i,r in enumerate(regions)}
    
    # Allocation
    Dxtg = np.zeros((nb_ages, nb_years, nb_regions))
    Extg = np.zeros_like(Dxtg)
    Muxtg = np.zeros_like(Dxtg)
    
    # Vectorization without triple loop
    Dxtg[
        df.age.map(age_idx),
        df.year.map(year_idx),
        df.region.map(reg_idx)
    ] = df.deaths.values
    
    Extg[
        df.age.map(age_idx),
        df.year.map(year_idx),
        df.region.map(reg_idx)
    ] = df.exposure.values

    Muxtg[
        df.age.map(age_idx),
        df.year.map(year_idx),
        df.region.map(reg_idx)
    ] = df.mortality_rate.values
    
    # Numerical safety
    Extg = np.maximum(Extg, 1e-12)
    Dxtg = np.maximum(Dxtg, 0.0)
    
    return Muxtg, Dxtg, Extg, xv, tv, regions
