# =============================================================================
# Eurostat Data Manager - Smart avec Shapefile Auto-Load
# =============================================================================

import pandas as pd
from eurostatapiclient import EurostatAPIClient
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Literal, Union


class EurostatConfig:
    """Configuration centralisée pour les chemins et paramètres par défaut."""
    
    # Chemin par défaut du shapefile NUTS
    DEFAULT_SHAPEFILE_PATH = Path("C:/Users/Idrissa Belem/Documents/GitHub/test_projet/NUTS_files/NUTS_RG_01M_2024_3035.shp")
    
    # Chemin par défaut des données
    DEFAULT_DATA_PATH = Path("../data")
    
    # Datasets disponibles
    DATASETS = {
        "mortality": "demo_r_mlife",
        "deaths": "demo_r_magec",
        "population": "demo_r_d2jan"
    }
    
    # Régions à exclure par pays
    EXCLUDE_REGIONS = {
        "FR": ["FRY1", "FRY2", "FRY3", "FRY4", "FRY5"],  # DOM-TOM
        "PT": ["PT20", "PT30"],                           # Açores, Madère
        "ES": ["ES63", "ES64", "ES70"],                   # Ceuta, Melilla, Canaries
        "NO": ["NO0B"],                                   # Svalbard
    }
    
    @classmethod
    def set_default_shapefile(cls, path: Union[str, Path]) -> None:
        """
        Modifie le chemin par défaut du shapefile globalement.
        
        Examples
        --------
        >>> EurostatConfig.set_default_shapefile("D:/data/NUTS_2024.shp")
        """
        cls.DEFAULT_SHAPEFILE_PATH = Path(path)
    
    @classmethod
    def set_default_data_path(cls, path: Union[str, Path]) -> None:
        """
        Modifie le chemin par défaut des données globalement.
        
        Examples
        --------
        >>> EurostatConfig.set_default_data_path("D:/eurostat_cache")
        """
        cls.DEFAULT_DATA_PATH = Path(path)


class Eurostat_data:
    """
    Gestionnaire unique pour toutes les opérations Eurostat.
    
    Le shapefile NUTS est chargé automatiquement depuis le chemin par défaut.
    Aucun besoin de manipuler 'shapef' dans votre code !
    
    Examples
    --------
    # Utilisation standard (shapefile auto-chargé)
    >>> manager = EurostatManager()
    >>> mortality = manager.load("mortality", "FR")
    
    # Avec un shapefile personnalisé
    >>> manager = EurostatManager(shapefile_path="D:/custom/nuts.shp")
    
    # Ou passer directement un GeoDataFrame
    >>> custom_shapef = gpd.read_file("path/to/shapefile.shp")
    >>> manager = EurostatManager(shapefile=custom_shapef)
    
    # Changer le défaut pour toute l'application
    >>> EurostatConfig.set_default_shapefile("D:/data/NUTS_2024.shp")
    >>> manager = EurostatManager()  # Utilisera le nouveau chemin
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
        Initialise le gestionnaire Eurostat.
        
        Parameters
        ----------
        shapefile : GeoDataFrame, optional
            Shapefile NUTS déjà chargé (priorité si fourni)
        shapefile_path : str or Path, optional
            Chemin personnalisé du shapefile (utilise le défaut si None)
        data_path : str or Path, optional
            Répertoire de stockage des données (utilise le défaut si None)
        language : str
            Langue des métadonnées Eurostat
        nuts_level : int
            Niveau NUTS par défaut (1, 2 ou 3)
        auto_load_shapefile : bool
            Si True, charge automatiquement le shapefile au premier usage
        """
        self._shapefile = shapefile
        self._shapefile_path = Path(shapefile_path) if shapefile_path else EurostatConfig.DEFAULT_SHAPEFILE_PATH
        self._auto_load = auto_load_shapefile
        self._shapefile_loaded = shapefile is not None
        
        self.data_path = Path(data_path) if data_path else EurostatConfig.DEFAULT_DATA_PATH
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.nuts_level = nuts_level
        
        # Client API Eurostat
        self.client = EurostatAPIClient("1.0", "json", language)
        
        # Cache des régions pour éviter de recalculer
        self._regions_cache: Dict[Tuple[str, int, bool], List[str]] = {}
    
    @property
    def shapefile(self) -> gpd.GeoDataFrame:
        """
        Accès au shapefile avec auto-chargement si nécessaire.
        
        L'utilisateur n'a jamais besoin d'appeler cette propriété directement.
        """
        if self._shapefile is None and self._auto_load:
            self._load_shapefile()
        
        if self._shapefile is None:
            raise ValueError(
                f"Shapefile non chargé. Chemin utilisé: {self._shapefile_path}\n"
                f"Vérifiez que le fichier existe ou définissez un autre chemin:\n"
                f"  EurostatConfig.set_default_shapefile('path/to/shapefile.shp')\n"
                f"  ou manager = EurostatManager(shapefile_path='path/to/shapefile.shp')"
            )
        
        return self._shapefile
    
    def _load_shapefile(self) -> None:
        """Charge le shapefile depuis le chemin configuré."""
        if not self._shapefile_path.exists():
            print(f"⚠️  Shapefile introuvable: {self._shapefile_path}")
            print(f"💡 Définissez le bon chemin avec:")
            print(f"   EurostatConfig.set_default_shapefile('votre/chemin.shp')")
            return
        
        print(f"📂 Chargement du shapefile NUTS... {self._shapefile_path.name}")
        self._shapefile = gpd.read_file(self._shapefile_path)
        self._shapefile_loaded = True
        print(f"✅ Shapefile chargé ({len(self._shapefile)} entités)")
    
    def set_shapefile(
        self,
        shapefile: Optional[gpd.GeoDataFrame] = None,
        shapefile_path: Optional[Union[str, Path]] = None
    ) -> 'Eurostat_data':
        """
        Définit un nouveau shapefile (chainable).
        
        Parameters
        ----------
        shapefile : GeoDataFrame, optional
            Shapefile déjà chargé
        shapefile_path : str or Path, optional
            Chemin vers un nouveau shapefile à charger
        
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
    # Méthodes principales - API simplifiée
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
        Charge un dataset Eurostat de manière intelligente.
        
        Parameters
        ----------
        dataset_type : str
            Type de dataset: "mortality", "deaths" ou "population"
        country : str
            Code pays ISO 2 lettres
        nuts_level : int, optional
            Niveau NUTS (utilise celui par défaut si None)
        download : bool
            Force le téléchargement même si le cache existe
        exclude_outremer : bool
            Exclut les régions non continentales
        
        Returns
        -------
        DataFrame nettoyé et prêt à l'emploi
        
        Examples
        --------
        >>> manager = EurostatManager()  # Shapefile auto-chargé !
        >>> mortality = manager.load("mortality", "FR")
        >>> deaths = manager.load("deaths", "BE", download=True)
        """
        nuts_level = nuts_level or self.nuts_level
        
        # Récupération des régions (avec cache)
        regions = self.get_regions(country, nuts_level, exclude_outremer)
        
        # Code du dataset
        dataset_code = EurostatConfig.DATASETS[dataset_type]
        
        # Nom de fichier avec type explicite
        filename_map = {
            "mortality": "mxt_raw",
            "deaths": "Dxt_raw",
            "population": "Lxt_raw"
        }
        filename = filename_map[dataset_type]
        
        # Chargement avec cache
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
        Charge tous les datasets pour un pays.
        
        Returns
        -------
        Dict avec clés: "mortality", "deaths", "population"
        
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
        Crée un tableau croisé âge × année pour une région donnée.
        
        Parameters
        ----------
        data : DataFrame
            Données sources (issues de load())
        region : str
            Code région NUTS
        gender : str
            Genre ("M", "F", "T")
        indicator : str
            Code indicateur Eurostat
        
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
        # Filtrage
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
        
        # Tri par âge croissant
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
        Récupère la liste des codes NUTS pour un pays (avec cache).
        
        Parameters
        ----------
        country : str
            Code pays ISO 2 lettres
        nuts_level : int, optional
            Niveau NUTS (utilise celui par défaut si None)
        exclude_outremer : bool
            Exclut les régions non continentales
        
        Returns
        -------
        Liste des codes NUTS
        
        Examples
        --------
        >>> regions_fr = manager.get_regions("FR", nuts_level=2)
        >>> regions_be = manager.get_regions("BE")
        """
        nuts_level = nuts_level or self.nuts_level
        
        # Clé de cache
        cache_key = (country, nuts_level, exclude_outremer)
        
        # Vérification du cache
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
        Filtre le shapefile NUTS par pays et niveau.
        
        Parameters
        ----------
        country : str
            Code pays ISO 2 lettres
        nuts_level : int, optional
            Niveau NUTS (utilise celui par défaut si None)
        exclude_outremer : bool
            Exclut les régions non continentales
        
        Returns
        -------
        GeoDataFrame filtré et trié
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
    # Méthodes utilitaires
    # =========================================================================
    
    @staticmethod
    def parse_age(label) -> Optional[int]:
        """
        Parse un label d'âge Eurostat en valeur numérique.
        
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
        return None
    
    def add_exclusion(self, country: str, regions: List[str]) -> 'Eurostat_data':
        """
        Ajoute des régions à exclure pour un pays (chainable).
        
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
        """Liste les datasets disponibles."""
        return list(EurostatConfig.DATASETS.keys())
    
    def cache_info(self, country: Optional[str] = None) -> Union[Dict[str, bool], Dict[str, Dict[str, bool]]]:
        """
        Vérifie quels datasets sont en cache.
        
        Parameters
        ----------
        country : str, optional
            Si None, vérifie tous les pays détectés
        
        Returns
        -------
        Dict avec status de chaque dataset
        
        Examples
        --------
        >>> manager.cache_info("FR")
        {'mortality': True, 'deaths': False, 'population': True}
        
        >>> manager.cache_info()  # Tous les pays
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
            # Auto-détection des pays en cache
            countries = set()
            for file in self.data_path.glob("*_*.csv"):
                country_code = file.stem.split("_")[0]
                if len(country_code) == 2:  # Code ISO
                    countries.add(country_code)
            
            return {
                c: self.cache_info(c) for c in sorted(countries)
            }
    
    def clear_cache(self, country: Optional[str] = None, dataset: Optional[str] = None) -> int:
        """
        Supprime les fichiers en cache.
        
        Parameters
        ----------
        country : str, optional
            Code pays (si None, tous les pays)
        dataset : str, optional
            Type de dataset (si None, tous les datasets)
        
        Returns
        -------
        Nombre de fichiers supprimés
        
        Examples
        --------
        >>> manager.clear_cache("FR")  # Supprime tout pour la France
        >>> manager.clear_cache("FR", "mortality")  # Supprime juste mortality FR
        >>> manager.clear_cache()  # Supprime tout le cache
        """
        count = 0
        
        if country and dataset:
            # Suppression ciblée
            filename_map = {"mortality": "mxt_raw", "deaths": "Dxt_raw", "population": "Lxt_raw"}
            filepath = self.data_path / f"{country}_{filename_map[dataset]}.csv"
            if filepath.exists():
                filepath.unlink()
                count = 1
                print(f"🗑️  [Cache] Supprimé: {filepath.name}")
        elif country:
            # Suppression d'un pays
            for file in self.data_path.glob(f"{country}_*.csv"):
                file.unlink()
                count += 1
            print(f"🗑️  [Cache] Supprimé: {count} fichiers pour {country}")
        else:
            # Suppression totale
            for file in self.data_path.glob("*_*.csv"):
                file.unlink()
                count += 1
            if count > 0:
                print(f"🗑️  [Cache] Supprimé: {count} fichiers")
        
        return count
    
    def stats(self) -> Dict[str, any]:
        """
        Statistiques sur le manager.
        
        Returns
        -------
        Dict avec diverses stats
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
        """Représentation lisible du manager."""
        stats = self.stats()
        shapefile_status = "✓ loaded" if stats['shapefile_loaded'] else f"⏳ will load from {Path(stats['shapefile_path']).name}"
        
        return (
            f"EurostatManager(\n"
            f"  shapefile={shapefile_status},\n"
            f"  cache={stats['cached_files']} fichiers ({stats['cache_size_mb']:.1f} MB),\n"
            f"  pays={stats['cached_countries']},\n"
            f"  data='{stats['data_path']}'\n"
            f")"
        )
    
    # =========================================================================
    # Méthodes privées (internes)
    # =========================================================================
    
    def _load_and_cache(
        self,
        dataset_code: str,
        regions: List[str],
        filename: str,
        country: str,
        download: bool
    ) -> pd.DataFrame:
        """Charge un dataset avec gestion du cache."""
        filepath = self.data_path / f"{country}_{filename}.csv"
        
        # Décision de téléchargement
        should_download = download or not filepath.exists()
        
        if should_download:
            data = self._download_dataset(dataset_code, regions)
            data.to_csv(filepath, index=False)
            print(f"⬇️  [Eurostat] Téléchargé → {filepath.name}")
        else:
            data = pd.read_csv(filepath)
            print(f"💾 [Eurostat] Cache → {filepath.name}")
        
        # Nettoyage automatique
        return self._clean_data(data)
    
    def _download_dataset(
        self,
        dataset_code: str,
        regions: List[str]
    ) -> pd.DataFrame:
        """Télécharge les données pour toutes les régions."""
        print(f"🌐 [Eurostat] Téléchargement en cours... ({len(regions)} régions)")
        
        frames = []
        for i, region in enumerate(regions, 1):
            if i % 5 == 0:  # Feedback tous les 5 régions
                print(f"  → Progression: {i}/{len(regions)}")
            
            response = self.client.get_dataset(dataset_code, params={"geo": region})
            df = response.to_dataframe()
            frames.append(df)
        
        return pd.concat(frames, ignore_index=True)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données (âges invalides, conversion types)."""
        data = data.copy()
        
        # Nettoyage des âges
        if "age" in data.columns:
            data = data[~data["age"].isin(["TOTAL", "UNK", "Y_OPEN"])]
            data["age"] = data["age"].map(self.parse_age)
            data = data.dropna(subset=["age"])
            data["age"] = data["age"].astype(int)
        
        # Conversion des années
        if "time" in data.columns:
            data["time"] = data["time"].astype(int)
        
        return data.reset_index(drop=True)


# =============================================================================
# Fonctions de compatibilité (wrapper simple)
# =============================================================================

def load_mxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Compatibilité: charge les taux de mortalité."""
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("mortality", country, download=download)


def load_dxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Compatibilité: charge les décès."""
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("deaths", country, download=download)


def load_lxt_raw(shapef=None, country="FR", nuts_level=2, data_path="../data", download=False):
    """Compatibilité: charge les populations."""
    manager = Eurostat_data(shapefile=shapef, data_path=data_path, nuts_level=nuts_level)
    return manager.load("population", country, download=download)


def age_year_pivot_table(data_raw, region, gender, indicator):
    """Compatibilité: crée un pivot âge × année."""
    manager = Eurostat_data(auto_load_shapefile=False)  # Pas besoin de shapefile pour pivot
    return manager.pivot_age_year(data_raw, region, gender, indicator)


def filter_shapefile(shapef=None, country="FR", nuts_level=2, exclude_outremer=True):
    """Compatibilité: filtre le shapefile."""
    manager = Eurostat_data(shapefile=shapef, nuts_level=nuts_level)
    return manager.filter_shapefile(country, exclude_outremer=exclude_outremer)


def parse_age(label):
    """Compatibilité: parse un label d'âge."""
    return Eurostat_data.parse_age(label)

