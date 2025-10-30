import logging
from pathlib import Path
from typing import Literal, Callable

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

GeometryColumn = Literal["geometry_osm", "geometry_gadm", "geometry_wiki"]
BenchmarkGeomType = Literal["GAUL", "GDIS"]


def _check_file_path(
        file_path: str | Path,
        allowed_suffixes: tuple[str, ...] = (".gpkg",)
) -> Path:
    """Check if file exists and has a valid extension."""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext not in allowed_suffixes:
        raise ValueError(
            f"Unsupported file extension: {ext}. Allowed: {allowed_suffixes}")
    if not p.exists():
        raise ValueError(f"File {p} does not exist.")
    return p


def check_geometries(geoseries: gpd.GeoSeries) -> None:
    """Check if geometries are valid."""
    if not all(geoseries.is_valid):
        raise ValueError("All geometries must be valid.")


def safe_wkt_loads(wkt_str: str | None) -> BaseGeometry | None:
    """Safely load a WKT string into a geometry."""  # TODO Remove if alternative works
    try:
        if isinstance(wkt_str, str):
            return wkt.loads(wkt_str)
    except Exception:
        return None
    return None

def parse_geometries(
        df: pd.DataFrame,
        geom_column: GeometryColumn,
        return_null_geom: bool = False,
        crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Parse WKT geometries and return a GeoDataFrame."""
    geoseries = df[geom_column].apply(safe_wkt_loads)
    gdf = gpd.GeoDataFrame(
        df.drop(columns=[geom_column]),
        geometry=geoseries,
        crs=crs
    )

    gdf_nonnull = gdf[~gdf.geometry.isnull()]
    check_geometries(gdf_nonnull.geometry)

    return gdf if return_null_geom else gdf_nonnull

def dissolve_units(
        gdf: gpd.GeoDataFrame,
        aggfunc: str | Callable | list[Callable] | dict[
            str, str | Callable] = 'first'
) -> gpd.GeoDataFrame:
    """Dissolve units by DisNo. in geodataframe."""
    if "DisNo." not in gdf.columns:
        raise KeyError("'DisNo.' column is required for dissolve")
    logger.info("Dissolving units")
    # only dissolve, if dissolvable
    # if "admin1" in gdf.columns:  # TODO @jäger check if this was necessary
    gdf = gdf.dissolve(by="DisNo.", aggfunc=aggfunc)
    gdf.reset_index(inplace=True)
    logger.info(f"{len(gdf)} geocoded records after dissolving")
    return gdf

def filter_by_disnos(gdf, disnos: list[str]) -> gpd.GeoDataFrame:
    """Select rows from gdf that correspond to disnos"""
    if "DisNo." not in gdf.columns:
        raise KeyError("'DisNo.' column not found")
    return gdf[gdf["DisNo."].isin(disnos)]


def list_disno_in_benchmark(
        benchmark_type: BenchmarkGeomType,
        benchmark_path: str | Path
) -> list[str]:
    """Return the name of the benchmark geometry column."""
    if benchmark_type == "GAUL":
        disnos = load_emdat_archive(
            benchmark_path, use_columns=["DisNo."], geocoded_only=True
        )["DisNo."].to_list()
    elif benchmark_type == "GDIS":
        disnos = pd.read_csv(benchmark_path)["DisNo."].to_list()
    else:
        raise ValueError(f"Invalid benchmark type: {benchmark_type}")
    return disnos

def load_llm_csv_batch(
        csv_file_path: Path | str, columns_to_keep: list[str]
) -> pd.DataFrame:
    """Load a batch CSV file and select relevant columns."""
    return pd.read_csv(csv_file_path, usecols=columns_to_keep)

def load_emdat_archive(
        file_path: str | Path,
        use_columns: list[str] | None = None,
        min_year: int | None = 2000,
        max_year: int | None = None,
        geocoded_only: bool = False,
) -> pd.DataFrame:
    """Loads and returns a non-geocoded EMDAT archive as pandas DataFrame."""

    # load dataframe with relevant columns
    usecols: list[str] | None
    if use_columns is None:
        usecols = None
    else:
        usecols = use_columns.copy()
        if "Start Year" not in usecols:
            usecols.append("Start Year")
    if geocoded_only:
        if usecols is None:
            usecols = ["Start Year", "Admin Units"]
        elif "Admin Units" not in usecols:
            usecols.append("Admin Units")
    df = pd.read_excel(file_path, usecols=usecols)

    if min_year is not None:
        df = df[df["Start Year"] >= min_year]

    if max_year is not None:
        df = df[df["Start Year"] <= max_year]
    if geocoded_only:
        df = df[~df["Admin Units"].isnull()]

    return df[use_columns if use_columns else df.columns]

def load_benchmark(
        benchmark: BenchmarkGeomType,
        benchmark_path: str | Path,
        keep_columns: list[str] | None = None
) -> gpd.GeoDataFrame:
    """Load benchmark geodataframe."""
    logger.info(f"Loading {benchmark} geometries...")
    if benchmark == "GAUL":
        gdf_benchmark = load_gaul(benchmark_path, keep_columns=keep_columns)
    elif benchmark == "GDIS":
        gdf_benchmark = load_gdis(benchmark_path, keep_columns=keep_columns)
    else:
        raise ValueError(f"Invalid benchmark type: {benchmark}")
    return gdf_benchmark


def load_gaul(
        gaul_path: str | Path,
        keep_columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load EM-DAT with GAUL admin units."""
    _check_file_path(gaul_path, (".gpkg",))
    gdf = gpd.read_file(gaul_path)
    if "DisNo." not in gdf.columns:
        if "disno_" in gdf.columns:
            gdf.rename(columns={"disno_": "DisNo."}, inplace=True)
        else:
            raise KeyError(
                "Expected 'DisNo.' (or 'disno_') column not found in GAUL input")
    if keep_columns is not None:
        gdf = gdf[keep_columns]
    return gdf


def load_gdis(
        path: str | Path,
        keep_columns: list[str] | None = None
) -> gpd.GeoDataFrame:
    """Load GDIS with GADM admin units."""
    _check_file_path(path, (".gpkg",))
    gdf = gpd.read_file(path)
    gdf.rename(
        columns={
            "disasterno": "DisNo.",
            "geolocation": "name",
            "level": "admin_level",
            "adm1": "admin1",
            "adm2": "admin2",
            "adm3": "admin3",
        },
        inplace=True,
    )
    gdf["admin_level"] = "Admin" + gdf["admin_level"]
    if keep_columns is not None:
        gdf = gdf[keep_columns]
    return gdf
