import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

logger = logging.getLogger(__name__)

GeometryColumn = Literal["geometry_osm", "geometry_gadm", "geometry_wiki"]
BatchNumber = Literal[1, 2, 3, 4, 5]


def load_llm_csv_batch(
        csv_file_path: Path | str,
        columns_to_keep: list[str]
) -> pd.DataFrame:
    """Load a batch CSV file and select relevant columns.
    """
    return pd.read_csv(csv_file_path, usecols=columns_to_keep)


def _check_file_path(file_path) -> Path:
    """Check the file path for existence and proper extension."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    extension = file_path.suffix.lower()
    if extension != '.gpkg':
        raise ValueError(f'Unsupported file extension: {extension}. '
                         f'Only .gpkg is supported.')
    if not file_path.exists():
        raise ValueError(f'File {file_path} does not exist.')
    return file_path


def safe_wkt_loads(wkt_str: str | None) -> BaseGeometry | None:
    """Safely load a WKT string into a geometry.
    """
    try:
        if isinstance(wkt_str, str):
            return wkt.loads(wkt_str)
    except Exception:
        return None
    return None


def check_geometries(geoseries: gpd.GeoSeries) -> bool:
    """Check if geometries are valid."""
    if not all(geoseries.is_valid):
        raise ValueError("All geometries must be valid.")


def load_geocoded_batch(
        file_path: str | Path,
        use_columns: list[str] | None = None,
        ignore_null_geom: bool = True
) -> gpd.GeoDataFrame:
    """
    Loads and returns a geocoded batch as a GeoDataFrame from a specified .gpkg
    file.
    """
    file_path = _check_file_path(file_path)

    gdf = gpd.read_file(file_path, columns=use_columns)

    if ignore_null_geom:
        gdf = gdf[~gdf['geometry'].isnull()]
        check_geometries(gdf['geometry'])
    else:
        check_geometries(gdf[~gdf['geometry'].isnull()]['geometry'])

    return gdf


def load_emdat_archive(
        file_path: str | Path,
        use_columns: list[str] | None = None,
        min_year: int | None = 2000,
        max_year: int | None = None,
        geocoded_only: bool = False
) -> pd.DataFrame:
    """Loads and returns a non-geocoded EMDAT archive as pandas DataFrame."""

    # add start year if not present in use_columns
    if use_columns and ('Start Year' not in use_columns):
        usecols = use_columns + ['Start Year']
    else:
        usecols = use_columns

    if geocoded_only:
        usecols = usecols + ['Admin Units']

    df = pd.read_excel(file_path, usecols=usecols)
    if min_year is not None:
        df = df[df['Start Year'] >= min_year]

    if max_year is not None:
        df = df[df['Start Year'] <= max_year]
    if geocoded_only:
        df = df[~df['Admin Units'].isnull()]

    return df[use_columns if use_columns else df.columns]


def check_geometries(geoseries: gpd.GeoSeries) -> None:
    """Check if geometries are valid.
    """
    if not geoseries.is_valid.all():
        raise ValueError("Some geometries are invalid.")


def parse_geometries(
        df: pd.DataFrame,
        geom_column: GeometryColumn,
        return_null_geom: bool = False,
        crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Parse WKT geometries and return a GeoDataFrame.
    """
    geoseries = df[geom_column].apply(safe_wkt_loads)
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_column]), geometry=geoseries,
                           crs=crs)

    gdf_nonnull = gdf[~gdf.geometry.isnull()]
    check_geometries(gdf_nonnull.geometry)

    return gdf if return_null_geom else gdf_nonnull

def load_GAUL(
        gaul_path: str | Path,
        disno: list[str] | None = None,
        keep_columns: list[str] | None = None
):
    """Load EM-DAT with GAUL admin units."""
    gdf = gpd.read_file(gaul_path)
    gdf.rename(columns={'disno_': 'DisNo.'}, inplace=True)
    if disno is not None:
        gdf = gdf[gdf['DisNo.'].isin(disno)]
    if keep_columns is not None:
        gdf = gdf[keep_columns]
    return gdf

def load_GDIS():
    raise NotImplementedError

if __name__ == "__main__":
    import os
    gdf = gpd.read_file("../data/llm_subbatches/llm_wiki_3.gpkg")
    print(gdf.info())
    print(gdf.head())
    print(gdf.geom_type.unique())
    print(gdf.area)