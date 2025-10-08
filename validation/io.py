from pathlib import Path

import geopandas as gpd
import pandas as pd


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


def _check_geometries(geoseries: gpd.GeoSeries) -> bool:
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
        _check_geometries(gdf['geometry'])
    else:
        _check_geometries(gdf[~gdf['geometry'].isnull()]['geometry'])

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
