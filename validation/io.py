"""IO utilities for validation and preprocessing.

This module centralizes small, focused helpers to read and prepare inputs used
by the project's validation workflows. It provides:

- File/path checks for expected formats.
- Readers for GAUL- and GDIS-based benchmark layers (GeoPackage expected).
- Parsers for WKT geometries produced by LLM-assisted geocoding batches.
- Convenience filters and dissolve operations keyed by EM-DAT DisNo.

Conventions
-----------
- Default storage CRS is EPSG:4326 (WGS84).
- Geometries are validated on load; invalid or null geometries can be dropped
  depending on the function's parameters.
"""

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
    """Validate a path and its extension.
    """
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext not in allowed_suffixes:
        raise ValueError(
            f"Unsupported file extension: {ext}. Allowed: {allowed_suffixes}")
    if not p.exists():
        raise ValueError(f"File {p} does not exist.")
    return p


def check_geometries(geoseries: gpd.GeoSeries) -> None:
    """Validate a GeoSeries of geometries.

    Parameters
    ----------
    geoseries
        GeoPandas GeoSeries to validate.

    Raises
    ------
    ValueError
        If any geometry is invalid.
    """
    if not all(geoseries.is_valid):
        raise ValueError("All geometries must be valid.")


def safe_wkt_loads(wkt_str: str | None) -> BaseGeometry | None:
    """Safely parse a WKT string.

    Parameters
    ----------
    wkt_str
        Well-Known Text representation of a geometry, or None.

    Returns
    -------
    shapely.geometry.base.BaseGeometry | None
        Parsed geometry if successful, otherwise None.
    """
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
    """Parses geometries in a DataFrame to create a GeoDataFrame.

    This function converts geometries stored in a DataFrame column into a
    GeoDataFrame. It also checks for null geometries and optionally removes
    them.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing geometry data.
    geom_column : GeometryColumn
        The name of the column in the DataFrame that contains geometry
        information.
    return_null_geom : bool, optional
        Whether to return the entire GeoDataFrame including rows with null
        geometries. Defaults to False, meaning rows with null geometries are
        excluded.
    crs : str, optional
        Coordinate reference system of the resulting GeoDataFrame. Defaults to
        "EPSG:4326".

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame created from the input DataFrame, with geometries parsed
        from the specified column and the specified CRS applied. Rows with null
        geometries are included or excluded based on the `return_null_geom`
        parameter.
    """
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
    """Dissolves a GeoDataFrame based on the DisNo. column

    This function groups geometries in the input GeoDataFrame by the "DisNo."
    column and aggregates their associated attributes using the specified
    aggregation function. The resulting GeoDataFrame is reindexed and includes
    a summary of the grouped geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame containing geometries to be dissolved. This
        GeoDataFrame must include a column named "DisNo." to specify the grouping.
    aggfunc : str or Callable or list of Callable or dict of str to (str or Callable), optional
        Aggregation function(s) to apply to the attributes of grouped geometries.
        The behavior of this parameter depends on its type:
        - If a string is provided, it specifies a predefined aggregation function
          (e.g., "sum", "mean", etc.).
        - If a callable is provided, it is used to compute aggregation for each
          group.
        - If a list of callables is supplied, each callable is applied to relevant
          columns in turn.
        - If a dictionary is passed, it maps column names to specific aggregation
          functions for those columns.
        The default value is "first".

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing geometries and attributes aggregated by the
        "DisNo." column, with updated indexing and summary records.

    Raises
    ------
    KeyError
        If the input GeoDataFrame does not contain a "DisNo." column.

    """
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
    """Filters a GeoDataFrame based on a list of disaster numbers (disnos).

    Given a GeoDataFrame a disaster number "DisNo." column and a list of
    disaster numbers, this function filters the GeoDataFrame to include only
    the rows where the "DisNo." value is present in the input list.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to be filtered. It must contain a column named "DisNo.".
    disnos : list of str
        The list of district numbers used to filter the GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        A filtered GeoDataFrame containing only rows where the "DisNo." value
        matches one of the district numbers in `disnos`.

    Raises
    ------
    KeyError
        If the GeoDataFrame does not contain a column named "DisNo.".
    """
    if "DisNo." not in gdf.columns:
        raise KeyError("'DisNo.' column not found")
    return gdf[gdf["DisNo."].isin(disnos)]


def list_disno_in_benchmark(
        benchmark_type: BenchmarkGeomType,
        benchmark_path: str | Path
) -> list[str]:
    """List disaster numbers (DisNo) from a benchmark file based on its type.

    Parameters
    ----------
    benchmark_type : BenchmarkGeomType
        Type of the benchmark, specifying the format for extracting disaster
        numbers. Allowed values are "GAUL" and "GDIS".
    benchmark_path : str or Path
        Path to the benchmark file to be processed.

    Returns
    -------
    list of str
        A list containing disaster numbers (DisNo) extracted from the benchmark
        file.

    Raises
    ------
    ValueError
        If the benchmark type is not "GAUL" or "GDIS".
    """
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
    """Loads a LLM-geocoded batch of data from a CSV file.

    Parameters
    ----------
    csv_file_path : Path or str
        The file path to the CSV file to be loaded. Accepts either a string or
        a pathlib Path object.
    columns_to_keep : list of str
        A list of column names to retain in the dataframe. Only these columns
        will be included in the resulting dataframe.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data loaded from the CSV file,
        filtered to include only the specified columns.
    """
    return pd.read_csv(csv_file_path, usecols=columns_to_keep)


def load_emdat_archive(
        file_path: str | Path,
        use_columns: list[str] | None = None,
        min_year: int | None = 2000,
        max_year: int | None = None,
        geocoded_only: bool = False,
) -> pd.DataFrame:
    """
    Loads data from an EM-DAT archive file, applies optional filtering based
    on years, and optionally filters for geocoded events.

    Parameters
    ----------
    file_path : str or Path
        The path to the EM-DAT Excel file to load.
    use_columns : list of str, optional
        A list of column names to load from the file. If None, all columns
        will be loaded by default.
    min_year : int, optional
        The minimum year for filtering the data. Rows where "Start Year"
        is less than this value will be excluded.
    max_year : int, optional
        The maximum year for filtering the data. Rows where "Start Year"
        is greater than this value will be excluded.
    geocoded_only : bool, optional
        If True, filters the data to include only events with geocoded
        locations (i.e., non-null "Admin Units").

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the filtered data from the specified
        EM-DAT archive file, respecting the optional parameters. If
        `use_columns` is provided, only those columns will be returned;
        otherwise, all columns are included in the output.
    """

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
    """Loads benchmark geometries from the specified benchmark type and path.

    Parameters
    ----------
    benchmark : BenchmarkGeomType
        The type of benchmark to load (e.g., "GAUL" or "GDIS").
    benchmark_path : str or Path
        The file path to the benchmark data.
    keep_columns : list of str, optional
        A list of column names to retain in the resulting GeoDataFrame. If None, all
        columns are kept by default.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the benchmark geometries with the specified configurations.
    """
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
    """Loads a GeoPackage file containing EM-DAT GAUL data into a GeoDataFrame.

    Parameters
    ----------
    gaul_path : str or Path
        The file path to the GAUL GeoPackage file. The file must have a .gpkg
        extension.

    keep_columns : list of str, optional
        A list of column names to retain in the resulting GeoDataFrame. If not
        provided, all columns will be included.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing data from the input GeoPackage file, filtered by
        the columns specified in 'keep_columns', if provided.
    """
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
    """
    Load a GeoPackage file containing GDIS GADM data into a GeoDataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the GeoPackage (.gpkg) file. This can be a string or a Path
        object.
    keep_columns : list[str] or None, optional
        List of column names to retain in the resulting GeoDataFrame. If None,
        all columns will be included.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the renamed and optionally filtered data
        from the GeoPackage.

    """
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
