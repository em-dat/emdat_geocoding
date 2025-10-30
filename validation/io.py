import logging
from pathlib import Path
from typing import Literal, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from validation.validation import BenchmarkGeomType

logger = logging.getLogger(__name__)

GeometryColumn = Literal["geometry_osm", "geometry_gadm", "geometry_wiki"]
BatchNumber = Literal[1, 2, 3, 4, 5]


def load_llm_csv_batch(
        csv_file_path: Path | str, columns_to_keep: list[str]
) -> pd.DataFrame:
    """Load a batch CSV file and select relevant columns."""
    return pd.read_csv(csv_file_path, usecols=columns_to_keep)


def _check_file_path(file_path) -> Path:
    """Check the file path for existence and proper extension."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    extension = file_path.suffix.lower()
    if extension != ".gpkg":
        raise ValueError(
            f"Unsupported file extension: {extension}. " f"Only .gpkg is supported."
        )
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist.")
    return file_path


def safe_wkt_loads(wkt_str: str | None) -> BaseGeometry | None:
    """Safely load a WKT string into a geometry."""
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
        ignore_null_geom: bool = True,
) -> gpd.GeoDataFrame:
    """
    Loads and returns a geocoded batch as a GeoDataFrame from a specified .gpkg
    file.
    """
    file_path = _check_file_path(file_path)

    gdf = gpd.read_file(file_path, columns=use_columns)

    if ignore_null_geom:
        gdf = gdf[~gdf["geometry"].isnull()]
        check_geometries(gdf["geometry"])
    else:
        check_geometries(gdf[~gdf["geometry"].isnull()]["geometry"])

    return gdf


def load_emdat_archive(
        file_path: str | Path,
        use_columns: list[str] | None = None,
        min_year: int | None = 2000,
        max_year: int | None = None,
        geocoded_only: bool = False,
) -> pd.DataFrame:
    """Loads and returns a non-geocoded EMDAT archive as pandas DataFrame."""

    # add start year if not present in use_columns
    if use_columns and ("Start Year" not in use_columns):
        usecols = use_columns + ["Start Year"]
    else:
        usecols = use_columns

    if geocoded_only:
        usecols = usecols + ["Admin Units"]

    df = pd.read_excel(file_path, usecols=usecols)
    if min_year is not None:
        df = df[df["Start Year"] >= min_year]

    if max_year is not None:
        df = df[df["Start Year"] <= max_year]
    if geocoded_only:
        df = df[~df["Admin Units"].isnull()]

    return df[use_columns if use_columns else df.columns]


def parse_geometries(
        df: pd.DataFrame,
        geom_column: GeometryColumn,
        return_null_geom: bool = False,
        crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Parse WKT geometries and return a GeoDataFrame."""
    geoseries = df[geom_column].apply(safe_wkt_loads)
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_column]), geometry=geoseries,
                           crs=crs)

    gdf_nonnull = gdf[~gdf.geometry.isnull()]
    check_geometries(gdf_nonnull.geometry)

    return gdf if return_null_geom else gdf_nonnull


def load_benchmark(benchmark: str, benchmark_path: str | Path,
                   keep_columns: list[str] | None = None):
    logger.info(f"Loading {benchmark} geometries...")
    if benchmark == "GAUL":
        gdf_benchmark = load_GAUL(benchmark_path, keep_columns=keep_columns)
    elif benchmark == "GDIS":
        gdf_benchmark = load_GDIS(benchmark_path, keep_columns=keep_columns)
    return gdf_benchmark


def load_GAUL(
        gaul_path: str | Path,
        keep_columns: list[str] | None = None,
):
    """Load EM-DAT with GAUL admin units."""
    gdf = gpd.read_file(gaul_path)
    gdf.rename(columns={"disno_": "DisNo."}, inplace=True)
    if keep_columns is not None:
        gdf = gdf[keep_columns]
    return gdf


def load_GDIS(path: str | Path, keep_columns: list[str] | None = None):
    """Load EM-DAT with GAUL admin units."""
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


def get_disnos_numbers_from_llmbatch(df: pd.DataFrame) -> list[str]:
    disnos = list(df["DisNo."].unique())
    return disnos


def make_batch(gdf, disnos: list[str]):
    """Select rows from gdf that correspond to disnos"""
    gdf = gdf[gdf["DisNo."].isin(disnos)]
    return gdf


def fix_GDIS_disno(gdis_gdf, df_emdat: pd.DataFrame):
    """Add ISO to gdis_gdf disno and fix wrong no's based on df_emdat"""
    # Identify incorrect GDIS ISO's and set to nan
    iso_indicator = ~gdis_gdf["iso3"].isin(df_emdat["ISO"])
    gdis_gdf.loc[iso_indicator, "iso3"] = np.nan
    # Create an ISO - Country mapping based on EM-DAT
    df_emdat["ISO"] = (
        df_emdat["ISO"]
        .fillna(df_emdat["DisNo."].str.replace(r"[\W\d_]", "", regex=True))
        .str.replace(" (the)", "")
    )
    country_iso_mapping = dict(
        (x, y) for x, y in
        df_emdat.groupby(["Country", "ISO"]).apply(list).index.values
    )
    # Create new ISO variable and fill nans based on ISO - Country mapping
    gdis_gdf["ISO"] = gdis_gdf["iso3"].fillna(
        gdis_gdf["country"].map(country_iso_mapping)
    )
    # Create new DisNo. variable in same format as emdat data set
    gdis_gdf["DisNo."] = gdis_gdf["DisNo."] + "-" + gdis_gdf["ISO"]
    return gdis_gdf


def dissolve_units(
        gdf: gpd.GeoDataFrame,
        aggfunc: str | Callable | list[Callable] | dict[str, Callable] = 'first'
) -> gpd.GeoDataFrame:
    """Dissolve units by DisNo. in geodataframe."""
    logging.info("Dissolving units")
    # only dissolve, if dissolvable  # TODO
    # if "admin1" in gdf.columns:  # TODO check if this is necessary
    gdf = gdf.dissolve(
        by="DisNo.",
        aggfunc=aggfunc
    )
    gdf.reset_index(inplace=True)
    logging.info(f"{len(gdf)} geocoded records after dissolving")
    return gdf


def list_disno_in_benchmark(
        benchmark_type: BenchmarkGeomType,
        benchmark_path: str | Path | None = None
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


if __name__ == "__main__":
    gdf = gpd.read_file("../data/llm_subbatches/llm_wiki_3.gpkg")
    print(gdf.info())
    print(gdf.head())
    print(gdf.geom_type.unique())
    print(gdf.area)
