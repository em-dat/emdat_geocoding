"""Script to create LLM-geocoded batches.

CSV files 1 to 5 sent by Michele are loaded and converted to geopackages that
only use on specific geometries, such as OSM, GADM, and Wikidata, with only
records that have GAUL admin units.

Notes
-----
`CSV_FILE_DIR` should contain:
['geoemdat_part1.csv', 'geoemdat_part2.csv', 'geoemdat_part3.csv',
 'geoemdat_part4.csv', 'geoemdat_part5.csv'].

Each file must include the following columns:
['DisNo.', 'name', 'admin_level', 'admin1', 'admin2', 'admin3',
 'geometry_osm', 'geometry_gadm', 'geometry_wiki', 'iso3'].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CSV_FILE_DIR = Path("Q:/Data/GEOEMDAT")
OUTPUT_DIR = Path("data/llm_geocoded_batches")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = [
    "DisNo.", "name", "admin_level", "admin1", "admin2", "admin3",
    "geometry_osm", "geometry_gadm", "geometry_wiki", "iso3",
]
COLUMN_NAMES_TO_DROP = ["admin3"]  # always empty

EMDAT_ARCHIVE_FILE = Path("data/241204_emdat_archive.xlsx")
emdat = pd.read_excel(EMDAT_ARCHIVE_FILE, usecols=["DisNo.", "Admin Units"])
DISNO_WITH_GAUL = emdat[~emdat["Admin Units"].isnull()]["DisNo."].tolist()
del emdat
# @wiebke we could add a DISNO_WITH_GDIS parameter here

GeometryColumn = Literal["geometry_osm", "geometry_gadm", "geometry_wiki"]
BatchNumber = Literal[1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_csv_file(batch_number: BatchNumber,
                  geom_column: GeometryColumn) -> pd.DataFrame:
    """Load a batch CSV file and select relevant columns.

    Parameters
    ----------
    batch_number : BatchNumber
        Batch number (1–5).
    geom_column : GeometryColumn
        Geometry column to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing selected columns.
    """
    file_name = f"geoemdat_part{batch_number}.csv"
    csv_file_path = CSV_FILE_DIR / file_name
    columns = [c for c in COLUMN_NAMES if "geometry" not in c] + [geom_column]
    for c in COLUMN_NAMES_TO_DROP:
        if c in columns:
            columns.remove(c)
    return pd.read_csv(csv_file_path, usecols=columns)


def safe_wkt_loads(wkt_str: str | None) -> BaseGeometry | None:
    """Safely load a WKT string into a geometry.

    Parameters
    ----------
    wkt_str : str or None
        WKT representation of the geometry.

    Returns
    -------
    BaseGeometry or None
        Parsed geometry, or None if invalid.
    """
    try:
        if isinstance(wkt_str, str):
            return wkt.loads(wkt_str)
    except Exception:
        return None
    return None


def _check_geometries(geoseries: gpd.GeoSeries) -> None:
    """Check if geometries are valid.

    Raises
    ------
    ValueError
        If invalid geometries are found.
    """
    if not geoseries.is_valid.all():
        raise ValueError("Some geometries are invalid.")


def parse_geometries(
        df: pd.DataFrame,
        geom_column: GeometryColumn,
        return_null_geom: bool = False,
) -> gpd.GeoDataFrame:
    """Parse WKT geometries and return a GeoDataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a geometry column.
    geom_column : GeometryColumn
        Geometry column name.
    return_null_geom : bool, default=False
        If False, drops rows with null geometries.

    Returns
    -------
    gpd.GeoDataFrame
        Parsed and validated GeoDataFrame.
    """
    geoseries = df[geom_column].apply(safe_wkt_loads)
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_column]), geometry=geoseries,
                           crs="EPSG:4326")

    gdf_nonnull = gdf[~gdf.geometry.isnull()]
    try:
        _check_geometries(gdf_nonnull.geometry)
    except ValueError:
        print(f"Warning: invalid geometries detected in '{geom_column}'.")
        valid_mask = gdf_nonnull.geometry.is_valid
        invalid_geoms = gdf_nonnull.loc[~valid_mask, "geometry"]

        df_invalid = pd.DataFrame({
            "invalid_ids": gdf_nonnull.loc[~valid_mask, "DisNo."],
            "geom_type": geom_column.split("_")[-1],
            "reasons": invalid_geoms.apply(explain_validity),
        })

        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        outfile = f"{timestamp}_invalid_geoms_{geom_column}.csv"
        df_invalid.to_csv(outfile, index=False)
        print(f"Saved invalid geometry log: {outfile}")
        raise

    return gdf if return_null_geom else gdf_nonnull


def make_llm_geocoded_batches(keep_disno: list=DISNO_WITH_GAUL) -> None:
    """Create LLM-geocoded GeoPackage batches."""
    print("Starting LLM-geocoded batch creation...")
    print(f"Files found in {CSV_FILE_DIR}: {os.listdir(CSV_FILE_DIR)}")

    for batch_number in range(1, 6):
        print(f"Processing batch {batch_number}...")
        for geom_column in ["geometry_osm", "geometry_gadm", "geometry_wiki"]:
            df = load_csv_file(batch_number, geom_column)
            # Keep only records with GAUL admin units for the validation
            df = df[df["DisNo."].isin(keep_disno)]
            gdf = parse_geometries(df, geom_column)

            suffix = geom_column.split("_")[-1]
            output_path = OUTPUT_DIR / f"llm_{suffix}_{batch_number}.gpkg"
            gdf.to_file(output_path)
            print(f"Saved: {output_path}")

    print("LLM-geocoded batch creation complete.")


if __name__ == "__main__":
    make_llm_geocoded_batches()
