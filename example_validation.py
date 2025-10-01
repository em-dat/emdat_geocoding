import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from geotools.geom_indices import calculate_area_indices
from validation.io import load_geocoded_batch, load_emdat_archive

# Define logger to track the validation process
logging.basicConfig(
    filename="example_validation.log",
    level=logging.INFO,
    filemode="a",
    style="{",
    format="{asctime} - {levelname} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Params
GEOMTYPE = "gadm"  # "gdis"
BATCH_START = 2000
BATCH_END = 2002

# Control whether to dissolve units before validation
# if True, units are merged to represent the entire disaster footprint to
# be compared to GAUL EM-DAT footprint, making the Jaccard index more meaningful
# if False, units are kept as they are, making the "a_in_b" more meaningful
DISSOLVE_UNITS = False

# Method for calculating areas
# If 'geodetic', calculate areas using WGS84 ellipsoid
# If 'equal_area', reproject to equal-area CRS and calculate planar areas
AREA_CALCULATION_METHOD: Literal["geodetic", "equal_area"] = "geodetic"

# GAUL or GDIS validation file
GEOCODED_BATCH_FILE = Path(f"data/geoloc_emdat_0002_{GEOMTYPE}.gpkg")
# LLM locations file
EMDAT_BATCH_FILE = Path(f"data/geoemdat_{BATCH_START}_{BATCH_END}.gpkg")
# EM-DAT tabular file
EMDAT_ARCHIVE_FILE = Path("data/241204_emdat_archive.xlsx")
OUTPUT_FILENAME = (
    f"validation_{GEOMTYPE}_{BATCH_START}_{BATCH_END}"
    f"{'_dissolved' if DISSOLVE_UNITS else ''}.csv"
)
OUTPUT_COLUMNS = [
    "dis_no",
    "name",
    "admin_level",
    "admin1",
    "admin2",
    "geom_type",
    "area_calculation_method",
    "area_a",
    "area_b",
    "intersection_area",
    "union_area",
    "a_in_b",
    "b_in_a",
    "jaccard",
]

# Official disaster ids from the Archive
# If 'geocoded_only', DISNO_OFFICIAL includes ids of geocoded disasters only.
DISNO_OFFICIAL = load_emdat_archive(
    file_path=EMDAT_ARCHIVE_FILE,
    min_year=BATCH_START,
    max_year=BATCH_END,
    use_columns=["DisNo."],
    geocoded_only=True,
)["DisNo."].to_list()


def main():
    logging.info(f"Starting {GEOMTYPE} validation".upper())
    # 1. Load GADM and EMDAT batches
    logging.info(f"Loading {GEOMTYPE} batch file {GEOCODED_BATCH_FILE}")
    geocoded_gdf = load_geocoded_batch(GEOCODED_BATCH_FILE)
    logging.info(f"{len(geocoded_gdf)} records loaded")
    logging.info(f"Columns: {', '.join(geocoded_gdf.columns)}")

    # 2. Dissolve units (optional)
    if DISSOLVE_UNITS:
        logging.info("Dissolving units")
        geocoded_gdf = geocoded_gdf.dissolve(
            by="DisNo.",
            aggfunc={"name": list, "admin_level": list, "admin1": list, "admin2": list},
        )
        geocoded_gdf.reset_index(inplace=True)
        logging.info(f"{len(geocoded_gdf)} records after dissolving")

    # 3. Filter based on official disaster ids
    logging.info(f"Filtering based on Dis No.")
    # Excluding unpublished and non-geocoded disasters
    geocoded_gdf = geocoded_gdf[geocoded_gdf["DisNo."].isin(DISNO_OFFICIAL)]
    logging.info(f"{len(geocoded_gdf)} records filtered based on Dis No.")

    # 4. Load EMDAT geocoded batch
    logging.info(f"Loading EMDAT batch file {EMDAT_BATCH_FILE}")
    emdat_gdf = load_geocoded_batch(EMDAT_BATCH_FILE)
    logging.info(f"{len(emdat_gdf)} records loaded")
    logging.info(f"Columns: {', '.join(emdat_gdf.columns)}")

    # 5. Run validation
    logging.info(f"Starting validation of {len(geocoded_gdf)} records")

    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    for ix, row in geocoded_gdf.iterrows():
        disno = row["DisNo."]
        geom_a = row["geometry"]
        geom_b = emdat_gdf[emdat_gdf["disno_"] == disno]["geometry"].iloc[0]
        indices: dict[str, float] = calculate_area_indices(geom_a, geom_b)
        results = [
            row["DisNo."],
            row["name"],
            row["admin_level"],
            row["admin1"],
            row["admin2"],
            GEOMTYPE,
            AREA_CALCULATION_METHOD,
        ] + list(indices.values())
        result_df.loc[ix] = results
    logging.info(f"Saving results to {OUTPUT_FILENAME}")
    result_df.to_csv(OUTPUT_FILENAME, index=False)
    logging.info(f"Validation complete".upper())


if __name__ == "__main__":
    main()
