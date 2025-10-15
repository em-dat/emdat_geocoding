# %%
import logging
from pathlib import Path
from typing import Literal

from validation.io import load_emdat_archive, load_geocoded_batch

from validation.validation import dissolve_units, validate

# Define logger to track the validation process
logging.basicConfig(
    filename="example_validation.log",
    level=logging.INFO,
    filemode="a",
    style="{",
    format="{asctime} - {levelname} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# %%
# Params
# Choose data from "llm_gadm", "llm_osm", "emdat_gaul", "gdis_gadm"
DATA_A = "emdat_gaul"
DATA_B = "gdis_gadm"
BATCH_START = 2000
BATCH_END = 2002

# Control whether to dissolve units before validation
# if True, units are merged to represent the entire disaster footprint to
# be compared to GAUL EM-DAT footprint, making the Jaccard index more meaningful
# if False, units are kept as they are, making the "a_in_b" more meaningful
DISSOLVE_UNITS = True

# Method for calculating areas
# If 'geodetic', calculate areas using WGS84 ellipsoid
# If 'equal_area', reproject to equal-area CRS and calculate planar areas
AREA_CALCULATION_METHOD: Literal["geodetic", "equal_area"] = "geodetic"

# A File
A_BATCH_FILE = Path(f"data/{DATA_A}_{BATCH_START}_{BATCH_END}.gpkg")
# B File
B_BATCH_FILE = Path(f"data/{DATA_B}_{BATCH_START}_{BATCH_END}.gpkg")
# EM-DAT tabular file
EMDAT_ARCHIVE_FILE = Path("data/241204_emdat_archive.xlsx")
OUTPUT_FILENAME = (
    f"validation_{DATA_A}_{DATA_B}_{BATCH_START}_{BATCH_END}"
    f"{'_dissolved' if DISSOLVE_UNITS else ''}.csv"
)
OUTPUT_FILENAME_GDIS = (
    f"validation_GDIS_{BATCH_START}_{BATCH_END}"
    f"{'_dissolved' if DISSOLVE_UNITS else ''}.csv"
)
OUTPUT_COLUMNS = [
    "dis_no",
    "name",
    "admin_level",
    "admin1",
    "admin2",
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

# %%
logging.info(f"Starting validation".upper())
# 1a. Load first gdf
logging.info(f"Loading {A_BATCH_FILE} batch file")
A_gdf = load_geocoded_batch(A_BATCH_FILE)
logging.info(f"{len(A_gdf)} records loaded")
logging.info(f"Columns: {', '.join(A_gdf.columns)}")

# 1b. Load second gdf
logging.info(f"Loading {B_BATCH_FILE} batch file")
B_gdf = load_geocoded_batch(B_BATCH_FILE)
logging.info(f"{len(B_gdf)} records loaded")
logging.info(f"Columns: {', '.join(B_gdf.columns)}")


# %%
# Filter based on official disaster ids
logging.info(f"Filtering based on Dis No.")
# Excluding unpublished and non-geocoded disasters
A_gdf = A_gdf[A_gdf["DisNo."].isin(DISNO_OFFICIAL)]
logging.info(f"{len(A_gdf)} records filtered based on Dis No. in {A_BATCH_FILE}")
B_gdf = B_gdf[B_gdf["DisNo."].isin(DISNO_OFFICIAL)]
logging.info(f"{len(B_gdf)} records filtered based on Dis No. in {B_BATCH_FILE}")


# %%
if DISSOLVE_UNITS:
    A_gdf = dissolve_units(A_gdf)
    B_gdf = dissolve_units(B_gdf)

# %%
validate(
    A_BATCH_FILE,
    B_BATCH_FILE,
    A_gdf,
    B_gdf,
    OUTPUT_COLUMNS,
    AREA_CALCULATION_METHOD,
    OUTPUT_FILENAME,
)
