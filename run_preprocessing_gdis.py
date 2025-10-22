# %% Imports
from pathlib import Path
from validation.io import (
    load_emdat_archive,
    load_GDIS,
    fix_GDIS_disno,
)

# %% Config
EMDAT_ARCHIVE_PATH = Path("data/241204_emdat_archive.xlsx")
GDIS_PATH_RAW = Path("data/gdis_raw.gpkg")
GDIS_PATH = Path("data/gdis.gpkg")

emdat_columns = ["ISO", "DisNo.", "Country"]
gdis_columns = [
    "DisNo.",
    "name",
    "admin_level",
    "admin1",
    "admin2",
    "admin3",
    "iso3",
    "country",
    "geometry",
]

# %% Load gdis and emdat
df_emdat = load_emdat_archive(EMDAT_ARCHIVE_PATH)
gdis_gdf = load_GDIS(GDIS_PATH, gdis_columns)
gdis_gdf = fix_GDIS_disno(gdis_gdf, df_emdat)
gdis_gdf.to_file(GDIS_PATH, driver="GPKG")

# # %% Make gdis batches based on llm csv batches
# for batch in batches:
#     llm_csv_batch = load_llm_csv_batch(f"{path_llm_batches}{batch}.csv", ["DisNo."])
#     disnos = get_disnos_numbers_from_llmbatch(llm_csv_batch)
#     gdis_gdf_batch = make_batch(gdis_gdf, disnos)
#     save_batch(gdis_gdf_batch, path_gdis_batches, batch)
