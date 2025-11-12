# %% Imports
from pathlib import Path
from validation.io import (
    load_emdat_archive,
    _load_GDIS,
    fix_GDIS_disno,
)
import pandas as pd

# %% Config
EMDAT_ARCHIVE_PATH = Path("data/241204_emdat_archive.xlsx")
GDIS_PATH_RAW = Path("data/gdis_raw.gpkg")
GDIS_PATH = Path("data/gdis.gpkg")
GDIS_DISNO_PATH = Path("data/gdis_disnos.csv")


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
gdis_gdf = _load_GDIS(GDIS_PATH_RAW, gdis_columns)
gdis_gdf = fix_GDIS_disno(gdis_gdf, df_emdat)
gdis_disno_df = pd.DataFrame(data=gdis_gdf["DisNo."].unique(), columns=["DisNo."])

# %% Save unique disaster numbers in GDIS
gdis_disno_df.to_csv(GDIS_DISNO_PATH, index=False)

# %% Save GDIS
gdis_gdf.to_file(GDIS_PATH, driver="GPKG")
