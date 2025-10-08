# %%
from pathlib import Path
from validation.io import load_geocoded_batch, load_emdat_archive
import numpy as np

# %% Paths
GDIS_FILE = Path(f"data/gdis.gpkg")
EMDAT_ARCHIVE_FILE = Path("data/241204_emdat_archive.xlsx")

# %% Load files
gdis_gdf = load_geocoded_batch(GDIS_FILE)
df_emdat = load_emdat_archive(
    file_path=EMDAT_ARCHIVE_FILE,
    use_columns=["ISO", "DisNo.", "Country"],
    min_year=None,
    max_year=None,
    geocoded_only=True,
)

# %% Preprocess GDIS so that it is compatible with geocoded LLM file
gdis_gdf.rename(
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

# Get disasteryear from disaster number
gdis_gdf["disasteryear"] = gdis_gdf["DisNo."].apply(lambda x: int(x[0:4]))


# %% Fix GDIS disaster number so that it corresponds to EM-DAT disaster number

# Identify incorrect GDIS ISO's and set to nan
iso_indicator = ~gdis_gdf["iso3"].isin(df_emdat["ISO"])
gdis_gdf.loc[iso_indicator, "iso3"] = np.nan

# Create an ISO - Country mapping based on EM-DAT
df_emdat["ISO"] = (
    df_emdat["ISO"]
    .fillna(df_emdat["DisNo."].str.replace("[\W\d_]", "", regex=True))
    .str.replace(" (the)", "")
)
country_iso_mapping = dict(
    (x, y) for x, y in df_emdat.groupby(["Country", "ISO"]).apply(list).index.values
)

# Create new ISO variable and fill nans based on ISO - Country mapping
gdis_gdf["ISO"] = gdis_gdf["iso3"].fillna(gdis_gdf["country"].map(country_iso_mapping))

# Create new DisNo. variable in same format as emdat data set
gdis_gdf["DisNo."] = gdis_gdf["DisNo."] + "-" + gdis_gdf["ISO"]


# %% Save GDIS File in batches
# Define batch
BATCH_START = 2000
BATCH_END = 2002
GDIS_BATCH_FILE = Path(f"data/gdis_{BATCH_START}_{BATCH_END}.gpkg")

# Filter on batch years
year_filter = (gdis_gdf["disasteryear"] >= BATCH_START) & (
    gdis_gdf["disasteryear"] <= BATCH_END
)
# Filter to batch year
gdis_gdf_batch = gdis_gdf[year_filter]
# Only keep relevant columns
gdis_gdf_batch = gdis_gdf_batch[
    ["DisNo.", "name", "admin_level", "admin1", "admin2", "admin3", "geometry"]
]
gdis_gdf_batch["admin_level"] = "Admin" + gdis_gdf_batch["admin_level"]
# Save
gdis_gdf_batch.to_file(GDIS_BATCH_FILE, driver="GPKG", overwrite=True)

# %%
