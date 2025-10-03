# %%
from pathlib import Path
from validation.io import load_geocoded_batch

# %%
BATCH_START = 2000
BATCH_END = 2002

# %% Load GDIS File
GEOCODED_FILE = Path(f"data/geoloc_emdat_gdis.gpkg")
GEOCODED_BATCH_FILE = Path(f"data/geoloc_emdat_0002_gdis.gpkg")

geocoded_gdf = load_geocoded_batch(GEOCODED_FILE)

# %% Save GDIS File in batches
geocoded_gdf["disasteryear"] = geocoded_gdf.disasterno.apply(lambda x: int(x[0:4]))
year_filter = (geocoded_gdf["disasteryear"] >= BATCH_START) & (
    geocoded_gdf["disasteryear"] <= BATCH_END
)
geocoded_gdf.rename(
    columns={
        "disasterno": "DisNo.",
        "level": "admin_level",
        "adm1": "admin1",
        "adm2": "admin2",
        "adm3": "admin3",
    }
)
geocoded_gdf["name"] = geocoded_gdf["id"]

# %%
geocoded_gdf_0002 = geocoded_gdf[year_filter]

# %%
geocoded_gdf_0002.to_file(GEOCODED_BATCH_FILE, driver="GPKG")


# %%
