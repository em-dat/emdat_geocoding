# %%
from pathlib import Path
from validation.io import load_geocoded_batch

# %% Paths
EMDAT_GAUL_BATCH_FILE = Path(f"data/emdat_gaul_2000_2002.gpkg")

# %% Load files
emdat_gaul_gdf_batch = load_geocoded_batch(EMDAT_GAUL_BATCH_FILE)
# %%

# reprocess GDIS so that it is compatible with geocoded LLM file
emdat_gaul_gdf_batch.rename(
    columns={
        "disno_": "DisNo.",
    },
    inplace=True,
)
# Save file
emdat_gaul_gdf_batch.to_file(EMDAT_GAUL_BATCH_FILE, driver="GPKG", overwrite=True)
##
# %%
