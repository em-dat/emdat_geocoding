# %%
from pathlib import Path
from validation.io import load_emdat_archive
import numpy as np
import pandas as pd

# %% Paths
GDIS_CSV_FILE = Path(f"data/pend-gdis-1960-2018-disasterlocations.csv")
GDIS_EMDAT_CSV_FILE = Path(f"data/gdis_emdat.csv")
EMDAT_ARCHIVE_FILE = Path("data/241204_emdat_archive.xlsx")

# %% Load files
df_gdis = pd.read_csv(GDIS_CSV_FILE)
df_emdat = load_emdat_archive(
    file_path=EMDAT_ARCHIVE_FILE,
    use_columns=["ISO", "DisNo.", "Country"],
    min_year=None,
    max_year=None,
    geocoded_only=True,
)

# %% Preprocess GDIS so that it is compatible with geocoded LLM file
df_gdis.rename(
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

# %% Fix GDIS disaster number so that it corresponds to EM-DAT disaster number

# Identify incorrect GDIS ISO's and set to nan
iso_indicator = ~df_gdis["iso3"].isin(df_emdat["ISO"])
df_gdis.loc[iso_indicator, "iso3"] = np.nan

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
df_gdis["ISO"] = df_gdis["iso3"].fillna(df_gdis["country"].map(country_iso_mapping))

# Create new DisNo. variable in same format as emdat data set
df_gdis["DisNo."] = df_gdis["DisNo."] + "-" + df_gdis["ISO"]

# %% Aggregate GDIS to country level
df_gdis_grouped = df_gdis.groupby("DisNo.", axis=0).agg(lambda x: list(pd.unique(x)))


df_gdis_emdat = df_gdis_grouped.merge(right=df_emdat, how="inner", on="DisNo.")

# %%
df_gdis_emdat.to_csv(GDIS_EMDAT_CSV_FILE)
