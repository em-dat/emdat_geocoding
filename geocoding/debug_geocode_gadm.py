# %%import pandas as pd
from gadm_utils import (
    match_location_to_gadm_new,
    read_admin,
    clean_location_name,
)
import tomllib
import pandas as pd

# %%
# Get the GADM layers
with open("../config.toml", "rb") as f:
    config = tomllib.load(f)

gadm_path = config["geocoding"].get("gadm_preprocessed_path") or config[
    "geocoding"
].get("gadm_path")
if not gadm_path:
    raise ValueError(
        "GADM path not found in config.toml ([geocoding].gadm_preprocessed_path or [geocoding].gadm_path)"
    )

gadm1 = read_admin(config["geocoding"]["gadm_preprocessed_path"], 1)
gadm2 = read_admin(config["geocoding"]["gadm_preprocessed_path"], 2)

# %%
# Load the geoparsed data wihtout polygon
parsed_csv = pd.read_csv(
    "C:/Users/wja209/Git/emdat_geocoding/data/llmGeodis_nogeom.csv", sep=","
)
parsed_csv = parsed_csv.dropna(subset=["admin1", "admin2", "admin3"], how="all")

# %%

admin1_count = 0
count_rows = 0
failed_rows = []
admin_results = []

for _, row in parsed_csv.iterrows():
    count_rows += 1
    row_dict = {
        "DisNo": row["DisNo."],
        "name": row["name"],
        "admin_level": row["admin_level"],
        "Admin1": [clean_location_name(row["admin1"])],
        "Admin2": [clean_location_name(row["admin2"])],
        "admin3": [clean_location_name(row["admin3"])],
        "emdat_iso3": row["DisNo."][-3:],
    }

    gadm_result = match_location_to_gadm_new(
        parsed_json=row_dict, gadm1=gadm1, gadm2=gadm2, iso3=row_dict["emdat_iso3"]
    )

    if gadm_result["Admin1"] is not None and len(gadm_result["Admin1"]) > 0:
        admin_results.append(
            {
                "DisNo": row["DisNo."],
                "name": row["name"],
                "Admin1": gadm_result["Admin1"],
                "Admin2": gadm_result["Admin2"],
            }
        )
        admin1_count += 1
    else:
        failed_rows.append(row)

    # if count_rows == 2000:
    #     foo = 2
    #     break

failed_df = pd.DataFrame(failed_rows)
admin_results_df = pd.DataFrame(admin_results)
admin1_count

# %%
