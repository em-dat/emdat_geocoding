import logging
from pathlib import Path
import pandas as pd

from geotools.geom_indices import calculate_area_indices


def validate(
    A_BATCH_FILE: Path,
    B_BATCH_FILE: Path,
    A_gdf,
    B_gdf,
    DISSOLVE_UNITS: bool,
    DISNO_OFFICIAL: list,
    OUTPUT_COLUMNS: list,
    AREA_CALCULATION_METHOD: str,
    OUTPUT_FILENAME: str,
) -> None:

    # 2. Dissolve units (optional)
    if DISSOLVE_UNITS:
        logging.info("Dissolving units")
        # only dissolve, if dissolvable
        if "admin1" in A_gdf.columns:
            A_gdf = A_gdf.dissolve(
                by="DisNo.",
                aggfunc={
                    "name": list,
                    "admin_level": list,
                    "admin1": list,
                    "admin2": list,
                },
            )
            A_gdf.reset_index(inplace=True)
            logging.info(f"{len(A_gdf)} geocoded records after dissolving")

        if "admin1" in B_gdf.columns:
            # only dissolve, if dissolvable
            logging.info("Dissolving units")
            B_gdf = B_gdf.dissolve(
                by="DisNo.",
                aggfunc={
                    "name": list,
                    "admin_level": list,
                    "admin1": list,
                    "admin2": list,
                },
            )
            B_gdf.reset_index(inplace=True)
            logging.info(f"{len(B_gdf)} gdis records after dissolving")

    # 3 Filter based on official disaster ids
    logging.info(f"Filtering based on Dis No.")
    # Excluding unpublished and non-geocoded disasters
    A_gdf = A_gdf[A_gdf["DisNo."].isin(DISNO_OFFICIAL)]
    logging.info(f"{len(A_gdf)} records filtered based on Dis No. in {A_BATCH_FILE}")
    B_gdf = B_gdf[B_gdf["DisNo."].isin(DISNO_OFFICIAL)]
    logging.info(f"{len(B_gdf)} records filtered based on Dis No. in {B_BATCH_FILE}")

    # 4 Run validation
    result_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # TO BE DONE: Dealing with Dis No. that are in A but not in B and vice versa. (Relevant for GDIS / EM-DAT)
    # Maybe different approaches for common disno and disno's that are only in A and only in B
    # e.g.,   common: disno = set(A_gdf["DisNo."]).intersection(set(B_gdf["DisNo."]))
    for ix, row in A_gdf.iterrows():
        disno = row["DisNo."]
        geom_a = row["geometry"]
        if disno in B_gdf["DisNo."]:  # quick fix for now
            print(f"Validating {disno}")
            geom_b = B_gdf[B_gdf["DisNo."] == disno]["geometry"].iloc[0]
            indices: dict[str, float] = calculate_area_indices(geom_a, geom_b)
            results = [
                row["DisNo."],
                row["name"],
                row["admin_level"],
                row["admin1"],
                row["admin2"],
                AREA_CALCULATION_METHOD,
            ] + list(indices.values())
            result_df.loc[ix] = results
        else:
            result_df.loc[ix] = None
            print(f"{disno} of {A_BATCH_FILE} is not in {B_BATCH_FILE}")
    logging.info(f"Saving results to {OUTPUT_FILENAME}")
    result_df.to_csv(OUTPUT_FILENAME, index=False)
    logging.info(f"Validation complete".upper())

    return
