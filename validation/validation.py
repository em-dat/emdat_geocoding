import logging
from pathlib import Path
import pandas as pd

from geotools.geom_indices import calculate_area_indices


def dissolve_units(gdf):

    logging.info("Dissolving units")
    # only dissolve, if dissolvable
    if "admin1" in gdf.columns:
        gdf = gdf.dissolve(
            by="DisNo.",
            aggfunc={
                "name": list,
                "admin_level": list,
                "admin1": list,
                "admin2": list,
            },
        )
        gdf = gdf.reset_index(inplace=False)
        logging.info(f"{len(gdf)} geocoded records after dissolving")

    return gdf


def validate(
    A_BATCH_FILE: Path,
    B_BATCH_FILE: Path,
    A_gdf,
    B_gdf,
    OUTPUT_COLUMNS: list,
    AREA_CALCULATION_METHOD: str,
    OUTPUT_FILENAME: str,
) -> None:

    # Run validation
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
