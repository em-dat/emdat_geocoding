"""Batch validation driver.

This script orchestrates geometry validation runs across combinations of:
- dissolving units or not,
- benchmark type (GAUL or GDIS), and
- prebuilt batch input files (.gpkg) under the configured batch directory.

Inputs, thresholds, and paths are read from config.toml. Outputs are CSV files
written to the configured validation_output_dir. Logs are written to
validation_all.log.
"""

import tomllib
import logging
from pathlib import Path
from validation import validation as v
from itertools import product

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=config["logging"]["level"],
    filename=config["logging"]["filename"],
    filemode=config["logging"]["filemode"],
    style=config["logging"]["style"],
    format=config["logging"]["format"],
    datefmt=config["logging"]["datefmt"],
)


DISSOLVE_UNITS = [True, False]
BENCHMARKS = ["GAUL", "GDIS"]
BATCHES = list(Path(config["path"]["batch_dir"]).glob("*.gpkg"))


def expected_output_path(
    subbatch: Path, benchmark: str, dissolve_units: bool, output_dir: Path
) -> Path:
    metadata = subbatch.stem.split("_")
    geom_type = "_".join(metadata[:2])
    batch_number = metadata[-1]
    output_filename = (
        f"{geom_type}_{benchmark.lower()}_batch{batch_number}"
        f"{'_dissolved' if dissolve_units else ''}"
        f".csv"
    )
    return output_dir / output_filename


def main():
    logging.info(f"Running validation script...".upper())
    combination = product(DISSOLVE_UNITS, BENCHMARKS, BATCHES)
    skip_if_exists = bool(
        config.get("validation", {}).get("skip_if_output_exists", False)
    )
    output_dir = Path(config["path"]["validation_output_dir"])
    for i, (dissolve_units, benchmark, batch) in enumerate(combination, start=1):
        try:
            # Skip GDIS against GDIS validation
            if batch.name.lower().startswith("gdis") and benchmark == "GDIS":
                continue
            # Skip validation if output exists (optional)
            if skip_if_exists:
                out_path = expected_output_path(
                    Path(batch), benchmark, dissolve_units, output_dir
                )
                if out_path.exists():
                    logging.info(
                        f"Skipping {batch} (benchmark={benchmark}, dissolve={dissolve_units}) because output exists: {out_path}"
                    )
                    continue

            v.compare_geometries(
                batch,
                benchmark,
                dissolve_units,
                config["geoprocessing"]["area_calculation_method"],
                config["path"]["emdat_archive_path"],
                config["path"]["emdat_gaul_path"],
                config["path"]["gdis_path"],
                config["path"]["gdis_disno_path"],
                config["path"]["validation_output_dir"],
            )
        except Exception as e:
            logging.exception(f"Exception occurred: {e}")

if __name__ == '__main__':
    logging.info(f"Running validation script...".upper())
    try:
        main()
    except Exception as e:
        logging.exception(f"Exception occurred: {e}")