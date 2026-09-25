"""Preprocessing driver for the GeoNames benchmark batch.

This script converts the GeoNames points ([geonames].points_path, produced by
geocoding/run_geonames_geocoding.py) into a GPKG batch file, filtered to the
DisNo. values that have EM-DAT GAUL geometries, like the LLM-geocoded batches.
"""

import tomllib
import logging
from pathlib import Path
from validation import preprocessing as pp
from validation.io import load_emdat_archive

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=config["logging"]["level"],
    filename=config["logging"]["filename"],
    filemode=config["logging"]["filemode"],
    style=config["logging"]["style"],
    format=config["logging"]["format"],
    datefmt=config["logging"]["datefmt"]
)


def main():
    disno_with_gaul = load_emdat_archive(
        config["path"]["emdat_archive_path"],
        use_columns=["DisNo."],
        geocoded_only=True
    )["DisNo."].to_list()
    output_dir = Path(config["path"]["batch_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pp.make_geonames_batch(
        points_path=config["geonames"]["points_path"],
        keep_disno=disno_with_gaul,
        output_dir=output_dir
    )


if __name__ == '__main__':
    logging.info(f"Running GeoNames preprocessing script...".upper())
    try:
        main()
    except Exception as e:
        logging.exception(f"Exception occurred: {e}")
