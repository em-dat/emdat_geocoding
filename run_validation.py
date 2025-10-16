import tomllib
import logging
from pathlib import Path
from validation import validation as v
from itertools import product

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=config["logging"]["level"],
    filename='validation.log',
    filemode=config["logging"]["filemode"],
    style=config["logging"]["style"],
    format=config["logging"]["format"],
    datefmt=config["logging"]["datefmt"]
)


DISSOLVE_UNITS = [True, False]
BENCHMARKS = ['GAUL'] # 'GDIS' to be implemented
SUBBATCHES = Path(config["path"]["llm_subbatch_dir"]).glob("*.gpkg")


def main():
    logging.info(f"Running validation script...".upper())
    combination = product(DISSOLVE_UNITS, BENCHMARKS, SUBBATCHES)
    for i, (dissolve_units, benchmark, subbatch) in enumerate(combination, start=1):

        try:
            v.validate_geometries(
                subbatch,
                benchmark,
                dissolve_units,
                config["geoprocessing"]["area_calculation_method"],
                config["path"]["emdat_archive_path"],
                config["path"]["emdat_gaul_path"],
                config["path"]["validation_output_dir"]
            )
        except Exception as e:
            logging.exception(f"Exception occurred: {e}")


if __name__ == "__main__":
    main()