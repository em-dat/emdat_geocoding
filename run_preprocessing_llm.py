import tomllib
import logging
from pathlib import Path
from validation import preprocessing as pp
from validation.io import load_emdat_archive

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=config["logging"]["level"],
    filename='preprocessing_llm.log',
    filemode=config["logging"]["filemode"],
    style=config["logging"]["style"],
    format=config["logging"]["format"],
    datefmt=config["logging"]["datefmt"]
)

def list_disno_with_gaul(emdat_archive_path: Path) -> list[str]:
    """List disno with GAUL admin units in the EM-DAT archive."""
    disno_with_gaul = load_emdat_archive(
        emdat_archive_path,
        use_columns=["DisNo."],
        geocoded_only=True
    )['DisNo.'].to_list()
    return disno_with_gaul

def main():
    disno_with_gaul = list_disno_with_gaul(config["path"]["emdat_archive_path"])
    output_dir = Path(config["path"]["batch_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pp.make_llm_geocoded_subbatches(
        csv_file_dir='Q:/Data/emdat_geocoding/GEOEMDAT',
        columns_to_keep=config["index"]["llm_columns_to_keep"],
        batch_numbers=config["index"]["batch_numbers"],
        keep_disno=disno_with_gaul,
        output_dir=output_dir,
        geometry_columns=config["index"]["llm_geom_columns"]
    )

if __name__ == '__main__':
    logging.info(f"Running preprocessing script...".upper())
    try:
        main()
    except Exception as e:
        logging.exception(f"Exception occurred: {e}")
