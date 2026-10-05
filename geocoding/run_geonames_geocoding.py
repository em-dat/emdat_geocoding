"""GeoNames geocoding of EM-DAT locations, the conventional benchmark for LLM-GeoDis.

Follows the automated part of Teber et al. (Geo-Disasters): EM-DAT location
strings are cleaned and split with regular expressions (geonames_cleaning.py),
then each location is searched in GeoNames within the event's country. The
result is one point per location, saved to [geonames].points_path.

Events are those of the validation subset ([path].gdis_disno_path). The GeoNames
free web service allows about 1,000 requests per hour, so the run takes days;
results are saved every 100 locations and a new run resumes where the previous
one stopped.

Run from the repository root: python geocoding/run_geonames_geocoding.py
"""
import logging
import tomllib
from pathlib import Path

import pandas as pd

from geonames_cleaning import process_emdat_locations
from geonames_client import geocode_event_locations

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # locations geocoded between two saves


def main():
    geonames = config["geonames"]
    if not geonames.get("username"):
        raise ValueError("Set [geonames].username in config.toml (a GeoNames account with web services enabled)")
    points_path = Path(geonames["points_path"])

    disnos = pd.read_csv(config["path"]["gdis_disno_path"], dtype=str)["DisNo."].dropna()
    emdat = pd.read_excel(config["path"]["emdat_archive_path"])
    emdat = emdat[emdat["DisNo."].isin(set(disnos))]
    logger.info(f"{len(emdat)} EM-DAT events in the validation subset")

    locations = process_emdat_locations(emdat)
    logger.info(f"{len(locations)} locations in {locations['DisNo.'].nunique()} events after cleaning")

    if points_path.exists():
        points = pd.read_csv(points_path, dtype={"DisNo.": str, "input_location": str})
    else:
        points = pd.DataFrame()
    # A few cleaned locations are empty strings, which the CSV reads back as NaN
    done = set(zip(points["DisNo."], points["input_location"].fillna(""))) if not points.empty else set()
    todo = locations[[(d, l) not in done for d, l in zip(locations["DisNo."], locations["Location"].fillna(""))]]
    logger.info(f"{len(done)} locations already geocoded, {len(todo)} to go")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = geocode_event_locations(
            todo.iloc[start:start + BATCH_SIZE],
            username=geonames["username"],
            max_requests_per_hour=geonames["max_requests_per_hour"],
            max_requests_per_day=geonames["max_requests_per_day"],
        )
        points = pd.concat([points, batch], ignore_index=True)
        points_path.parent.mkdir(parents=True, exist_ok=True)
        points.to_csv(points_path, index=False)
        logger.info(f"{len(points)} / {len(locations)} locations geocoded and saved")

    logger.info(f"GeoNames points: {points['lat'].notna().sum()} of {len(points)} locations found")


if __name__ == "__main__":
    main()
