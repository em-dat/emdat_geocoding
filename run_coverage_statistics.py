"""Coverage of the geocoding methods: share of events each method locates.

The scope is the events attempted by both LLM-GeoDis ([geocoding].location_file_path)
and the GeoNames run ([geonames].points_path, GDIS-listed events), with an EM-DAT
GAUL footprint, in the years covered by all sources (2000-2018: LLM-GeoDis starts
in 2000, GDIS ends in 2018).

An event is located by a method if at least one of its locations has a geometry
from that method. The LLM-GeoDis files keep only located locations, so location
counts are given without a rate for LLM-GeoDis; the GeoNames location rate uses
the locations of the regex split, which differ from the GPT-4o split.

Inputs: reliability_db.csv ([geocoding].reliability_db_path) for the OSM and
Wikidata sources, and geonames_benchmark_llm_points.csv (run_geonames_benchmark.py)
for the GADM units and their gadm_source. Output: coverage_by_method.csv in the
configured validation_output_dir.
"""

import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from validation.io import load_emdat_archive

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

YEARS = (2000, 2018)


def main():
    output_dir = Path(config["path"]["validation_output_dir"])
    archive = load_emdat_archive(config["path"]["emdat_archive_path"], use_columns=["DisNo."],
                                 min_year=YEARS[0], max_year=YEARS[1], geocoded_only=True)
    llm_input = pd.read_csv(config["geocoding"]["location_file_path"], usecols=["DisNo."], dtype=str)
    geonames = pd.read_csv(config["geonames"]["points_path"], dtype={"DisNo.": str})
    scope = set(archive["DisNo."]) & set(llm_input["DisNo."]) & set(geonames["DisNo."])
    logging.info(f"Coverage scope: {len(scope)} events")

    rel = pd.read_csv(config["geocoding"]["reliability_db_path"], dtype={"DisNo.": str},
                      usecols=["DisNo.", "osm_count", "wiki_count"])
    gadm = pd.read_csv(output_dir / "geonames_benchmark_llm_points.csv", dtype=str,
                       usecols=["DisNo.", "gadm_source"])
    geonames, rel, gadm = (d[d["DisNo."].isin(scope)] for d in (geonames, rel, gadm))
    located = {  # method: one row per located location
        "LLM-GeoDis (any source)": rel,
        "LLM-GeoDis (GADM)": gadm,
        "LLM-GeoDis (GADM, matched by name)": gadm[gadm["gadm_source"] == "name_match"],
        "LLM-GeoDis (OSM)": rel[rel["osm_count"] > 0],
        "LLM-GeoDis (Wiki)": rel[rel["wiki_count"] > 0],
        "GeoDisasters (GeoNames)": geonames[geonames["lat"].notna()],
    }
    coverage = pd.DataFrame([
        dict(method=method, events_in_scope=len(scope), events_located=d["DisNo."].nunique(),
             locations_attempted=len(geonames) if "GeoNames" in method else np.nan,
             locations_located=len(d))
        for method, d in located.items()])
    coverage["event_rate"] = coverage["events_located"] / coverage["events_in_scope"]
    coverage["location_rate"] = coverage["locations_located"] / coverage["locations_attempted"]
    coverage.to_csv(output_dir / "coverage_by_method.csv", index=False)
    logging.info("Coverage statistics done")


if __name__ == "__main__":
    logging.info("Running coverage statistics...".upper())
    try:
        main()
    except Exception as e:
        logging.exception(f"Exception occurred: {e}")
