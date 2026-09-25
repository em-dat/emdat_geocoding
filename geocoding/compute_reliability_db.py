"""Build reliability_db.csv, the input of compute_reliability.ipynb and
main_figures.ipynb, from the GADM-projected LLM-GeoDis CSVs.

One row per location, with the inputs of the reliability score:
- osm_count, gadm_count, wiki_count: 1 if the geometry is present, else 0.
  Only GADM units matched by name count as GADM (gadm_source "name_match");
  units assigned by overlap with the OSM or Wikidata geometry do not, since
  they are derived from these geometries.
- gadm_osm_overlap_pct: area(GADM & OSM) / area(GADM) * 100 (planar area in
  degrees), when both are present.
- wiki_in_gadm, wiki_in_osm: 1 if the Wikidata point lies inside the GADM
  (OSM) polygon, else 0, when both are present.

Reads the CSVs in [geocoding].projected_files_dir, writes
[geocoding].reliability_db_path. Run from the repository root:
python geocoding/compute_reliability_db.py
"""
import os
import tomllib

import numpy as np
import pandas as pd
import shapely

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

CHUNKSIZE = 2000  # rows read at a time; the LLM-GeoDis parts are several GB each
ATTRIBUTES = ["DisNo.", "name", "admin_level", "admin1", "admin2", "admin3"]
GEOMETRIES = ["geometry_osm", "geometry_gadm", "geometry_wiki"]


def reliability_inputs(chunk):
    geoms = {c: shapely.from_wkt(chunk[c].where(chunk[c].notna(), None).to_numpy())
             for c in GEOMETRIES}
    osm, gadm, wiki = geoms["geometry_osm"], geoms["geometry_gadm"], geoms["geometry_wiki"]
    if "gadm_source" in chunk.columns:
        gadm = np.where(chunk["gadm_source"].to_numpy() == "name_match", gadm, None)
    has = {"osm": ~shapely.is_missing(osm), "gadm": ~shapely.is_missing(gadm),
           "wiki": ~shapely.is_missing(wiki)}

    out = chunk[ATTRIBUTES].copy()
    for source in ["osm", "gadm", "wiki"]:
        out[f"{source}_count"] = has[source].astype(int)

    overlap = np.full(len(out), np.nan)
    both = has["gadm"] & has["osm"]
    if both.any():
        g, o = shapely.make_valid(gadm[both]), shapely.make_valid(osm[both])
        overlap[both] = shapely.area(shapely.intersection(g, o)) / shapely.area(g) * 100
    out["gadm_osm_overlap_pct"] = overlap

    for source, polygons in [("gadm", gadm), ("osm", osm)]:
        inside = np.full(len(out), np.nan)
        both = has["wiki"] & has[source]
        if both.any():
            inside[both] = shapely.within(
                wiki[both], shapely.make_valid(polygons[both])).astype(float)
        out[f"wiki_in_{source}"] = inside
    return out


def main():
    csv_folder = config["geocoding"]["projected_files_dir"]
    output_path = config["geocoding"]["reliability_db_path"]
    tables = []
    for filename in sorted(os.listdir(csv_folder)):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(csv_folder, filename)
        header = pd.read_csv(filepath, nrows=0).columns
        columns = ATTRIBUTES + GEOMETRIES + (["gadm_source"] if "gadm_source" in header else [])
        for chunk in pd.read_csv(filepath, usecols=columns, dtype=str, chunksize=CHUNKSIZE):
            tables.append(reliability_inputs(chunk))
        print(f"{filename}: done")
    reliability_db = pd.concat(tables, ignore_index=True)
    reliability_db.to_csv(output_path, index=False)
    print(f"{len(reliability_db)} locations written to {output_path}")


if __name__ == "__main__":
    main()
