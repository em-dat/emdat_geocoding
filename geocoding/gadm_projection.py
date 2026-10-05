import os
import tomllib

import geopandas as gpd
import pandas as pd
import pycountry
from rapidfuzz import fuzz
from shapely import wkt
from shapely.validation import make_valid

from gadm_utils import (EMDAT_TO_GADM_ISO3, match_location_to_gadm,
                        normalize_string, read_admin)

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

gadm_path = config["geocoding"].get("gadm_preprocessed_path") or config["geocoding"].get("gadm_path")
if not gadm_path:
    raise ValueError("GADM path not found in config.toml ([geocoding].gadm_preprocessed_path or [geocoding].gadm_path)")

gadm1 = read_admin(gadm_path, 1)
gadm2 = read_admin(gadm_path, 2)


csv_folder = config["geocoding"]["geolocated_files_dir"]
output_folder = config["geocoding"]["projected_files_dir"]

CHUNKSIZE = 2000  # rows read at a time; the LLM-GeoDis parts are several GB each
NAME_MATCH_THRESHOLD = 85  # same threshold as in match_location_to_gadm

_admin1_matches = {}
_admin2_matches = {}
_gadm2_names = {}


def admin1_match(name, iso3):
    key = (iso3, name)
    if key not in _admin1_matches:
        hits = match_location_to_gadm({"Admin1": [name]}, gadm1, gadm2, iso3)["Admin1"]
        _admin1_matches[key] = hits[0] if hits else None
    return _admin1_matches[key]


def admin2_matches(name, iso3):
    """GADM Admin2 units matching `name`, searched in every Admin1 of the
    country (the parent Admin1 parsed by GPT is not kept in the output)."""
    key = (iso3, name)
    if key not in _admin2_matches:
        gadm_iso3 = EMDAT_TO_GADM_ISO3.get(iso3, iso3)
        if gadm_iso3 not in _gadm2_names:
            country = gadm2[gadm2['iso3'] == gadm_iso3]
            _gadm2_names[gadm_iso3] = (country['ADMIN2'].map(normalize_string).tolist(),
                                       country['ADMIN1'].tolist())
        names, parents = _gadm2_names[gadm_iso3]
        query = normalize_string(name)
        candidate_parents = sorted({p for n, p in zip(names, parents)
                                    if fuzz.ratio(query, n) > NAME_MATCH_THRESHOLD})
        parsed_json = {"Admin2": [{"name": name, "Admin1": p} for p in candidate_parents]}
        _admin2_matches[key] = match_location_to_gadm(parsed_json, gadm1, gadm2, iso3)["Admin2"]
    return _admin2_matches[key]


def pick_by_overlap(candidates, geom):
    """Among same-name GADM units, keep the one overlapping `geom` most
    (the only one containing it if `geom` is a point)."""
    if not hasattr(geom, "geom_type"):
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == 'Point':
        containing = [c for c in candidates if c['geometry'].intersects(geom)]
        return containing[0] if len(containing) == 1 else None
    areas = [geom.intersection(c['geometry']).area for c in candidates]
    best = max(range(len(candidates)), key=areas.__getitem__)
    return candidates[best] if areas[best] > 0 else None


def rematch_gadm(df):
    """Derive the GADM unit of each Admin1/Admin2 row from its name.

    Names are matched with match_location_to_gadm. The parent Admin1 parsed
    by GPT is not kept in the output, so Admin2 names are searched in every
    Admin1 of the country; same-name units are resolved with the location's
    OSM (or Wikidata) geometry, otherwise with the stored Admin1. Rows without
    a name match, and Admin3 rows (GADM Admin3 units are not used), get no
    GADM unit here and are left to fill_gadm.
    """
    df = df.copy()
    df['gadm_source'] = None
    for col in ['admin1', 'admin2', 'geometry_gadm', 'gadm_source']:
        df[col] = df[col].astype(object)

    for index, row in df.iterrows():
        level = row['admin_level']
        hit = None
        if level == 'Admin1':
            hit = admin1_match(row['name'], row['iso3'])
        elif level == 'Admin2':
            candidates = admin2_matches(row['name'], row['iso3'])
            if len(candidates) == 1:
                hit = candidates[0]
            elif candidates:
                geom = row['geometry_osm'] if pd.notna(row['geometry_osm']) else row['geometry_wiki']
                hit = pick_by_overlap(candidates, geom)
                if hit is None:
                    same_parent = [c for c in candidates if c['gadm_admin1'] == row['admin1']]
                    hit = same_parent[0] if len(same_parent) == 1 else None
        elif level != 'Admin3':
            continue

        if hit:
            df.at[index, 'admin1'] = hit['gadm_admin1']
            df.at[index, 'admin2'] = hit.get('gadm_admin2')
            df.at[index, 'geometry_gadm'] = hit['geometry']
            df.at[index, 'gadm_source'] = 'name_match'
        else:
            if level != 'Admin3':
                df.at[index, 'admin1'] = None
                df.at[index, 'admin2'] = None
            df.at[index, 'geometry_gadm'] = None
    return df


def fill_gadm(df, gadm1):
    df = df.copy()
    
    for index, row in df[df['geometry_gadm'].isna()].iterrows():

        current_geom = row['geometry_osm'] if pd.notna(row['geometry_osm']) else row['geometry_wiki']

        if current_geom is None:
            continue

        # Fix invalid geometries
        if not current_geom.is_valid:
            current_geom = make_valid(current_geom)

        gadm_filtered = gadm1[gadm1['iso3'] == row['iso3']].copy()
        gadm_filtered['geometry'] = gadm_filtered['geometry'].apply(
            lambda g: make_valid(g) if not g.is_valid else g
        )

        if gadm_filtered.empty:
            continue

        if current_geom.geom_type == 'Point':
            match = gadm_filtered[gadm_filtered['geometry'].apply(lambda g: current_geom.intersects(g))]
            if not match.empty:
                best = match.iloc[0]
                df.at[index, 'admin1'] = best['NAME_1']
                df.at[index, 'geometry_gadm'] = best['geometry']
            else:
                # Nearest neighbor fallback
                distances = gadm_filtered['geometry'].apply(lambda g: current_geom.distance(g))
                nearest_idx = distances.idxmin()
                print(f"  -> nearest fallback for {row['name']} ({row['iso3']}): {gadm_filtered.at[nearest_idx, 'NAME_1']}")
                df.at[index, 'admin1'] = gadm_filtered.at[nearest_idx, 'NAME_1']
                df.at[index, 'geometry_gadm'] = gadm_filtered.at[nearest_idx, 'geometry']
        else:
            intersections = gadm_filtered['geometry'].apply(lambda g: current_geom.intersection(g))
            intersection_areas = intersections.apply(lambda g: g.area)
            max_idx = intersection_areas.idxmax()
            if intersection_areas[max_idx] > 0:
                df.at[index, 'admin1'] = gadm1.at[max_idx, 'NAME_1']
                df.at[index, 'geometry_gadm'] = gadm1.at[max_idx, 'geometry']
            else:
                # Nearest neighbor fallback
                distances = gadm_filtered['geometry'].apply(lambda g: current_geom.distance(g))
                nearest_idx = distances.idxmin()
                print(f"  -> nearest fallback for {row['name']} ({row['iso3']}): {gadm_filtered.at[nearest_idx, 'NAME_1']}")
                df.at[index, 'admin1'] = gadm_filtered.at[nearest_idx, 'NAME_1']
                df.at[index, 'geometry_gadm'] = gadm_filtered.at[nearest_idx, 'geometry']
    
    return df


def project_file(filepath, output_path):
    """Re-match, then project to GADM, one geocoded CSV, chunk by chunk."""
    n_rows = n_kept = 0
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=CHUNKSIZE)):
        n_rows += len(chunk)

        # Remove rows without any geometry
        nan_rows = chunk[['geometry_osm', 'geometry_wiki', 'geometry_gadm']].isna().all(axis=1)
        chunk = chunk[~nan_rows].copy()
        n_kept += len(chunk)

        chunk['iso3'] = chunk['DisNo.'].str[-3:]

        for col in ['geometry_wiki', 'geometry_osm', 'geometry_gadm']:
            chunk[col] = chunk[col].apply(
                lambda x: wkt.loads(x) if isinstance(x, str) else x)

        chunk = rematch_gadm(chunk)
        missing_before = chunk['geometry_gadm'].isna()
        chunk = fill_gadm(chunk, gadm1)
        chunk.loc[missing_before & chunk['geometry_gadm'].notna(), 'gadm_source'] = 'overlap'

        chunk.to_csv(output_path, mode='w' if i == 0 else 'a', header=(i == 0), index=False)
    print(f"{os.path.basename(filepath)}: {n_rows} rows read, {n_kept} written to {output_path}")


os.makedirs(output_folder, exist_ok=True)
for filename in sorted(os.listdir(csv_folder)):
    if filename.endswith(".csv"):
        project_file(os.path.join(csv_folder, filename),
                     os.path.join(output_folder, filename))
