import os
import tomllib

import geopandas as gpd
import pandas as pd
import pycountry
from shapely import wkt
from shapely.validation import make_valid

from gadm_utils import read_admin

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

gadm_path = config["geocoding"].get("gadm_preprocessed_path") or config["geocoding"].get("gadm_path")
if not gadm_path:
    raise ValueError("GADM path not found in config.toml ([geocoding].gadm_preprocessed_path or [geocoding].gadm_path)")

gadm1 = read_admin(gadm_path, 1)


csv_folder = config["geocoding"]["geolocated_files_dir"]
all_dataframes = []

for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(csv_folder, filename)
        df = pd.read_csv(filepath)
        all_dataframes.append(df)

# Concatenate all dataframes
concatenated_output = pd.concat(all_dataframes, ignore_index=True)

print(len(concatenated_output["DisNo."].unique()))

nan_rows = concatenated_output[['geometry_osm', 'geometry_wiki', 'geometry_gadm']].isna().all(axis=1)

# Count the number of such rows
nan_count = nan_rows.sum()

# Remove those rows from the DataFrame
concatenated_output_cleaned = concatenated_output[~nan_rows]

print(len(nan_rows))
print(len(concatenated_output_cleaned))

concatenated_output_cleaned['iso3'] = concatenated_output_cleaned['DisNo.'].str[-3:]

for col in ['geometry_wiki', 'geometry_osm', 'geometry_gadm']:
    concatenated_output_cleaned[col] = concatenated_output_cleaned[col].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else x)



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


concatenated_output_cleaned = fill_gadm(concatenated_output_cleaned, gadm1)
concatenated_output_cleaned.to_csv("./data/LLMGeoDis.csv", index=False)