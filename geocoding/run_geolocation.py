import pandas as pd
import numpy as np
import unicodedata
from rapidfuzz import fuzz, process
from openai import OpenAI
import json
import re
import osmnx as osm
import re
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from shapely.geometry import Point
from geopy.distance import geodesic
import random
import geopandas as gpd
import os 
import pycountry
from pathlib import Path
from shapely import wkt
import geopandas as gpd
import math
from shapely import wkt as sh_wkt


TOKEN = "your_token"

client = OpenAI(
    api_key=TOKEN,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)


def read_admin(path_admin,adl):
    adms=gpd.read_file(os.path.join(path_admin, "gadm_410_L%s.shp"%str(adl)))
    
    adms = adms.rename(columns={'GID_0': 'iso3'})
    iso3_mapping = {'Z01':'IND', 'Z02':'CHN', 'Z03':'CHN', 'Z04':'IND', 'Z05':'IND', 'Z06':'PAK', 'Z07':'IND', 'Z08':'CHN', 'Z09':'IND'}
    adms["iso3"] = adms["iso3"].replace(iso3_mapping)
    adms = adms[~adms["iso3"].isin(["XKO",None])]
    pyi3=[pycountry.countries.get(alpha_3=i3) for i3 in adms.iso3]
    adms=adms[[x is not None for x in pyi3]] # som admin2 will be deleted belonging to ['China', 'India', 'Pakistan', 'Kosovo'] as they are in conflicted areas
    adms["iso2"]=[pycountry.countries.get(alpha_3=i3).alpha_2 for i3 in adms.iso3]

    adms["ADMIN0"]=adms["COUNTRY"]
    if adl ==1:
        adms["ADMIN1"]=adms["NAME_1"]
    elif adl ==2:
        adms["ADMIN1"]=adms["NAME_1"]
        adms["ADMIN2"]=adms["NAME_2"]
    
    
    return adms

def normalize_string(s):
    """
    Lowercase, remove accents, and strip whitespace.
    """
    if not s:
        return ""
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s.strip()


def clean_gpt_json(raw_text):
    """
    Extract JSON from GPT response and convert to standard JSON format.
    """
    # Find the first { ... } block
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in GPT response.")
    
    json_str = match.group(0)

    # Replace smart quotes with normal quotes
    json_str = json_str.replace("“", '"').replace("”", '"').replace("’", "'")

    # Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

    return json.loads(json_str)


def parse_emdat_location(location_string, country):
    """
    Uses GPT to parse an EM-DAT Location string into hierarchical Admin1/Admin2/Admin3 JSON.
    """
    prompt = f"""
You are an expert geographer tasked with parsing disaster location strings into a hierarchical structured format.
Each location can be:
- Admin1: state/province
- Admin2: county/municipality
- Admin3: city/town/village

Rules:
1. Locations may be separated by ';', ',', or 'and'.
2. Text in parentheses indicates the higher-level admin 
   (e.g., "Bamberg (South Carolina)" → Bamberg is Admin2 in South Carolina).
3. Keep the highest-precision locations only:
   - Do not include Admin1 if a more detailed location exists inside it.
   - Do not include Admin2 if an Admin3 exists inside it.
4. Correct minor typos, missing accents, or formatting errors. Always refer to the provided country to disambiguate. Prefer common forms from Google Maps, GADM, or OSM **only if clearly recognizable within that country**. Remove generic descriptors like "area", "village", "region", or "district" **only when not part of the official name**. If unsure, keep the original name. Do not invent, remove, or add any location not in the input string. Return **only JSON** with keys "Admin1", "Admin2", "Admin3", and include Admin1 for each Admin2/Admin3.

Example:
Input: "Belford Roxo Area, Duque de Caxias, Northern Nova Iguaçu, Japeri Village, São João de Meriti, São Gonçalo Region, Mesquita, Nilópolis, Nova Iguaçu, and Queimados Municipalities (Rio de Janeiro); São Paulo, Santa Catarina, Paraná and Rio Grande do Sul"
Country: "Brazil"

Output:
{{
  "Admin1": ["São Paulo", "Santa Catarina", "Paraná", "Rio Grande do Sul"],
  "Admin2": [
    {{"name": "Belford Roxo", "Admin1": "Rio de Janeiro"}},
    {{"name": "Duque de Caxias", "Admin1": "Rio de Janeiro"}},
    {{"name": "Nova Iguaçu", "Admin1": "Rio de Janeiro"}},
    {{"name": "Japeri", "Admin1": "Rio de Janeiro"}},
    {{"name": "São João de Meriti", "Admin1": "Rio de Janeiro"}},
    {{"name": "São Gonçalo", "Admin1": "Rio de Janeiro"}},
    {{"name": "Mesquita", "Admin1": "Rio de Janeiro"}},
    {{"name": "Nilópolis", "Admin1": "Rio de Janeiro"}},
    {{"name": "Queimados", "Admin1": "Rio de Janeiro"}}
  ],
  "Admin3": []
}}

Now parse this location string: "{location_string}"
Country: "{country}"
"""


    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            stream=False,
            messages=[{"role": "user", "content": prompt}]
        )
        # GPT returns text, parse as JSON
        gpt_output = clean_gpt_json(response.choices[0].message.content.strip())
        #print(gpt_output)
        #parsed_json = json.loads(gpt_output)
        return gpt_output
    except Exception as e:
        print(f"Error parsing location with GPT: {e}")
        return None

    
def clean_location_name(name: str) -> str:
    """
    Clean a location name:
    - Remove generic descriptors (region, village, district, etc.).
    - Remove text in parentheses.
    - Strip extra spaces, trailing commas, and stray 'None' values.
    - Return a clean "Name[, Country]" string.
    """
    if not name or not isinstance(name, str):
        return None

    # Remove parenthetical content
    name = re.sub(r"\([^)]*\)", "", name)

    # Generic keywords to remove
    keywords = [
        "region", "regions", "province", "provinces", "state", "states", "division", "divisions",
        "territory", "territories", "department", "departments", "departement", "departements",
        "departamento", "departamentos", "governorate", "governorates", "district", "districts",
        "municipality", "municipalities", "municipal", "municipal district", "municipal districts",
        "city", "cities", "county", "counties", "prefecture", "prefectures", "borough", "boroughs",
        "commune", "communes", "parish", "parishes", "ward", "wards", "sector", "sectors", "zone","tower",
        "zones", "subdistrict", "subdistricts", "subdivision", "subdivisions", "locality", "localities",
        "township", "townships", "town", "towns", "village", "villages", "hamlet", "hamlets",
        "regency", "regencies", "area", "areas", "island", "islands", "isl\\.", "lake", "lakes",
        "river", "rivers", "mount", "mounts", "mountain", "mountains", "valley", "valleys",
        "peninsula", "peninsulas"
    ]
    pattern = r"\b(" + "|".join(keywords) + r")\b"
    name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    # Replace multiple spaces and commas
    name = re.sub(r"\s{2,}", " ", name)
    name = re.sub(r"\s*,\s*", ", ", name)
    name = re.sub(r"(, )+", ", ", name)  # repeated commas
    name = name.strip(", ").strip()

    # Remove any leftover 'None' or empty segments
    parts = [p.strip() for p in name.split(",") if p.strip() and p.strip().lower() != "none"]
    return ", ".join(parts) if parts else None


def try_geocode(name: str, pause: float = 0.5):
    """
    Attempts geocoding with original name, falls back to cleaned name if needed.
    Adds a small sleep to reduce rate-limit issues.
    """
    time.sleep(pause)  # reduce Nominatim 429 errors
    try:
        return osm.geocode_to_gdf(name)["geometry"].iloc[0]
    except Exception as e1:
        cleaned = clean_location_name(name)
        if cleaned != name:
            time.sleep(pause)
            try:
                return osm.geocode_to_gdf(cleaned)["geometry"].iloc[0]
            except Exception as e2:
                #print(f"OSM geocode failed for '{name}' and '{cleaned}': {e2}")
                return None
        else:
            #print(f"OSM geocode failed for '{name}': {e1}")
            return None

def geolocate_hierarchical(parsed_json, country):
    """
    Takes GPT hierarchical JSON output and geolocates each Admin1/Admin2/Admin3 entry using OSM.
    Adds country context to improve geocoding.
    Returns a dict with the same structure, including geometry (or None if not found).
    """
    result = {"Admin1": [], "Admin2": [], "Admin3": []}

    # --- Admin1 ---
    for admin1 in parsed_json.get("Admin1", []):
        if not admin1:  # skip empty
            continue
        full_name = f"{admin1}, {country}"
        geom = try_geocode(full_name)
        result["Admin1"].append({
            "name": admin1,
            "geometry": geom
        })

    # --- Admin2 ---
    for admin2 in parsed_json.get("Admin2", []):
        if isinstance(admin2, dict):
            full_name = f"{admin2['name']}, {admin2.get('Admin1', country)}"
            geom = try_geocode(full_name)
            result["Admin2"].append({
                "name": admin2["name"],
                "Admin1": admin2.get("Admin1", None),
                "geometry": geom
            })
        elif isinstance(admin2, str):
            full_name = f"{admin2}, {country}"
            geom = try_geocode(full_name)
            result["Admin2"].append({
                "name": admin2,
                "Admin1": None,
                "geometry": geom
            })

    # --- Admin3 ---
    for admin3 in parsed_json.get("Admin3", []):
        if isinstance(admin3, dict):
            full_name = f"{admin3['name']}, {admin3.get('Admin1', country)}"
            geom = try_geocode(full_name)
            result["Admin3"].append({
                "name": admin3["name"],
                "Admin1": admin3.get("Admin1", None),
                "geometry": geom
            })
        elif isinstance(admin3, str):
            full_name = f"{admin3}, {country}"
            geom = try_geocode(full_name)
            result["Admin3"].append({
                "name": admin3,
                "Admin1": None,
                "geometry": geom
            })

    return result


# --- Wikidata query function ---
def wikidata_geocode(place_name):
    """
    Resolve a place name to Wikidata QID + coordinates (returns multiple candidates)
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?item ?itemLabel ?coord ?countryLabel
    WHERE {{
      ?item rdfs:label "{place_name}"@en .
      OPTIONAL {{ ?item wdt:P625 ?coord. }}
      OPTIONAL {{ ?item wdt:P17 ?country. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    matches = []
    for r in results["results"]["bindings"]:
        coord = r.get("coord", {}).get("value")
        if coord:
            lon, lat = map(float, coord.replace("Point(", "").replace(")", "").split())
            matches.append({
                "qid": r["item"]["value"].split("/")[-1],
                "label": r["itemLabel"]["value"],
                "country": r.get("countryLabel", {}).get("value"),
                "coord": Point(lon, lat)
            })
    return matches

# --- Candidate selection function ---
def select_closest_candidate(candidates, parent_point=None, emdat_country=None):
    """
    Select the best candidate from Wikidata, ensuring it's in the EM-DAT country.
    """
    if not candidates:
        return None

    if emdat_country:
        emdat_norm = normalize_string(emdat_country)
        # Keep only candidates in the correct country
        candidates = [c for c in candidates if normalize_string(c.get("country", "")) == emdat_norm]
        if not candidates:
            # No candidates in EM-DAT country
            return None

    # If no parent point, take first candidate
    if parent_point is None:
        return candidates[0]

    # Otherwise, pick the closest candidate to the parent point
    closest = min(candidates, key=lambda c: geodesic(
        (c["coord"].y, c["coord"].x),
        (parent_point.y, parent_point.x)
    ).km)
    return closest

# --- Safe wrapper for Wikidata geocoding ---
def safe_wikidata_geocode(name, retries=3, delay=5, base_wait=1):
    """
    Wrapper with rate-limiting and retry handling.
    """
    for attempt in range(retries):
        try:
            time.sleep(base_wait)
            return wikidata_geocode(name)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = delay + random.uniform(0, 2)
                #print(f"Rate limited on '{name}', retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                #print(f"Wikidata geocode failed for '{name}': {e}")
                return []
    #print(f"Giving up on '{name}' after {retries} retries.")
    return []

# --- Hierarchical geolocation using Wikidata ---
def wikidata_geolocate_hierarchy(parsed_json, emdat_country):
    """
    Hierarchically geolocate Admin1/Admin2/Admin3 using Wikidata.
    Ensures points are only in the EM-DAT country.
    """
    result = {"Admin1": [], "Admin2": [], "Admin3": []}

    # --- Admin1 ---
    for admin1_name in parsed_json.get("Admin1", []):
        candidates = safe_wikidata_geocode(admin1_name)
        selected = select_closest_candidate(candidates, parent_point=None, emdat_country=emdat_country)
        geom = selected["coord"] if selected else None
        result["Admin1"].append({"name": admin1_name, "geometry": geom})

    # --- Admin2 ---
    for admin2 in parsed_json.get("Admin2", []):
        parent = next((a["geometry"] for a in result["Admin1"] if a["name"] == admin2["Admin1"]), None)
        candidates = safe_wikidata_geocode(admin2["name"])
        selected = select_closest_candidate(candidates, parent_point=parent, emdat_country=emdat_country)
        geom = selected["coord"] if selected else None
        result["Admin2"].append({
            "name": admin2["name"],
            "Admin1": admin2["Admin1"],
            "geometry": geom
        })

    # --- Admin3 ---
    for admin3 in parsed_json.get("Admin3", []):
        parent = next((a["geometry"] for a in result["Admin2"] 
                       if a["name"] == admin3["name"] and a["Admin1"] == admin3["Admin1"]), None)
        candidates = safe_wikidata_geocode(admin3["name"])
        selected = select_closest_candidate(candidates, parent_point=parent, emdat_country=emdat_country)
        geom = selected["coord"] if selected else None
        result["Admin3"].append({
            "name": admin3["name"],
            "Admin1": admin3["Admin1"],
            "geometry": geom
        })

    return result



def match_location_to_gadm(parsed_json, gadm1, gadm2, country):
    """
    Match GPT-parsed disaster locations to GADM Admin1 and Admin2 units.
    
    Args:
        parsed_json: dict with keys Admin1, Admin2, Admin3 from GPT
        gadm1, gadm2: GeoDataFrames containing GADM data
        country: str, country name from EM-DAT
    
    Returns:
        dict with matched GADM names and geometries (Admin1/Admin2 only)
    """
    results = {"Admin1": [], "Admin2": []}

    # --- Admin1 matching ---
    for admin1_name in parsed_json.get("Admin1", []):
        gadm1_country = gadm1[gadm1["COUNTRY"].map(normalize_string) == normalize_string(country)]
        if gadm1_country.empty:
            continue

        match, score, _ = process.extractOne(
            normalize_string(admin1_name),
            gadm1_country["ADMIN1"].map(normalize_string),
            scorer=fuzz.ratio,
        )

        if score > 85:
            row = gadm1_country[
                gadm1_country["ADMIN1"].map(normalize_string) == match
            ].iloc[0]
            results["Admin1"].append({
                "name": admin1_name,
                "gadm_admin1": row["ADMIN1"],
                "geometry": row["geometry"]
            })

    # --- Admin2 matching (strict: inside Admin1) ---
    for admin2 in parsed_json.get("Admin2", []):
        admin2_name = admin2["name"]
        admin1_name = admin2["Admin1"]

        gadm2_country = gadm2[gadm2["COUNTRY"].map(normalize_string) == normalize_string(country)]
        if gadm2_country.empty:
            continue

        # constrain inside Admin1
        gadm2_admin1 = gadm2_country[
            gadm2_country["ADMIN1"].map(normalize_string) == normalize_string(admin1_name)
        ]
        if gadm2_admin1.empty:
            continue

        match, score, _ = process.extractOne(
            normalize_string(admin2_name),
            gadm2_admin1["ADMIN2"].map(normalize_string),
            scorer=fuzz.ratio,
        )

        if score > 85:
            row = gadm2_admin1[
                gadm2_admin1["ADMIN2"].map(normalize_string) == match
            ].iloc[0]
            results["Admin2"].append({
                "name": admin2_name,
                "Admin1": admin1_name,
                "gadm_admin1": row["ADMIN1"],
                "gadm_admin2": row["ADMIN2"],
                "geometry": row["geometry"]
            })

    return results


def gadm_geolocate_hierarchy(parsed_json, emdat_country, gadm1, gadm2):
    """
    Hierarchically match GPT-parsed JSON to GADM1/GADM2 only.
    Returns matched names and geometries in a structured dict.
    Admin3 remains empty since we don't have GADM3.
    """
    # Match Admin1/Admin2
    gadm_result = match_location_to_gadm(parsed_json, gadm1, gadm2, emdat_country)

    # Prepare structured output
    result = {"Admin1": [], "Admin2": [], "Admin3": []}  # keep Admin3 key for compatibility

    # --- Admin1 ---
    for admin1 in gadm_result.get("Admin1", []):
        result["Admin1"].append({
            "name": admin1["name"],
            "gadm_admin1": admin1["gadm_admin1"],
            "geometry": admin1["geometry"]
        })

    # --- Admin2 ---
    for admin2 in gadm_result.get("Admin2", []):
        result["Admin2"].append({
            "name": admin2["name"],
            "Admin1": admin2["Admin1"],
            "gadm_admin1": admin2["gadm_admin1"],
            "gadm_admin2": admin2["gadm_admin2"],
            "geometry": admin2["geometry"]
        })

    # --- Admin3 ---
    # Keep empty list; no GADM3
    result["Admin3"] = []

    return result


def merge_location_geometries_strict(osm_json, wiki_json, gadm_json):
    """
    Merge geometries by taking the union of names across OSM, Wikidata, and GADM per admin level.
    Returns a DataFrame with one row per unique name per level and separate geometry_* columns.
    """
    rows = []
    for level in ["Admin1", "Admin2", "Admin3"]:
        # collect unique names from all sources
        names = set()
        for e in gadm_json.get(level, []):
            n = e.get("name")
            if n:
                names.add(n)
        for e in osm_json.get(level, []):
            n = e.get("name")
            if n:
                names.add(n)
        for e in wiki_json.get(level, []):
            n = e.get("name")
            if n:
                names.add(n)

        for name in sorted(names):
            # find matching entries (first match) in each source
            gadm_entry = next((e for e in gadm_json.get(level, []) if e.get("name") == name), None)
            osm_entry = next((e for e in osm_json.get(level, []) if e.get("name") == name), None)
            wiki_entry = next((e for e in wiki_json.get(level, []) if e.get("name") == name), None)

            geom_gadm = gadm_entry.get("geometry") if gadm_entry else None
            geom_osm = osm_entry.get("geometry") if osm_entry else None
            geom_wiki = wiki_entry.get("geometry") if wiki_entry else None

            # fallback strategies for admin labels:
            admin1 = None
            admin2 = None
            if gadm_entry:
                admin1 = gadm_entry.get("gadm_admin1") if "gadm_admin1" in gadm_entry else (gadm_entry.get("name") if level == "Admin1" else None)
                admin2 = gadm_entry.get("gadm_admin2") if "gadm_admin2" in gadm_entry else None
            else:
                # try to pull Admin1 from OSM or Wiki entries (they sometimes include Admin1)
                if osm_entry:
                    admin1 = osm_entry.get("Admin1") or admin1
                if wiki_entry:
                    admin1 = wiki_entry.get("Admin1") or admin1

            rows.append({
                "name": name,
                "admin_level": level,
                "admin1": admin1,
                "admin2": admin2,
                "admin3": None,
                "geometry_gadm": geom_gadm,
                "geometry_osm": geom_osm,
                "geometry_wiki": geom_wiki
            })
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=[
            "name", "admin_level", "admin1", "admin2", "admin3",
            "geometry_gadm", "geometry_osm", "geometry_wiki"
        ])
    return df

gadm1 = read_admin("/path_to_gadm/GADM",1)
gadm2 = read_admin("/path_to_gadm/GADM",2)


final_rows = []
geocode_cache = {}  # key -> full cached row dict

def make_cache_key(level, name, admin1=None, admin2=None, admin3=None, country=None):
    return (level, name, admin1, admin2, admin3, country)

def get_cached_row(level, name, admin1=None, admin2=None, admin3=None, country=None):
    key = make_cache_key(level, name, admin1, admin2, admin3, country)
    return geocode_cache.get(key)

def store_cached_row(level, row_dict, admin1=None, admin2=None, admin3=None, country=None):
    """
    Store a copy of the row in cache as a plain dict for robustness.
    """
    key = make_cache_key(level, row_dict.get("name") if isinstance(row_dict, dict) else row_dict["name"],
                         admin1, admin2, admin3, country)
    # convert pandas Series -> dict if needed
    if hasattr(row_dict, "to_dict"):
        to_store = row_dict.to_dict()
    elif isinstance(row_dict, dict):
        to_store = row_dict.copy()
    else:
        # fallback: try to coerce
        try:
            to_store = dict(row_dict)
        except Exception:
            to_store = {"name": str(row_dict)}
    geocode_cache[key] = to_store


# --- Folders ---
input_dir = Path("original_files")
output_dir = Path("geolocated_files")
log_dir = Path("geolocated_logs")
output_dir.mkdir(exist_ok=True)
log_dir.mkdir(exist_ok=True)


def retry_parse_emdat_location(location_string, country, retries=5, base_wait=1, max_wait=30):
    """
    Retry wrapper that treats a None response as failure and retries with exponential backoff.
    Returns parsed JSON or None after exhausted retries.
    """
    for attempt in range(retries):
        try:
            resp = parse_emdat_location(location_string, country)
            if resp:
                return resp
            else:
                wait_time = min(base_wait * (2 ** attempt), max_wait)
                print(f"[parse] empty/None response for '{location_string[:60]}...' attempt {attempt+1}/{retries}, retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
        except Exception as e:
            errstr = str(e)
            if "429" in errstr or "Too Many Requests" in errstr:
                wait_time = min(base_wait * (2 ** attempt) + random.uniform(0, 2), max_wait)
                print(f"[parse] rate limited on attempt {attempt+1}/{retries} — waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
            else:
                print(f"[parse] error parsing location: {e}")
                return None
    print(f"[parse] giving up after {retries} retries for: {location_string[:80]}")
    return None


def process_skipped_rows(file, skipped_disnos=None, prefix="retry_"):
    """
    Process all rows in the specified file.
    The file already contains only the remaining (unprocessed) EM-DAT entries.
    """
    try:
        emdat = pd.read_excel(file)
    except Exception as e:
        print(f"Could not read {file.name}: {e}")
        return

    required_cols = {"Location", "Country", "DisNo."}
    if not required_cols.issubset(emdat.columns):
        print(f"Skipping {file.name}, missing columns {required_cols - set(emdat.columns)}")
        return

    # Use all rows in the file (already filtered externally)
    emdat_filtered = emdat.copy()
    print(f"Processing {file.name}, {len(emdat_filtered)} rows.")

    final_rows = []
    skipped_disnos_new = []

    # --- Process each row ---
    for idx, row in emdat_filtered.iterrows():
        try:
            location_str = row["Location"]
            country = row["Country"]
            dis_no = row["DisNo."]

            # Step 1: parse the EM-DAT location (with retries)
            emdat_json = retry_parse_emdat_location(location_str, country)
            if emdat_json is None:
                print(f"[WARN] Could not parse location for DisNo {dis_no}. Skipping.")
                skipped_disnos_new.append(dis_no)
                continue

            # Step 2: check cache first
            cached_all = True
            cached_rows = {"Admin1": [], "Admin2": [], "Admin3": []}

            for level in ["Admin1", "Admin2", "Admin3"]:
                for loc in emdat_json.get(level, []):
                    if isinstance(loc, dict):
                        name_val = loc.get("name")
                        admin1_val = loc.get("Admin1")
                        admin2_val = loc.get("Admin2")
                        admin3_val = loc.get("Admin3")
                    else:
                        name_val = loc
                        admin1_val = admin2_val = admin3_val = None

                    cached_row = get_cached_row(level, name_val, admin1_val, admin2_val, admin3_val, country)
                    if cached_row:
                        cached_rows[level].append(cached_row)
                    else:
                        cached_all = False

            # Step 3: geocode only if needed
            if not cached_all:
                osm_json = geolocate_hierarchical(emdat_json, country)
                gadm_json = gadm_geolocate_hierarchy(emdat_json, country, gadm1, gadm2)
                wiki_json = wikidata_geolocate_hierarchy(emdat_json, country)

                location_df = merge_location_geometries_strict(osm_json, wiki_json, gadm_json)

                if location_df.empty:
                    print(f"[WARN] No geocoding results for DisNo {dis_no}. Skipping.")
                    skipped_disnos_new.append(dis_no)
                    continue

                # store in cache
                for _, loc_row in location_df.iterrows():
                    store_cached_row(
                        loc_row["admin_level"],
                        loc_row.to_dict(),
                        admin1=loc_row.get("admin1"),
                        admin2=loc_row.get("admin2"),
                        admin3=loc_row.get("admin3"),
                        country=country
                    )
            else:
                rows_list = []
                for level in ["Admin1", "Admin2", "Admin3"]:
                    for cached_row in cached_rows[level]:
                        new_row = cached_row.copy()
                        new_row["DisNo."] = dis_no
                        rows_list.append(new_row)
                if rows_list:
                    location_df = pd.DataFrame(rows_list)
                else:
                    print(f"[WARN] cached_all True but no cached rows found for DisNo {dis_no}.")
                    skipped_disnos_new.append(dis_no)
                    continue

            location_df["DisNo."] = dis_no

            if not location_df.empty:
                final_rows.append(location_df)
            else:
                skipped_disnos_new.append(dis_no)

        except Exception as e:
            print(f"Error in row {idx} of {file.name}: {e}")
            skipped_disnos_new.append(row.get("DisNo."))
            continue

    # --- Save outputs for this file ---
    if final_rows:
        final_df = pd.concat(final_rows, ignore_index=True)

        cols = ["DisNo.", "name", "admin_level", "admin1", "admin2", "admin3",
                "geometry_osm", "geometry_gadm", "geometry_wiki"]
        for c in cols:
            if c not in final_df.columns:
                final_df[c] = None

        for col in ["geometry_osm", "geometry_gadm", "geometry_wiki"]:
            final_df[col] = final_df[col].apply(lambda g: (g.wkt if hasattr(g, "wkt") else (g if isinstance(g, str) else None)))

        csv_file = output_dir / f"{prefix}{file.stem}.csv"
        final_df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file.name} (rows: {len(final_df)})")

        for geom_col, suffix in [
            ("geometry_osm", "_osm.gpkg"),
            ("geometry_gadm", "_gadm.gpkg"),
            ("geometry_wiki", "_wiki.gpkg")
        ]:
            if final_df[geom_col].notna().any():
                gdf = gpd.GeoDataFrame(
                    final_df.drop(columns=["geometry_osm", "geometry_gadm", "geometry_wiki"]),
                    geometry=final_df[geom_col].apply(lambda x: sh_wkt.loads(x) if pd.notna(x) else None),
                    crs="EPSG:4326"
                )
                gpkg_file = output_dir / f"{prefix}{file.stem}{suffix}"
                gdf2 = gdf[gdf.geometry.notna()].copy()
                if not gdf2.empty:
                    gdf2.to_file(gpkg_file, driver="GPKG")
                    print(f"Saved {gpkg_file.name} (features: {len(gdf2)})")
                else:
                    print(f"No valid geometries to save for {gpkg_file.name}")
    else:
        print("No final rows to save for this file.")

    if skipped_disnos_new:
        skip_file = log_dir / f"{prefix}skipped_{file.stem}.txt"
        with open(skip_file, "w") as f:
            for dis in skipped_disnos_new:
                f.write(str(dis) + "\n")
        print(f"Skipped {len(skipped_disnos_new)} rows, saved to {skip_file.name}")


# --- Process all remaining .xlsx files directly ---
print("Processing remaining EM-DAT files from:", input_dir)

for xlsx_file_path in input_dir.glob("*.xlsx"):
    print(f"Processing {xlsx_file_path.name}")
    process_skipped_rows(xlsx_file_path)
