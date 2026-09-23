"""GADM helpers shared by the geocoding scripts.

Moved unchanged from ``run_geolocation.py`` so that ``gadm_projection.py`` can
reuse them without running the geocoding pipeline on import.
"""
import unicodedata

import geopandas as gpd
import pycountry
from rapidfuzz import fuzz, process


def read_admin(path_admin, adl):
    # GADM GPKG layers are typically named 'ADM_0', 'ADM_1', etc.
    # or 'gadm_410_0', 'gadm_410_1' depending on the exact version/download
    # For the world GPKG, they are usually 'ADM_0', 'ADM_1', ...
    layer_name = f"ADM_{adl}"
    adms = gpd.read_file(path_admin, layer=layer_name)

    adms = adms.rename(columns={'GID_0': 'iso3'})
    iso3_mapping = {'Z01': 'IND', 'Z02': 'CHN', 'Z03': 'CHN', 'Z04': 'IND',
                    'Z05': 'IND', 'Z06': 'PAK', 'Z07': 'IND', 'Z08': 'CHN',
                    'Z09': 'IND'}
    adms["iso3"] = adms["iso3"].replace(iso3_mapping)
    adms = adms[~adms["iso3"].isin(["XKO", None])]
    pyi3 = [pycountry.countries.get(alpha_3=i3) for i3 in adms.iso3]
    adms = adms[[x is not None for x in
                 pyi3]]  # som admin2 will be deleted belonging to ['China', 'India', 'Pakistan', 'Kosovo'] as they are in conflicted areas
    adms["iso2"] = [pycountry.countries.get(alpha_3=i3).alpha_2 for i3 in
                    adms.iso3]

    adms["ADMIN0"] = adms["COUNTRY"]
    if adl == 1:
        adms["ADMIN1"] = adms["NAME_1"]
    elif adl == 2:
        adms["ADMIN1"] = adms["NAME_1"]
        adms["ADMIN2"] = adms["NAME_2"]

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
        gadm1_country = gadm1[
            gadm1["COUNTRY"].map(normalize_string) == normalize_string(country)]
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

        gadm2_country = gadm2[
            gadm2["COUNTRY"].map(normalize_string) == normalize_string(country)]
        if gadm2_country.empty:
            continue

        # constrain inside Admin1
        gadm2_admin1 = gadm2_country[
            gadm2_country["ADMIN1"].map(normalize_string) == normalize_string(
                admin1_name)
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
