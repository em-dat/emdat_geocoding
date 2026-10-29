"""GADM helpers shared by the geocoding scripts.

Used by ``run_geolocation.py`` and ``gadm_projection.py``; kept in a separate
module so they can be imported without running the geocoding pipeline.
"""

import unicodedata
import pandas as pd
import geopandas as gpd
import pycountry
from rapidfuzz import fuzz, process
import re

# EM-DAT ISO codes whose territory GADM files under another ISO3 code
EMDAT_TO_GADM_ISO3 = {
    "SPI": "ESP",  # Canary Islands
    "SCG": "SRB",  # Serbia Montenegro
}


def read_admin(path_admin, adl):
    # GADM GPKG layers are typically named 'ADM_0', 'ADM_1', etc.
    # or 'gadm_410_0', 'gadm_410_1' depending on the exact version/download
    # For the world GPKG, they are usually 'ADM_0', 'ADM_1', ...
    layer_name = f"ADM_{adl}"
    adms = gpd.read_file(path_admin, layer=layer_name)

    adms = adms.rename(columns={"GID_0": "iso3"})
    iso3_mapping = {
        "Z01": "IND",
        "Z02": "CHN",
        "Z03": "CHN",
        "Z04": "IND",
        "Z05": "IND",
        "Z06": "PAK",
        "Z07": "IND",
        "Z08": "CHN",
        "Z09": "IND",
    }
    adms["iso3"] = adms["iso3"].replace(iso3_mapping)
    adms = adms[~adms["iso3"].isin(["XKO", None])]
    pyi3 = [pycountry.countries.get(alpha_3=i3) for i3 in adms.iso3]
    adms = adms[
        [x is not None for x in pyi3]
    ]  # som admin2 will be deleted belonging to ['China', 'India', 'Pakistan', 'Kosovo'] as they are in conflicted areas
    adms["iso2"] = [pycountry.countries.get(alpha_3=i3).alpha_2 for i3 in adms.iso3]

    adms["ADMIN0"] = adms["COUNTRY"]
    if adl == 1:
        adms["ADMIN1"] = adms["NAME_1"]
    elif adl == 2:
        adms["ADMIN1"] = adms["NAME_1"]
        adms["ADMIN2"] = adms["NAME_2"]

    return adms


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
        "region",
        "regions",
        "province",
        "provinces",
        "state",
        "states",
        "division",
        "divisions",
        "territory",
        "territories",
        "department",
        "departments",
        "departement",
        "departements",
        "departamento",
        "departamentos",
        "governorate",
        "governorates",
        "district",
        "districts",
        "municipality",
        "municipalities",
        "municipal",
        "municipal district",
        "municipal districts",
        "city",
        "cities",
        "county",
        "counties",
        "prefecture",
        "prefectures",
        "borough",
        "boroughs",
        "commune",
        "communes",
        "parish",
        "parishes",
        "ward",
        "wards",
        "sector",
        "sectors",
        "zone",
        "tower",
        "zones",
        "subdistrict",
        "subdistricts",
        "subdivision",
        "subdivisions",
        "locality",
        "localities",
        "township",
        "townships",
        "town",
        "towns",
        "village",
        "villages",
        "hamlet",
        "hamlets",
        "regency",
        "regencies",
        "area",
        "areas",
        "island",
        "islands",
        "isl\\.",
        "lake",
        "lakes",
        "river",
        "rivers",
        "mount",
        "mounts",
        "mountain",
        "mountains",
        "valley",
        "valleys",
        "peninsula",
        "peninsulas",
        "oblast",
        "krai",
        "republic",
        "federal",
        "sheng",
        "ziziqu",
        "autonomous",
    ]
    pattern = r"\b(" + "|".join(keywords) + r")\b"
    name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    # Replace multiple spaces and commas
    name = re.sub(r"\s{2,}", " ", name)
    name = re.sub(r"\s*,\s*", ", ", name)
    name = re.sub(r"(, )+", ", ", name)  # repeated commas
    name = name.strip(", ").strip()

    # Remove any leftover 'None' or empty segments
    parts = [
        p.strip() for p in name.split(",") if p.strip() and p.strip().lower() != "none"
    ]
    return ", ".join(parts) if parts else None


def normalize_string(s):
    """
    Lowercase, remove accents, and strip whitespace.
    """
    if not s:
        return ""
    s = s.lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    return s.strip()


def match_location_to_gadm_new(parsed_json, gadm1, gadm2, iso3):
    """
    Match GPT-parsed disaster locations to GADM Admin1 and Admin2 units.

    Args:
        parsed_json: dict with keys Admin1, Admin2, Admin3 from GPT
        gadm1, gadm2: GeoDataFrames containing GADM data
        iso3: str, ISO3 code of the country from EM-DAT (column ISO)

    Returns:
        dict with matched GADM names and geometries (Admin1/Admin2 only)
    """
    results = {"Admin1": [], "Admin2": []}
    iso3 = EMDAT_TO_GADM_ISO3.get(iso3, iso3)

    # --- Admin1 matching ---
    for admin1_name in parsed_json.get("Admin1", []) or []:
        gadm1_country = gadm1[gadm1["iso3"] == iso3]
        admin1_name_raw = parsed_json["name"]
        if gadm1_country is None or gadm1_country.empty:
            continue

        # Build a list of all possible names for each row
        admin1_candidates = []

        for idx, row in gadm1_country.iterrows():
            # ADMIN1 itself
            names = [row["ADMIN1"]]

            # Add VARNAME alternatives
            if pd.notna(row["VARNAME_1"]):
                names.extend(row["VARNAME_1"].split("|"))

            for name in names:
                admin1_candidates.append({"name": name, "index": idx})

        # Extract just the names for fuzzy matching
        candidate_names = [
            clean_location_name(normalize_string(x["name"])) for x in admin1_candidates
        ]

        result = process.extractOne(
            clean_location_name(normalize_string(admin1_name)),
            candidate_names,
            scorer=fuzz.ratio,
        )

        match, score = result[:2] if result else (None, 0)

        if score >= 80:
            # Find the original GADM row that the match came from
            matched_candidate = admin1_candidates[candidate_names.index(match)]
            row = gadm1_country.loc[matched_candidate["index"]]
            results["Admin1"].append(
                {
                    "name": admin1_name,
                    "gadm_admin1": row["ADMIN1"],
                    "geometry": row["geometry"],
                }
            )
        else:
            result = process.extractOne(
                clean_location_name(normalize_string(admin1_name_raw)),
                candidate_names,
                scorer=fuzz.ratio,
            )

            match, score = result[:2] if result else (None, 0)
            if score >= 80:
                # Find the original GADM row that the match came from
                matched_candidate = admin1_candidates[candidate_names.index(match)]
                row = gadm1_country.loc[matched_candidate["index"]]
                results["Admin1"].append(
                    {
                        "name": admin1_name,
                        "gadm_admin1": row["ADMIN1"],
                        "geometry": row["geometry"],
                    }
                )

    # --- Admin2 matching (strict: inside Admin1) ---
    for admin2 in parsed_json.get("Admin2", []):
        if admin2 is None:
            continue

        admin2_name = admin2
        admin1_name = parsed_json.get("Admin1", [])[0]

        gadm2_country = gadm2[gadm2["iso3"] == iso3]
        if gadm2_country.empty:
            continue

        # constrain inside Admin1
        gadm2_admin1 = gadm2_country[
            gadm2_country["ADMIN1"].map(normalize_string).map(clean_location_name)
            == clean_location_name(normalize_string(admin1_name))
        ]
        if gadm2_admin1.empty:
            continue

        # Build a list of all possible names for each row
        admin2_candidates = []

        for idx, row in gadm2_admin1.iterrows():
            # ADMIN2 itself
            names = [row["ADMIN2"]]

            # Add VARNAME alternatives
            if pd.notna(row["VARNAME_2"]):
                names.extend(row["VARNAME_2"].split("|"))

            for name in names:
                admin2_candidates.append({"name": name, "index": idx})

        # Extract just the names for fuzzy matching
        candidate_names = [
            clean_location_name(normalize_string(x["name"])) for x in admin2_candidates
        ]

        result = process.extractOne(
            clean_location_name(normalize_string(admin2_name)),
            candidate_names,
            scorer=fuzz.ratio,
        )

        match, score = result[:2] if result else (None, 0)

        if score >= 80:
            # Find the original GADM row that the match came from
            matched_candidate = admin2_candidates[candidate_names.index(match)]
            row = gadm2_admin1.loc[matched_candidate["index"]]
            results["Admin2"].append(
                {
                    "name": admin2_name,
                    "Admin1": admin1_name,
                    "gadm_admin1": row["ADMIN1"],
                    "gadm_admin2": row["ADMIN2"],
                    "geometry": row["geometry"],
                }
            )

    return results


def match_location_to_gadm(parsed_json, gadm1, gadm2, iso3):
    """
    Match GPT-parsed disaster locations to GADM Admin1 and Admin2 units.

    Args:
        parsed_json: dict with keys Admin1, Admin2, Admin3 from GPT
        gadm1, gadm2: GeoDataFrames containing GADM data
        iso3: str, ISO3 code of the country from EM-DAT (column ISO)

    Returns:
        dict with matched GADM names and geometries (Admin1/Admin2 only)
    """
    results = {"Admin1": [], "Admin2": []}
    iso3 = EMDAT_TO_GADM_ISO3.get(iso3, iso3)

    # --- Admin1 matching ---
    for admin1_name in parsed_json.get("Admin1", []):
        gadm1_country = gadm1[gadm1["iso3"] == iso3]
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
            results["Admin1"].append(
                {
                    "name": admin1_name,
                    "gadm_admin1": row["ADMIN1"],
                    "geometry": row["geometry"],
                }
            )

    # --- Admin2 matching (strict: inside Admin1) ---
    for admin2 in parsed_json.get("Admin2", []):
        admin2_name = admin2["name"]
        admin1_name = admin2["Admin1"]

        gadm2_country = gadm2[gadm2["iso3"] == iso3]
        if gadm2_country.empty:
            continue

        # constrain inside Admin1
        gadm2_admin1 = gadm2_country[
            gadm2_country["ADMIN1"].map(normalize_string)
            == normalize_string(admin1_name)
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
            results["Admin2"].append(
                {
                    "name": admin2_name,
                    "Admin1": admin1_name,
                    "gadm_admin1": row["ADMIN1"],
                    "gadm_admin2": row["ADMIN2"],
                    "geometry": row["geometry"],
                }
            )

    return results
