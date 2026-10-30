"""GADM helpers shared by the geocoding scripts.

Used by ``run_geolocation.py`` and ``gadm_projection.py``; kept in a separate
module so they can be imported without running the geocoding pipeline.
"""
import re
import unicodedata

import geopandas as gpd
import pandas as pd
import pycountry
from rapidfuzz import fuzz, process

# EM-DAT ISO codes whose territory GADM files under another ISO3 code
EMDAT_TO_GADM_ISO3 = {
    "SPI": "ESP",  # Canary Islands
    "SCG": "SRB",  # Serbia Montenegro
}

# Admin1 name matching (match_admin1)
ADMIN1_FUZZY_THRESHOLD = 80  # minimum fuzz.ratio score (inclusive), e.g. Herat/Hirat = 80
# Generic terms removed from both the parsed name and the GADM names
ADMIN1_GENERIC_TERMS = [
    "region", "regions", "province", "provinces", "state", "states", "division",
    "divisions", "territory", "territories", "department", "departments",
    "departement", "departements", "departamento", "departamentos", "governorate",
    "governorates", "district", "districts", "municipality", "municipalities",
    "municipal", "municipal district", "municipal districts", "city", "cities",
    "county", "counties", "prefecture", "prefectures", "borough", "boroughs",
    "commune", "communes", "parish", "parishes", "ward", "wards", "sector",
    "sectors", "zone", "tower", "zones", "subdistrict", "subdistricts",
    "subdivision", "subdivisions", "locality", "localities", "township",
    "townships", "town", "towns", "village", "villages", "hamlet", "hamlets",
    "regency", "regencies", "area", "areas", "island", "islands", "isl\\.",
    "lake", "lakes", "river", "rivers", "mount", "mounts", "mountain",
    "mountains", "valley", "valleys", "peninsula", "peninsulas", "oblast",
    "krai", "kray", "republic", "federal", "sheng", "zizhiqu", "ziziqu", "shi",
    "autonomous",
]
_GENERIC_TERMS_RE = re.compile(r"\b(" + "|".join(ADMIN1_GENERIC_TERMS) + r")\b")
# Letters that NFD decomposition leaves as is, and separators
_TRANSLITERATION = str.maketrans({
    "ı": "i", "ł": "l", "ø": "o", "đ": "d", "ð": "d", "þ": "th", "ß": "ss",
    "æ": "ae", "œ": "oe", "ʻ": "", "ʼ": "", "’": "", "'": "", "`": "",
    "-": " ", "_": " ", ".": " ",
})
_admin1_candidates = {}


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


def simplify_name(s):
    """
    Lowercase, remove accents, transliterate the letters NFD leaves as is
    (e.g. Turkish dotless i) and treat hyphens, dots and apostrophes as spaces.
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r"\s+", " ", s.translate(_TRANSLITERATION)).strip()


def clean_admin_name(s):
    """
    simplify_name() without bracketed text and generic terms such as Province,
    Oblast or Sheng. A name made only of such terms (e.g. Rivers) is kept.
    """
    simple = simplify_name(s)
    cleaned = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", simple)
    cleaned = re.sub(r"\s+", " ", _GENERIC_TERMS_RE.sub(" ", cleaned)).strip(" ,")
    return cleaned or simple


def admin1_candidates(gadm1, iso3):
    """GADM Admin1 names and alternative names (VARNAME_1) of one country."""
    key = (id(gadm1), iso3)
    if key not in _admin1_candidates:
        rows = []
        for idx, unit in gadm1[gadm1["iso3"] == iso3].iterrows():
            rows.append((idx, unit["ADMIN1"], "name", unit["ADMIN1"]))
            variants = unit.get("VARNAME_1")
            if isinstance(variants, str):
                rows += [(idx, unit["ADMIN1"], "variant", v.strip())
                         for v in variants.split("|") if v.strip()]
        cands = pd.DataFrame(rows, columns=["index", "unit", "kind", "label"])
        cands["simple"] = cands["label"].map(simplify_name)
        cands["clean"] = cands["label"].map(clean_admin_name)
        cands["words"] = cands["clean"].map(lambda c: " ".join(sorted(c.split())))
        _admin1_candidates[key] = cands
    return _admin1_candidates[key]


def match_admin1(name, gadm1, iso3):
    """
    Index in gadm1 of the GADM Admin1 unit matching `name`, or None.

    Stages, from strictest to loosest; the first stage with a hit decides:
    official name, alternative name (VARNAME_1), official name once generic
    terms are removed from both sides, same words in another order, then
    fuzzy matching on official names (score >= ADMIN1_FUZZY_THRESHOLD).
    If a stage finds units with different names, the name is ambiguous: None.
    Rows sharing an ADMIN1 name count as one unit, and the first is returned.
    """
    cands = admin1_candidates(gadm1, iso3)
    if cands.empty or not simplify_name(name):
        return None
    simple, cleaned = simplify_name(name), clean_admin_name(name)
    words = " ".join(sorted(cleaned.split()))
    is_name = cands["kind"] == "name"

    def unique_unit(hits):
        units = cands.loc[hits].drop_duplicates("unit")
        return units["index"].iloc[0] if len(units) == 1 else None

    for hits in (is_name & (cands["simple"] == simple),
                 ~is_name & (cands["simple"] == simple),
                 is_name & (cands["clean"] == cleaned),
                 cands["words"] == words):
        if hits.any():
            return unique_unit(hits)
    scores = process.cdist([cleaned], cands["clean"].tolist(), scorer=fuzz.ratio)[0]
    scores[~is_name.to_numpy()] = -1  # fuzzy matching on official names only
    best = scores.max()
    if best >= ADMIN1_FUZZY_THRESHOLD:
        return unique_unit(scores == best)
    return None


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
        idx = match_admin1(admin1_name, gadm1, iso3)
        if idx is not None:
            row = gadm1.loc[idx]
            results["Admin1"].append({
                "name": admin1_name,
                "gadm_admin1": row["ADMIN1"],
                "geometry": row["geometry"]
            })

    # --- Admin2 matching (strict: inside Admin1) ---
    for admin2 in parsed_json.get("Admin2", []):
        admin2_name = admin2["name"]
        admin1_name = admin2["Admin1"]

        gadm2_country = gadm2[gadm2["iso3"] == iso3]
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
