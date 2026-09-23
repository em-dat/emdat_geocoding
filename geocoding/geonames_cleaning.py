"""EM-DAT location cleaning of Teber et al. (Geo-Disasters), used by the
GeoNames benchmark.

Ported from https://github.com/khalilT/geocode_disasters (commit 7abd89b),
src/utils/constants.py, src/utils/functions.py and scripts/1_clean_emdat.py,
distributed under the following licence:

    MIT License

    Copyright (c) 2024 Khalil Teber

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import re

import pandas as pd

# Ported from khalilT-geocode_disasters-7abd89b/src/utils/constants.py
REP_BURKINA = {
    "west central": "centre-ouest",
    "north central": "centre-nord",
    "north loroum": "loroum",
    "west tuy": "tuy",
    "sector 12": "ouagadougou",
    "sect 30": "ouagadougou",
    "arr. 10": "ouagadougou",
    "arr. 4": "ouagadougou",
    "arr. 7": "ouagadougou",
    "arr.8": "ouagadougou",
    "arr.3": "ouagadougou",
    "centrall": "centre",
    "north": "nord",
    "west": "centre-ouest",
    "east": "est",
}

REP_HAITI = {
    "north east": "nord est",
    "north-east": "nord est",
    "north west": "nord ouest",
    "north-west": "nord ouest",
    "northwest": "nord ouest",
    "north": "nord",
    "south west": "sud ouest",
    "south-west": "sud ouest",
    "south east": "sud est",
    "south": "sud",
}

REP_CHAD = {
    "n'djamena region": "ndjamena",
    "near n'djamena": "ndjamena",
    "n'djam centre": "ndjamena",
    "n'djam est": "ndjamena",
    "n'djam sud": "ndjamena",
    "n'djamena": "ndjamena",
}

REP_PHILIPPINES = {
    "Region XIII": "Caraga",
    "Region XII": "Soccsksargen",
    "Region XI": "Davao Region",
    "Region X": "Northern Mindanao",
    "Region IX": "Zamboanga Peninsula",
    "Region VIII": "Eastern Visayas",
    "Region VII": "Central Visayas",
    "Region VI": "Western Visayas",
    "Region V": "Bicol region",
    "Region IV-A": "Calabarzon",
    "Region IV": "Southern Tagalog",
    "Region III": "Central Luzon",
    "Region II": "Cagayan Valley",
    "Region I": "Ilocos region",
    "XIII": "Caraga",
    "XII": "Soccsksargen",
    "XI": "Davao Region",
    "X": "Northern Mindanao",
    "IX": "Zamboanga Peninsula",
    "VIII": "Eastern Visayas",
    "VII": "Central Visayas",
    "VI": "Western Visayas",
    "V": "Bicol region",
    "IV-A": "Calabarzon",
    "IV": "Southern Tagalog",
    "III": "Central Luzon",
    "II": "Cagayan Valley",
    "I": "Ilocos region",
    "12": "Soccsksargen",
    "11": "Davao Region",
    "10": "Northern Mindanao",
    "9": "Zamboanga Peninsula",
    "8": "Eastern Visayas",
    "7": "Central Visayas",
    "6": "Western Visayas",
    "5": "Bicol region",
    "4": "Southern Tagalog",
    "3": "Central Luzon",
    "2": "Cagayan Valley",
    "1": "Ilocos region",
}

REP_US_STATES = {
    "al": "alabama",
    "ak": "alaska",
    "az": "arizona",
    "ar": "arkansas",
    "ca": "california",
    "co": "colorado",
    "ct": "connecticut",
    "de": "delaware",
    "fl": "florida",
    "ga": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois",
    "in": "indiana",
    "ia": "iowa",
    "ks": "kansas",
    "ky": "kentucky",
    "la": "louisiana",
    "me": "maine",
    "md": "maryland",
    "ma": "massachusetts",
    "mi": "michigan",
    "mn": "minnesota",
    "ms": "mississippi",
    "mo": "missouri",
    "mt": "montana",
    "ne": "nebraska",
    "nv": "nevada",
    "nh": "new hampshire",
    "nj": "new jersey",
    "nm": "new mexico",
    "ny": "new york",
    "nc": "north carolina",
    "nd": "north dakota",
    "oh": "ohio",
    "ok": "oklahoma",
    "or": "oregon",
    "pa": "pennsylvania",
    "ri": "rhode island",
    "sc": "south carolina",
    "sd": "south dakota",
    "tn": "tennessee",
    "tx": "texas",
    "ut": "utah",
    "vt": "vermont",
    "va": "virginia",
    "wa": "washington",
    "wv": "west virginia",
    "wi": "wisconsin",
    "wy": "wyoming",
}

REPLACE_TERMS = [
    "Near", "Between", "Provinces", "Province", "Prov.", "Districts", "District", 
    "Dis.", "Div.", "Regions", "Region", "states", "state", "(cities)", "(City)", 
    "cities", "City", "Regency", "districts", "county", "Departments", "Department", 
    "municipalities", "Municipality", "=", "Level 2", "-", "N.A. on the source", 
    "islands", "island", "of the", " isl.(archip.)", " isl.", "area", " in"
]

# Ported from khalilT-geocode_disasters-7abd89b/src/utils/functions.py

def split_and_clean_locations(location):
    # Define the pattern to identify "Level 1" and its surroundings
    pattern = re.compile(r"^(.*?)(Level 1\s*(.*))$", re.IGNORECASE)

    match = pattern.match(location)
    if match:
        before_level_1 = match.group(1).strip()
        after_level_1 = match.group(3).strip()
        # Format the output with "Level 1" part in brackets
        cleaned_entries = [f"{before_level_1} ({after_level_1})"]
    else:
        # Proceed with the original split logic if "Level 1" is not found
        entries = re.split(r";", location)
        cleaned_entries = []
        for entry in entries:
            if "(" in entry and ")" in entry:
                cleaned_entries.append(entry.strip())
            else:
                sub_entries = re.split(r",\s*(?![^()]*\))", entry)
                cleaned_entries.extend(
                    [
                        sub_entry.strip()
                        for sub_entry in sub_entries
                        if sub_entry.strip()
                    ]
                )

    return cleaned_entries

def split_text(text):
    if text.count("(") > 1:
        return re.split(r",\s*(?=\S)", re.sub(r"\),\s*(?=\S)", "),\n", text))
    return re.split(r"\),\s*(?=\S)", text)

def extract_locations(row):
    locations = []
    if "(" in row:
        parts = row.split("(")
        locs = parts[0].strip().split(",")
        regions = parts[1].replace(")", "").split(",")
        for loc in locs:
            for sub_loc in loc.strip().split("/"):
                for region in regions:
                    locations.append([sub_loc.strip(), region.strip()])
    elif "," in row:
        for loc in row.split(","):
            locations.append([loc.strip(), None])
    else:
        for loc in row.strip().split("/"):
            locations.append([loc.strip(), None])
    return locations

def remove_str_if_last(s):
    if isinstance(s, str) and s.endswith(","):
        return s[:-1]
    return s

def clean_location_string(location, iso):
    """
    Apply country-specific corrections and general cleaning to a location string.
    Based on Khalil Teber's 1_clean_emdat.py.
    """
    if pd.isna(location):
        return location
    
    location = location.lower()
    
    # Country-specific corrections
    if iso == "PHL":
        for k, v in REP_PHILIPPINES.items():
            location = re.sub(rf"\b{re.escape(k.lower())}\b", v.lower(), location)
    elif iso == "BFA":
        for k, v in REP_BURKINA.items():
            location = re.sub(rf"\b{re.escape(k.lower())}\b", v.lower(), location)
    elif iso == "HTI":
        for k, v in REP_HAITI.items():
            location = re.sub(rf"\b{re.escape(k.lower())}\b", v.lower(), location)
    elif iso == "TCD":
        for k, v in REP_CHAD.items():
            location = re.sub(rf"\b{re.escape(k.lower())}\b", v.lower(), location)
            
    # Standardize separation
    location = re.sub(r"\) and\b", "),", location)
    location = re.sub(r"\b(and|between|&|\+)\b", ",", location)
    
    return location

def process_emdat_locations(df):
    """
    Full pipeline to process EM-DAT location strings into a list of geocodable locations.
    """
    # Filter for natural events with locations (including those that already have Admin Units for benchmark validation)
    mask = (df["Location"].notna()) & (df["Disaster Group"] == "Natural")
    working_df = df[mask].copy()
    
    # Step 1: Initial cleaning
    working_df["Location"] = working_df.apply(lambda x: clean_location_string(x["Location"], x["ISO"]), axis=1)
    
    # Step 2: Split and Parse
    expanded_rows = []
    for _, row in working_df.iterrows():
        for loc in split_and_clean_locations(row["Location"]):
            expanded_rows.append([row["DisNo."], row["ISO"], loc])
            
    expanded_df = pd.DataFrame(expanded_rows, columns=["DisNo.", "ISO", "Individual_Location"])
    
    # Step 3: Remove generic terms
    for term in REPLACE_TERMS:
        expanded_df["Individual_Location"] = expanded_df["Individual_Location"].str.replace(term, " ", case=False, regex=False)
        
    # Step 4: Further splitting
    expanded_df["Individual_Location"] = expanded_df["Individual_Location"].apply(split_text)
    expanded_df = expanded_df.explode("Individual_Location", ignore_index=True)
    expanded_df["Individual_Location"] = expanded_df["Individual_Location"].str.replace(
        r"\b(and| & |between| \+ | \) and)\b", ",", regex=True
    )
    
    # Step 5: Extract structured locations
    structured_data = []
    for _, row in expanded_df.iterrows():
        for loc_parts in extract_locations(row["Individual_Location"]):
            structured_data.append([row["DisNo."], row["ISO"], row["Individual_Location"], loc_parts[0], loc_parts[1]])
            
    structured_df = pd.DataFrame(structured_data, columns=["DisNo.", "ISO", "Raw_Part", "Location_Before", "Bracketed"])
    
    # Step 6: Appended location for geocoding
    structured_df["Appended"] = (
        structured_df["Location_Before"]
        + ","
        + structured_df["Bracketed"].apply(lambda x: f" {x}" if x else "")
    )
    structured_df["Appended"] = structured_df["Appended"].apply(remove_str_if_last)
    
    # Step 7: Final Filtering and Corrections
    # Remove locations that are only numbers
    structured_df = structured_df[~structured_df["Location_Before"].str.isdigit().fillna(False)]
    
    # USA specific state corrections
    usa_mask = (structured_df["ISO"] == "USA") & (structured_df["Location_Before"].str.len() == 2)
    if usa_mask.any():
        for k, v in REP_US_STATES.items():
            structured_df.loc[usa_mask, "Location_Before"] = structured_df.loc[usa_mask, "Location_Before"].str.replace(rf"^{k}$", v, regex=True, case=False)
        # Update Appended after USA corrections
        structured_df.loc[usa_mask, "Appended"] = (
            structured_df.loc[usa_mask, "Location_Before"]
            + ","
            + structured_df.loc[usa_mask, "Bracketed"].apply(lambda x: f" {x}" if x else "")
        ).apply(remove_str_if_last)

    # ISO corrections (AZO -> PRT, DFR -> DEU, SCG -> SRB)
    iso_map = {"AZO": "PRT", "DFR": "DEU", "SCG": "SRB"}
    structured_df["ISO"] = structured_df["ISO"].replace(iso_map)
    
    # Montenegro correction
    mne_mask = (structured_df["ISO"] == "SRB") & (structured_df["Bracketed"].str.lower() == "montenegro")
    structured_df.loc[mne_mask, "ISO"] = "MNE"
    
    # Remove obsolete countries
    structured_df = structured_df[~structured_df["ISO"].isin(["YUG", "SUN", "ANT"])]
    
    # Clean up results
    results_df = structured_df[["DisNo.", "ISO", "Appended"]].rename(columns={"Appended": "Location"})
    results_df = results_df[results_df["Location"] != "nan"]
    results_df = results_df.drop_duplicates()
    
    return results_df.reset_index(drop=True)
