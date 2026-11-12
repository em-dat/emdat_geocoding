"""Re-query the Wikidata points of the countries that got none.

In the dataset versions published before November 2026,
run_geolocation.select_closest_candidate kept a Wikidata candidate only if its
country *name* equalled the EM-DAT country name. Countries whose names differ
(e.g. "United States" / "United States of America", "Viet Nam", "Türkiye") got
no Wikidata point at all. run_geolocation.py now tests
the country on Wikidata entities (wikidata_utils.country_qids).

This script applies the corrected selection to an existing dataset without
re-running the whole pipeline: in every country where no location has a
Wikidata point, each location is queried again as in run_geolocation.py (exact
English label, LIMIT 50, candidates with coordinates in the country; Admin2: the
candidate closest to the Wikidata point of its Admin1, otherwise the first
candidate). Only the geometry_wiki column of these rows changes; all other rows
and columns are copied unchanged.

Usage (from the repository root):
    python geocoding/update_wikidata_points.py <input_dir> <output_dir> [--cache <file.jsonl>]

<input_dir> holds LLMGeoDis_part1..5 as .csv or .zip; the corrected parts are
written to <output_dir> as .csv, identical to the input except for the
re-queried geometry_wiki values. Queries are cached in the
--cache file (default: <output_dir>/wikidata_cache.jsonl), so an interrupted
run resumes without querying again.
"""
import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import requests
import shapely
from geopy.distance import geodesic

from wikidata_utils import EXTRA_COUNTRY_QIDS

URL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "emdat-geocoding/1.0 (research script)",
           "Accept": "application/sparql-results+json"}
PARTS = [f"LLMGeoDis_part{n}" for n in range(1, 6)]
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


class Sparql:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                k, v = json.loads(line)
                self.cache[k] = v

    def __call__(self, query):
        if query in self.cache:
            return self.cache[query]
        for attempt in range(8):
            time.sleep(1)
            try:
                r = requests.get(URL, params={"query": query}, headers=HEADERS, timeout=90)
            except requests.RequestException:
                time.sleep(10 + 10 * attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(10 + 10 * attempt)
                continue
            r.raise_for_status()
            self.cache[query] = r.json()["results"]["bindings"]
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps([query, self.cache[query]]) + "\n")
            return self.cache[query]
        raise RuntimeError(f"Wikidata query failed: {query[:80]}")


def candidates(sparql, name):
    """Candidates of run_geolocation.wikidata_geocode: (country QID, lon, lat)."""
    label = name.replace("\\", "\\\\").replace('"', '\\"')
    q = f"""SELECT ?item ?coord ?country WHERE {{
      ?item rdfs:label "{label}"@en .
      OPTIONAL {{ ?item wdt:P625 ?coord. }}
      OPTIONAL {{ ?item wdt:P17 ?country. }}
    }} LIMIT 50"""
    out = []
    for r in sparql(q):
        coord = r.get("coord", {}).get("value")
        if coord and coord.startswith("Point("):
            try:
                lon, lat = map(float, coord[6:-1].split())
            except ValueError:
                continue
            out.append((r.get("country", {}).get("value", "").split("/")[-1], lon, lat))
    return out


def select(cands, allowed, parent=None):
    """run_geolocation.select_closest_candidate, country tested on QIDs."""
    cands = [c for c in cands if c[0] in allowed]
    if not cands:
        return None
    if parent is None:
        return cands[0]
    return min(cands, key=lambda c: geodesic((c[2], c[1]), (parent[2], parent[1])).km)


def open_part(input_dir, part):
    csv_path = input_dir / f"{part}.csv"
    if csv_path.exists():
        return csv_path.open(encoding="utf-8", newline="")
    z = zipfile.ZipFile(input_dir / f"{part}.zip")
    return io.TextIOWrapper(z.open(f"{part}.csv"), encoding="utf-8", newline="")


def read_rows(input_dir):
    """Yield (part, row) over the parts, row as a list of strings."""
    for part in PARTS:
        with open_part(input_dir, part) as f:
            reader = csv.reader(f)
            yield part, next(reader)  # header
            yield from ((part, row) for row in reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--cache", type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sparql = Sparql(args.cache or args.output_dir / "wikidata_cache.jsonl")

    # Pass 1: countries without any Wikidata point, and their locations
    header, rows, has_wiki = None, [], set()
    for i, (part, row) in enumerate(read_rows(args.input_dir)):
        if row[0] == "DisNo.":
            header = row
            col = {c: k for k, c in enumerate(header)}
            continue
        iso3 = row[col["DisNo."]][-3:]
        if row[col["geometry_wiki"]]:
            has_wiki.add(iso3)
        rows.append((iso3, row[col["name"]], row[col["admin_level"]], row[col["admin1"]]))
    countries = sorted({r[0] for r in rows} - has_wiki)
    print(f"{len(countries)} countries without Wikidata points: {' '.join(countries)}", flush=True)

    iso_qids = {}
    for r in sparql("SELECT ?c ?iso WHERE { ?c wdt:P298 ?iso . }"):
        iso_qids.setdefault(r["iso"]["value"], set()).add(r["c"]["value"].split("/")[-1])
    allowed = {c: iso_qids.get(c, set()) | EXTRA_COUNTRY_QIDS.get(c, set()) for c in countries}

    new_points, n_hit = {}, 0
    todo = [k for k, r in enumerate(rows) if r[0] in allowed]
    print(f"{len(todo)} locations to query", flush=True)
    for j, k in enumerate(todo):
        iso3, name, level, admin1 = rows[k]
        parent = None
        if level == "Admin2" and admin1:
            parent = select(candidates(sparql, admin1), allowed[iso3])
        hit = select(candidates(sparql, name), allowed[iso3], parent) if name else None
        if hit:
            new_points[k] = shapely.Point(hit[1], hit[2]).wkt
            n_hit += 1
        if j % 1000 == 0:
            print(f"  {j}/{len(todo)} locations", flush=True)
    print(f"{n_hit} of {len(todo)} locations got a Wikidata point", flush=True)

    # Pass 2: write the parts, geometry_wiki replaced in the re-queried rows
    k = -1
    current, out, writer = None, None, None
    for part, row in read_rows(args.input_dir):
        if part != current:
            if out:
                out.close()
            current = part
            out = (args.output_dir / f"{part}.csv").open("w", encoding="utf-8", newline="")
            writer = csv.writer(out, lineterminator="\r\n")
            writer.writerow(row)  # header
            print(f"writing {part}", flush=True)
            continue
        k += 1
        if k in new_points:
            row[col["geometry_wiki"]] = new_points[k]
        writer.writerow(row)
    out.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
