"""Wikidata country entities of an EM-DAT country.

Wikidata candidates are kept when their country (P17) is the event's country.
The test is made on Wikidata entities (QIDs), found from the ISO 3166-1 alpha-3
code (P298) of the EM-DAT ISO column. Comparing country names instead fails for
every country whose Wikidata name differs from the EM-DAT name (e.g. "United
States" and "United States of America"), which then gets no Wikidata point.
"""
from SPARQLWrapper import SPARQLWrapper, JSON

# Country entities accepted besides those whose P298 is the EM-DAT ISO code
EXTRA_COUNTRY_QIDS = {
    "NLD": {"Q55"},  # P298 NLD is on the Kingdom of the Netherlands (Q29999); places use Q55
    "SCG": {"Q403", "Q236", "Q37024"},  # Serbia and Montenegro: Serbia, Montenegro, the union
    "SPI": {"Q29"},  # EM-DAT code of the Canary Islands: Spain
    # Territories: their places give the sovereign country as P17
    "ASM": {"Q30"}, "GUM": {"Q30"}, "MNP": {"Q30"}, "VIR": {"Q30"},
    "GLP": {"Q142"}, "GUF": {"Q142"}, "MTQ": {"Q142"}, "REU": {"Q142"},
    "PYF": {"Q142"}, "NCL": {"Q142"}, "WLF": {"Q142"},
    "HKG": {"Q148"}, "CYM": {"Q145"}, "SHN": {"Q145"}, "NIU": {"Q664"},
}

_iso3_qids = None


def _load_iso3_qids():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery("SELECT ?c ?iso WHERE { ?c wdt:P298 ?iso . }")
    sparql.setReturnFormat(JSON)
    out = {}
    for r in sparql.query().convert()["results"]["bindings"]:
        out.setdefault(r["iso"]["value"], set()).add(r["c"]["value"].split("/")[-1])
    return out


def country_qids(iso3):
    """Wikidata QIDs accepted as the country of an EM-DAT ISO3 code."""
    global _iso3_qids
    if _iso3_qids is None:
        _iso3_qids = _load_iso3_qids()
    return _iso3_qids.get(iso3, set()) | EXTRA_COUNTRY_QIDS.get(iso3, set())
