# emdat_geocoding

Scripts for validating and comparing EM-DAT geocoded geometries

## Extra data

To be able to run the example, you need to download the data batch files
(expired on Sept 29, 2025):

- https://we.tl/t-8QWc2x2pH6

## Repository and File Descriptions

### Overview
A small toolkit to validate and compare EM-DAT geocoded geometries (GADM/OSM) against official EM-DAT/GAUL footprints.

- Core: geotools/geom_indices.py — computes geodetic areas, containment, and Jaccard; supports geodetic or equal-area.
- IO: validation/io.py — loads .gpkg batches and the EM-DAT Excel archive.
- Pipeline: example_validation.py — runs validation and writes a CSV (+ log).
- Optional: example_figures.ipynb for exploration.

### Data (expected paths)
- data/geoloc_emdat_0002_<gadm|osm>.gpkg
- data/geoemdat_<YYYY_START>_<YYYY_END>.gpkg
- data/241204_emdat_archive.xlsx

### Outputs
- validation_gadm_2000_2002*.csv

### Installation
- Python 3.13+
- see requirements in pyproject.toml

### Quick start
1) Put example data under data/.
2) Optionally adjust params in example_validation.py.
3) Run: python example_validation.py
4) Outputs: CSV + example_validation.log

## Validation summary

- Two modes: per-unit (containment |A∩B|/|A|) and dissolved disaster (Jaccard |A∩B|/|A∪B|).
- Also report |A∩B|/|B| to detect GAUL-within-GADM cases.
- Natural hazards only; GAUL footprints from EM-DAT 1900–2023.

Returned indices: area_a, area_b, intersection_area, union_area, a_in_b, b_in_a, jaccard. See geotools/geom_indices.py for details.


| Index                                                                                       | Label                     | Description                                                     |
|---------------------------------------------------------------------------------------------|---------------------------|-----------------------------------------------------------------|
| ![equation](https://latex.codecogs.com/png.latex?\|A\|)                                     | "area_a"                  | The area of A, in square meters.                                |
| ![equation](https://latex.codecogs.com/png.latex?\|B\|)                                     | "area_b"                  | The area of B, in square meters.                                |
| ![equation](https://latex.codecogs.com/png.latex?\|A%20\cap%20B\|)                          | "intersection_area"       | The area of intersection between $A$ and $B$, in square meters. |
| ![equation](https://latex.codecogs.com/png.latex?\|A%20\cup%20B\|)                          | "union_area"              | The area of union between $A$ and $B$, in square meters.        |
| ![equation](https://latex.codecogs.com/png.latex?\frac{\|A%20\cap%20B\|}{\|A\|})            | "a_in_b"                  | The ratio of the intersection area to $A$.                      |
| ![equation](https://latex.codecogs.com/png.latex?\frac{\|A%20\cap%20B\|}{\|B\|})            | "b_in_a"                  | The ratio of the intersection area to $B$ .                     |
| ![equation](https://latex.codecogs.com/png.latex?\frac{\|A%20\cap%20B\|}{\|A%20\cup%20B\|}) | "jaccard"                 | Intersection over Union, a.k.a., the Jaccard Index.             |
