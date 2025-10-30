# EM-DAT Geocoding — Validation and Comparison Toolkit

NOTE: this README has been AI-generated and still need to be reviewed and updated. 

Code repository associated with the paper: <ADD PAPER REFERENCE>

This repository contains the code for the validation and comparison of 
LLM-assisted described in the paper.
A small, reproducible toolkit to preprocess, geocode (incl. LLM‑assisted disambiguation), and validate EM‑DAT location geometries against two benchmarks:

- GDIS: a GADM‑based geocoding benchmark
- EM‑DAT GAUL: official EM‑DAT geometries

It also compares GDIS and EM‑DAT GAUL together for additional insights. Geometric comparisons are primarily area‑based.


## Key features
- Batch preprocessing for GDIS and LLM‑assisted EM‑DAT locations
- Geometry indices: geodetic area, containment ratios, Jaccard (IoU)
- Vectorized operations with geometry hygiene (validity, dissolve when needed)
- Config‑driven paths/parameters via `config.toml`
- Structured logging and reproducible outputs with config snapshots


## Project structure
- `config.toml` — central configuration (paths, logging, options)
- `validation/geom_indices.py` — area/overlap metrics (geodetic or equal‑area)
- `validation/validation.py` — batch comparison pipelines and metrics
- `validation/io.py` — IO helpers for archives, GPKG batches, CSVs
- `run_preprocessing_gdis.py` — prepare GDIS batches
- `run_preprocessing_llm.py` — prepare LLM‑assisted EM‑DAT batches
- `run_validation.py` — run validations between sources
- `run_all.py` — orchestrate end‑to‑end runs
- `output/` — results, logs; per‑run stamped filenames
- `data/` — lightweight inputs (no geometries)

See also: `comparison_figures.ipynb` for exploration.


## Installation
- Python 3.13+
- Dependencies listed in `pyproject.toml` (and locked in `uv.lock`)

Example using pip:
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -e .
```
Or, if you use uv:
```bash
uv sync
```


## Configuration
All tunables live in `config.toml`.

- `[path]`
  - `batch_dir` — folder for generated batches (heavy geometries)
  - `emdat_gaul_path` — EM‑DAT GAUL benchmark GPKG
  - `gdis_path` — GDIS GPKG (source or benchmark)
  - `emdat_archive_path` — EM‑DAT Excel archive (lightweight; no geometries)
  - `gdis_disno_path` — CSV with GDIS `DisNo.` mapping
  - `validation_output_dir` — where CSV outputs and logs are written
- `[geoprocessing]`
  - `area_calculation_method` — `'geodetic'` (default) or `'equal_area'`
    - For equal‑area, the code reprojects to an equal‑area CRS (e.g., EPSG:6933). This is documented inline in code comments.
- `[index]`
  - `llm_columns_to_keep` — non‑geometry columns kept from LLM preprocessing
  - `llm_geom_columns` — expected geometry columns (OSM/GADM/Wikidata)
  - `batch_numbers` — integers to select sub‑batches for processing
- `[validation]`
  - `skip_if_output_exists` — skip sub‑batches if the expected output CSV exists
- `[logging]` — level, filename, format; can be overridden in `run_*.py`

Adjust these paths to your local environment before running.


## Inputs
Placeholders — replace with your locations if different.

- EM‑DAT Excel archive (lightweight):
  - Default: `data/241204_emdat_archive.xlsx`
  - Source/DOI: doi:10.14428/DVN/I0LTPH
- GDIS geometries (GPKG):
  - Example path: `Q:/Data/emdat_geocoding/GDIS/gdis.gpkg`
- EM‑DAT GAUL geometries (GPKG):
  - Example path: `Q:/Data/emdat_geocoding/EMDAT_GAUL/geoemdat_gaul.gpkg`
- Batch directory for generated heavy files:
  - Example path: `Q:/Data/emdat_geocoding/batch_files`
- Optional: GDIS `DisNo.` CSV mapping:
  - Default: `data/gdis_disnos.csv`

If your inputs live elsewhere or require credentials, insert your own paths or instructions here: <ADD LOCAL/REMOTE ACCESS INSTRUCTIONS>.


## Workflows
You can run individual steps or the full pipeline. All scripts honor `config.toml` and write logs to the repo root or `output/`.

### 1) Preprocess GDIS batches
Prepares GDIS data for validation (geometries, attributes, batching):
```bash
python run_preprocessing_gdis.py
```
- Reads from `gdis_path`
- Writes batches under `batch_dir`
- Log file: `.log` (overridden to a task‑specific name by the script)

### 2) Preprocess LLM‑assisted EM‑DAT batches
Creates candidate geometries per EM‑DAT location string using OSM/GADM/Wikidata and an LLM for disambiguation ranking. Stores chosen geometry and candidate set when needed.
```bash
python run_preprocessing_llm.py
```
- Reads EM‑DAT archive and any cached sources
- Writes batches under `batch_dir`
- Log: see `.log` or `output/preprocessing_llm_*.log`

### 3) Validation and comparison
Runs the geometric comparison between sources (GDIS vs EM‑DAT GAUL, LLM vs EM‑DAT GAUL, etc.).
```bash
python run_validation.py
```
- Outputs CSVs under `validation_output_dir` with run‑stamped names
- Writes a run log, e.g., `validation_all.log`

### 4) End‑to‑end
```bash
python run_all.py
```
Runs the sequence above with sensible defaults.


## Geometry and CRS conventions
- Storage CRS: EPSG:4326 unless otherwise stated
- Area computations:
  - Default: geodetic area calculations (spheroid‑aware)
  - Alternative: reproject to an equal‑area CRS (e.g., EPSG:6933) before area; choice is explained in code comments
- Geometry hygiene: enforce validity, dissolve multipart features where appropriate, and favor topology‑safe operations


## Validation metrics
Computed in `validation/geom_indices.py`. For each pair of geometries A (candidate) and B (benchmark):

- `area_a`, `area_b` — areas (square meters)
- `intersection_area`, `union_area`
- `a_in_b` = |A ∩ B| / |A| — containment of A by B
- `b_in_a` = |A ∩ B| / |B| — containment of B by A
- `jaccard` = |A ∩ B| / |A ∪ B| — Intersection over Union

Two modes are typically reported:
- Per‑unit containment (|A ∩ B| / |A|)
- Dissolved “disaster footprint” Jaccard (|A ∩ B| / |A ∪ B|)

When helpful, we also report |A ∩ B| / |B| to detect GAUL‑within‑GADM cases.


## Outputs
- CSV files in `output/` (or the configured `validation_output_dir`):
  - Examples: `validation_gadm_2000_2002_*.csv`, `validation_all_YYYYMMDD_HHMM.csv`
- Logs in the repo root or `output/`, e.g., `preprocessing_gdis.log`, `preprocessing_llm.log`, `validation_all.log`
- Optionally, figures in `comparison_figures.ipynb`

Each export should include a config snapshot and ideally a commit hash for reproducibility. If yours are missing, add placeholders: <ADD COMMIT HASH/CONFIG SNAPSHOT STEP>.


## Reproducibility
- Keep `config.toml` under version control and avoid hard‑coded paths in code
- When randomness is used (e.g., sampling), set and document seeds
- Stamp outputs by date/time; store logs alongside the outputs


## Troubleshooting
- Geometry validity errors: ensure inputs are valid; consider buffering by 0 or using topology‑preserving fixing before area/overlay
- CRS mismatches: verify inputs are in EPSG:4326 or explicitly reproject
- Memory/performance: process in batches (`batch_numbers`); cache expensive lookups
- Paths on Windows: prefer forward‑portable paths via `config.toml`; avoid hard‑coding drive letters in code


## Citation
If you use this toolkit in a publication, please cite EM‑DAT and GDIS appropriately, and reference this repository. Add your preferred citation text here: <ADD CITATION>.


## License and contribution
- License: <ADD LICENSE>
- Contributions are welcome. Please match existing patterns and idioms, add type hints, and use numpydoc docstrings when modifying or adding functions/classes.
