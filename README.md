# EM‑DAT Geocoding — Code and Data (LLM‑GeoDis)

This repository contains the code and inputs to reproduce the
LLM‑assisted geocoding (LLM‑GeoDis) and the geometry comparison/validation
workflows described in the associated manuscript: [Anonymized]

## Overview

Code and data allow reproducing the following steps:
- Geoparse EM‑DAT textual locations with GPT‑4o into GADM 4.1, OSM and
  Wikidata administrative units and points.
- Project and harmonize geometries to GADM 4.1.
- Compare and validate against GDIS (GADM‑based) and EM‑DAT GAUL 2015
  benchmarks.
- Generate descriptive statistics and figures.

## Repository layout

The repository is organized following the logical workflow of the project, from
geocoding to validation and reporting.

### Data files (put under `data/`)

This repository contains both raw and preprocessed data, enabling to run part 
of the workflow.

- `241204_emdat_archive.xlsx` — EM‑DAT FAIR Archive 1900–2023 (DOI:
  10.14428/DVN/I0LTPH). Used to list `DisNo.` with GAUL geometries.
- `gdis_disnos.csv` — EM‑DAT `DisNo.` identifiers geocoded by GDIS (from
  https://doi.org/10.7927/ZZ3B‑8Y61). Terms of use: see source.
- `LLMGeoDis_part1.zip` … `LLMGeoDis_part5.zip` — the LLM‑GeoDis database
  split into five ZIPs. After download, unzip all into a single folder such as
  `data/LLMGeoDis/`. The unzipped content consists of CSV files grouped by
  batch and provider (GADM/OSM/Wikidata) with columns including
  `DisNo.`, `name`, `admin_level`, `admin1`, `admin2`, `iso3` and one or more
  geometry columns (e.g., `geometry_gadm`, `geometry_osm`, `geometry_wiki`).
- `geoemdat_gaul.gpkg` — EM‑DAT GAUL 2015 geometries (benchmark). Point
  `path.emdat_gaul_path` to this file.
- `pend-gdis-1960-2018-disasterlocations.csv` — GDIS reference. You may need to
  convert to GPKG aligned to GADM to use as `path.gdis_path`, or rely on
  external GDIS distributions that provide a GADM‑based GPKG.
- `input_emdat.csv` — EM‑DAT input used during LLM geoparsing (for reference).
- `reliability_db.csv` — reliability annotations for geoparsing (for reference).
- `synthetic_EMDAT_locations.csv` — synthetic examples used during development
  (for reference/testing).

External sources (not redistributed here):

- GADM 4.1 geometries (GeoPackage format required): https://gadm.org/download_world.html (Direct link: [gadm_410-gpkg.zip](https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip)). Unzip and place the `.gpkg` file in `data/`.
- Full GDIS dataset: https://doi.org/10.7927/ZZ3B‑8Y61


### 1. Geocoding (Reference)

*Located in `geocoding/`. These scripts were used to generate the LLM‑GeoDis
dataset.*

- `run_geolocation.py`: Main workflow for LLM‑assisted geoparsing using GPT‑4o.
  It extracts location names from EM‑DAT and maps them to administrative units.
- `gadm_projection.py`: Handles the projection and harmonization of coordinates
  and administrative names to the GADM 4.1 reference.

### 2. Preprocessing & Batching

*Scripts to prepare data for validation.*

- `run_preprocessing_llm.py`: Converts the raw LLM‑GeoDis CSV parts (from
  Zenodo) into standardized GeoPackage (GPKG) batches for comparison.
- `run_preprocessing_gdis.py`: Prepares GDIS data into comparable GPKG batches,
  filtered to match the disaster events present in the EM‑DAT benchmarks.
- `validation/preprocessing.py`: Underlying utilities for batching and spatial
  data cleaning.

### 3. Geometry Comparison & Validation

*The core validation pipeline.*

- `run_validation.py`: The driver script that iterates through all batches,
  benchmarks (GAUL/GDIS), and processing options (dissolved vs. individual
  units).
- `validation/validation.py`: Orchestrates the comparison logic: aligning model
  outputs with benchmarks and invoking metric calculations.
- `validation/geom_indices.py`: Implementation of spatial metrics (Jaccard
  index, containment, geodetic area calculations).
- `validation/io.py`: Robust I/O helpers for reading and writing spatial
  formats (GPKG, CSV).
- `run_all.py`: A master script to run the entire preprocessing and validation
  sequence end‑to‑end.

### 4. Reporting & Visualization

*Notebooks for statistical analysis and figure generation.*

- `main_figures.ipynb`: Generates descriptive statistics and figures regarding
  the LLM‑GeoDis dataset (e.g., coverage, reliability).
- `comparison_figures.ipynb`: Analyzes validation outputs from the `output/`
  folder and generates comparative performance plots.
- `compute_reliability.ipynb`: Focuses on reliability metrics and consensus
  between different geocoding sources.
- `validate_geoparsing.ipynb`: Detailed check of the geoparsing accuracy.

## Figure and Table Reproducibility

To reproduce the figures and tables presented in the manuscript, follow the
mapping below:

| Figure/Table                              | Source Notebook             | Input Data / Requirements                                      |
|:------------------------------------------|:----------------------------|:---------------------------------------------------------------|
| **Dataset Statistics** (Coverage, Counts) | `main_figures.ipynb`        | `LLMGeoDis` CSV parts, `input_emdat.csv`, `reliability_db.csv` |
| **Yearly Trends** (Geometry counts)       | `main_figures.ipynb`        | `reliability_db.csv`                                           |
| **Comparison Metrics** (Jaccard, Overlap) | `comparison_figures.ipynb`  | CSV files in `output/` (generated by `run_all.py`)             |
| **Reliability Analysis**                  | `compute_reliability.ipynb` | `reliability_db.csv`, `LLMGeoDis` batches                      |
| **Geoparsing Validation**                 | `validate_geoparsing.ipynb` | `input_emdat.csv`, LLM outputs                                 |

*Note: Ensure all Zenodo data files are placed in the `data/` folder as
described below before running the notebooks.*

## Python requirements and configuration instructions

### Install Python and dependencies

- Python: 3.13 or newer (see `pyproject.toml`)
- We recommend `uv` for fast, reproducible envs:
    - Install uv: https://docs.astral.sh/uv/getting-started/installation/
    - Create and sync env:
        - `uv venv`
        - `uv sync`

### Place data and configure paths

- Check all input data under `data/` (see “Data files” below).
- Edit `config.toml`:
    - `path.batch_dir`: folder for generated batch GPKGs (local path you
      control).
    - `path.csv_file_dir`: folder where you unzipped the `LLMGeoDis_part*.zip`
      CSV parts (e.g., `data/LLMGeoDis`).
    - `path.emdat_gaul_path`: path to EM-DAT GAUL geometries (e.g.
      `data/geoemdat_gaul.gpkg`).
    - `path.gdis_path`: path to GDIS geometries (GADM-based).
    - `path.emdat_archive_path`: `data/241204_emdat_archive.xlsx` (provided).
    - `path.gdis_disno_path`: `data/gdis_disnos.csv` (provided).
    - `geocoding.api_key`, `geocoding.base_url`: API credentials and
      OpenAI-compatible endpoint. Change both to switch providers (e.g.,
      Mistral, OpenRouter, Azure OpenAI).
    - `geocoding.model`: chat/completions model name (e.g., `gpt-4o`,
      `mistral-large-latest`).
    - `geocoding.temperature`, `geocoding.max_tokens`: optional generation
      controls for the geocoding prompt.
    - `geocoding.gadm_path`: path to GADM 4.1 GeoPackage file for the
      geocoding workflow (e.g. `data/gadm_410-gpkg.gpkg`).
    - `geocoding.input_dir`: folder containing the original EM-DAT Excel files
      to geocode (e.g. `original_files`).
    - `geocoding.geolocated_files_dir`: output folder for geocoded CSVs (e.g.
      `geolocated_files`).
    - `geocoding.log_dir`: folder for geocoding logs and skipped rows (e.g.
      `geolocated_logs`).

## Execution workflows

### 1. Geocoding workflow (Reference)

If you wish to reproduce the geocoding from raw EM-DAT files:
1.  Configure the `[geocoding]` section in `config.toml` (API keys, input/output directories, GADM path).
2.  Run the LLM-assisted geoparsing:
    ```bash
    python geocoding/run_geolocation.py
    ```
3.  Project and harmonize results to GADM 4.1:
    ```bash
    python geocoding/gadm_projection.py
    ```

### 2. Comparison workflow

Provided that the LLM-GeoDis CSV parts have been unzipped into `data/LLMGeoDis/` (or generated via the workflow above):

1.  Configure the `[path]` section in `config.toml` (point to unzipped CSVs, benchmarks, and batch directory).
2.  Run the full validation pipeline:
    ```bash
    python run_all.py
    ```
    Alternatively, you can run the steps separately:
    - `python run_preprocessing_llm.py` (create GPKG batches from LLM CSVs)
    - `python run_preprocessing_gdis.py` (create GPKG batches from GDIS)
    - `python run_validation.py` (run geometry comparison)

3.  Outputs are written to `output/`:
    - `<provider>_<benchmark>_batch<n>.csv`
    - `<provider>_<benchmark>_batch<n>_dissolved.csv` (when dissolving by `DisNo.`)

## Reproducing figures and tables

Once the comparison workflow is complete and the `output/` folder is populated:
1.  Launch Jupyter Notebook: `jupyter notebook`
2.  Open and run the relevant notebooks (e.g., `main_figures.ipynb`, `comparison_figures.ipynb`) as mapped in the [Figure and Table Reproducibility](#figure-and-table-reproducibility) section.

## Miscaleneous notes

- Storage CRS: EPSG:4326. Area computations use geodetic areas by default (see
  `config.toml` and `validation/geom_indices.py`).

## Reuse, licensing and citation

- Code: see `LICENSE` in this repository.
- Data: see individual sources for terms of use (EM‑DAT FAIR Archive, GDIS,
  GADM, Wikidata, OSM, and the Zenodo dataset are subject to their own
  licenses/terms). The Zenodo record lists CC-BY 4.0 for LLM‑GeoDis.

