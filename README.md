# EM-DAT Geocoding - Code and Data (LLM-GeoDis)

This repository contains the code and inputs to reproduce the
LLM-assisted geocoding (LLM-GeoDis) and the geometry comparison/validation
workflows described in the associated manuscript: [Anonymized]

## Overview

Code and data allow reproducing the following steps:

- Geoparse EM-DAT textual locations with GPT-4o into GADM 4.1, OSM, and
  Wikidata administrative units and points.
- Project and harmonize geometries to GADM 4.1.
- Compare and validate against GDIS (GADM-based) and EM-DAT GAUL 2015
  benchmarks.
- Generate descriptive statistics and figures.

## Repository Layout

```text
.
├── data/                  # Raw and preprocessed input data
├── geocoding/             # Scripts for LLM-assisted geoparsing and geocoding
├── validation/            # Scripts for validation and spatial metric logic
├── validation_outputs/    # Results from geometry comparisons
├── run_*.py               # Main driver scripts for the comparison workflow
└── *.ipynb                # Notebooks for analysis and visualization
```

Some additional repository folders are created while running the workflow, based
on the configuration settings in `config.toml`.

### 0. Data Files (`data/`)

*Contains raw and preprocessed data. Key files:*

- `241204_emdat_archive.xlsx`: EM-DAT Archive with GAUL id codes.
- `gdis_disnos.csv`: EM-DAT identifiers geocoded by GDIS.
- `geoemdat_gaul.gpkg`: GeoPackage with EM-DAT-GAUL geometries.
- `LLMGeoDis.csv`: Pre-processed dataset (or unzipped parts
  `LLMGeoDis_part1.csv` to `part5.csv` from Zenodo).
- `input_emdat.csv`: EM-DAT input used for geoparsing reference.
- `reliability_db.csv`: Database with source counts and spatial agreement used
  for reliability and coverage analysis.

### 1. Geocoding (`geocoding/`)

*Workflow to generate the LLM-GeoDis dataset:*

- `run_geolocation.py`: Main GPT-4o geoparsing script.
- `gadm_preprocessing.py`: Prepares GADM layers for the pipeline.
- `gadm_projection.py`: Harmonizes coordinates/names to GADM 4.1.

### 2. Preprocessing & Validation

*Scripts and modules to prepare data and run comparisons:*

- `run_preprocessing_llm.py` & `run_preprocessing_gdis.py`: Convert raw data to
  standardized batches.
- `run_validation.py`: Driver for the geometry comparison pipeline.
- `run_all.py`: Master script for end-to-end execution.
- `validation/`: Package containing spatial metrics (`geom_indices.py`),
  comparison logic (`validation.py`), and I/O helpers (`io.py`).

### 3. Reporting & Visualization

*Notebooks for generating manuscript figures and statistics:*

- `main_figures.ipynb`: Dataset coverage and yearly trends.
- `comparison_figures.ipynb`: Validation results analysis.
- `compute_reliability.ipynb`: Reliability metrics and consensus.
- `validate_geoparsing.ipynb`: Geoparsing accuracy checks.

##### Figure and Table Reproducibility

To reproduce the figures and tables presented in the manuscript, follow the
mapping below:

| Figure/Table                        | Source Notebook             | Input Data / Requirements                                                            |
|:------------------------------------|:----------------------------|:-------------------------------------------------------------------------------------|
| Table 1, Figure 1, Figure 2         | N.A.                        | Descriptive table/figure generated manually                                          |
| Figure 3, 4, A1, A2, B1, B2, B3, B4 | `main_figures.ipynb`        | `input_emdat.csv`, `LLMGeoDis.csv` (or parts), `reliability_db.csv`, GADM 4.1 layers |
| Figure 5, D1, D2                    | `comparison_figures.ipynb`  | `241204_emdat_archive.xlsx`, `validation_outputs/*.csv`                              |
| Figure 6                            | `validate_geoparsing.ipynb` | GADM 4.1 layers (for synthetic sample generation)                                    |
| Figure C1, C2                       | `compute_reliability.ipynb` | `reliability_db.csv`, GADM 4.1 layers                                                |

*Note: Ensure all Zenodo data files are placed in the `data/` folder as
described below before running the notebooks. GADM 4.1 layers refers to a
processed GeoPackage containing `ADM_1` and `ADM_2` layers, which can be
generated from raw GADM 4.1 data using `geocoding/gadm_preprocessing.py`.*

## Python Requirements and Configuration Instructions

### Install Python and Dependencies

- Python: 3.13 or newer (see `pyproject.toml`)
- We recommend `uv` for fast, reproducible envs:
    - Install uv: https://docs.astral.sh/uv/getting-started/installation/
    - Create and sync env:
        - `uv venv`
        - `uv sync`

### Place Data and Configure Paths

- Check all input data under `data/` (see "Data files" above).
- Edit `config.toml` for required paths and API settings. Each parameter is
  documented inline in that file.

## Execution Workflows

### 1. Geocoding Workflow

If you wish to reproduce the geocoding from raw EM-DAT files:

1. Configure the `[geocoding]` section in `config.toml` (API keys, input/output
   directories, GADM path).
2. Preprocess the GADM data to create the administrative layers expected by the
   pipeline:
   ```bash
   python geocoding/gadm_preprocessing.py
   ```
3. Run the LLM-assisted geoparsing:
   ```bash
   python geocoding/run_geolocation.py
   ```
4. Project and harmonize results to GADM 4.1:
   ```bash
   python geocoding/gadm_projection.py
   ```

### 2. Comparison Workflow

Provided that the LLM-GeoDis CSV parts have been unzipped into
`data/LLMGeoDis/` (or generated via the workflow above):

1. Configure the `[path]` section in `config.toml` (point to unzipped CSVs,
   benchmarks, and batch directory).
2. Run the full validation pipeline:
   ```bash
   python run_all.py
   ```
   Alternatively, you can run the steps separately:
    - `python run_preprocessing_llm.py` (create GPKG batches from LLM CSVs)
    - `python run_preprocessing_gdis.py` (create GPKG batches from GDIS)
    - `python run_validation.py` (run geometry comparison)

3. Outputs are written to `validation_outputs/`:
    - `<provider>_<benchmark>_batch<n>.csv`
    - `<provider>_<benchmark>_batch<n>_dissolved.csv` (when dissolving by
      `DisNo.`)

### 3. Reproducing Figures and Tables

Once the comparison workflow is complete and the `validation_outputs/` folder is
populated:

1. Launch Jupyter Notebook: `jupyter notebook`
2. Open and run the relevant notebooks (e.g., `main_figures.ipynb`,
   `comparison_figures.ipynb`) as mapped in
   the [Figure and Table Reproducibility](#figure-and-table-reproducibility)
   section.

## Miscellaneous Notes

- Coordinate Reference System (CRS): EPSG:4326.
- Area computations use geodetic areas by default (see
  `config.toml` and `validation/geom_indices.py`).
- EM-DAT Data Versions: This repository uses two versions of EM-DAT to balance
  redistribution rights with data richness:
    - `241204_emdat_archive.xlsx`: A **FAIR archived version** (limited to 2023)
      used as the stable benchmark for geometry validation (Figures 5, D1, D2).
      It ensures future research uses the same baseline.
    - `input_emdat.csv` / `LLMGeoDis.csv`: These include more recent geocoding (
      extending to 2024) based on a version that cannot be redistributed in
      full downloaded from the EM-DAT data portal. These are used for the main
      geoparsing reference and coverage
      analysis (Figures 3, 4).

## Reuse, Licensing, and Citation

- Code: see `LICENSE` in this repository.
- Data: see individual sources for terms of use (EM-DAT FAIR Archive, GDIS,
  GADM, Wikidata, OSM, and the Zenodo dataset are subject to their own
  licenses/terms). The Zenodo record lists CC-BY 4.0 for LLM-GeoDis.

