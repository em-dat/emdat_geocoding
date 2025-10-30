import logging
from dataclasses import fields, asdict
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd

from validation.geom_indices import calculate_geom_indices, GeomIndices
from validation.io import (
    check_geometries,
    load_benchmark,
    make_batch,
    dissolve_units,
    list_disno_in_benchmark,
)

logger = logging.getLogger(__name__)

GEOMINDICES_FIELDS = [f.name for f in fields(GeomIndices)]
OUTPUT_COLUMNS = [
                     "dis_no",
                     "name",
                     "admin_level",
                     "admin1",
                     "admin2",
                     "geom_type_a",
                     "geom_type_b",
                     "batch_number",
                     "area_calculation_method",
                 ] + GEOMINDICES_FIELDS

LLMGeomType = Literal["gadm", "osm", "wiki"]
BenchmarkGeomType = Literal["GAUL", "GDIS"]
AreaCalculationMethod = Literal["geodetic", "equal_area"]

def validate_geometries(
        gpkg_batch_path: str | Path,
        benchmark: BenchmarkGeomType = "GAUL",
        dissolved_units: bool = False,
        area_calculation_method: AreaCalculationMethod = "geodetic",
        emdat_archive_path: str | Path | None = None,
        emdat_gaul_path: str | Path | None = None,
        gdis_path: str | Path | None = None,
        gdis_disno_path: str | Path | None = None,
        output_dir: str | Path = Path("output"),
):
    """Validate the geometry of the GeoDataFrame."""
    logger.info(
        f"Validating geometries in {gpkg_batch_path} vs. {benchmark} "
        f"(Dissolve: {dissolved_units})"
    )
    gpkg_batch_path = Path(gpkg_batch_path)
    # Get and use metadata
    metadata = gpkg_batch_path.stem.split("_")
    geom_type = "_".join(metadata[:2])
    batch_number = metadata[-1]

    # Get disnos of benchmark file
    if benchmark == "GAUL":
        benchmark_path = Path(emdat_gaul_path)
        disnos_benchmark = list_disno_in_benchmark(benchmark,
                                                   emdat_archive_path)
    elif benchmark == "GDIS":
        benchmark_path = Path(gdis_path)
        disnos_benchmark = list_disno_in_benchmark(benchmark, gdis_disno_path)
    else:
        raise ValueError(f"Invalid benchmark type: {benchmark}")

    # Load model gdf
    gdf_llm = gpd.read_file(gpkg_batch_path)
    logger.info(f"{len(gdf_llm)} records loaded")

    gdf_llm = gdf_llm[gdf_llm["DisNo."].isin(disnos_benchmark)]
    logger.info(f"{len(gdf_llm)} records filtered based on Dis No.")
    check_geometries(gdf_llm["geometry"])
    disno_list = gdf_llm["DisNo."].unique().tolist()

    # Load benchmark gdf and make batch corresponding to gdf_llm
    gdf_benchmark = load_benchmark(benchmark, benchmark_path,
                                   keep_columns=["DisNo.", "geometry"])
    gdf_benchmark = make_batch(gdf_benchmark, disno_list)
    check_geometries(gdf_benchmark["geometry"])
    logger.info(f"{len(gdf_benchmark)} records loaded")

    # Dissolve units (functions takes care to only dissolve what is dissolvable)
    if dissolved_units:
        aggfunc = {"name": list, "admin_level": list, "admin1": list,
                   "admin2": list}
        gdf_llm = dissolve_units(gdf_llm, aggfunc=aggfunc)

    if benchmark == "GDIS":  # GAUL is already dissolved
        gdf_benchmark = dissolve_units(gdf_benchmark)

    # Perform actual validation
    logging.info(f"Starting geometry validation...")
    records = []
    geom_dict = dict(zip(gdf_benchmark["DisNo."], gdf_benchmark["geometry"]))
    for ix, row in gdf_llm.iterrows():
        geom_a = row["geometry"]
        geom_b = geom_dict.get(row["DisNo."])
        indices: GeomIndices = calculate_geom_indices(
            geom_a,
            geom_b,
            method=area_calculation_method,
            shapely_make_valid=False,
            check_geometry=False,
        )
        metrics = asdict(indices)
        results = [
            row["DisNo."],
            row["name"],
            row["admin_level"],
            row["admin1"],
            row["admin2"],
            geom_type,
            benchmark,
            batch_number,
            area_calculation_method
        ] + [metrics[f] for f in GEOMINDICES_FIELDS]
        records.append(results)

    # Save validation results
    output_filename = (
        f"{geom_type}_{benchmark.lower()}_batch{batch_number}"
        f"{'_dissolved' if dissolved_units else ''}"
        f".csv"
    )
    output_dir = Path(output_dir)
    output_path = output_dir / output_filename
    logger.info(f"Saving results to {output_path}")
    result_df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Validation complete".upper())
