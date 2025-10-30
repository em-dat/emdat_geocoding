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
    filter_by_disnos,
    dissolve_units,
    list_disno_in_benchmark,
    BenchmarkGeomType,
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
AreaCalculationMethod = Literal["geodetic", "equal_area"]

def compare_geometries(
        gpkg_batch_path: str | Path,
        benchmark: BenchmarkGeomType = "GAUL",
        dissolved_units: bool = False,
        area_calculation_method: AreaCalculationMethod = "geodetic",
        emdat_archive_path: str | Path | None = None,
        emdat_gaul_path: str | Path | None = None,
        gdis_path: str | Path | None = None,
        gdis_disno_path: str | Path | None = None,
        output_dir: str | Path | None = Path("output"),
):
    """
    Compare the geometries from a batch of GeoPackages against a chosen
    benchmark dataset and perform comparison based on specific area
    calculation methods.

    This function processes geometries through several steps: filtering,
    dissolving (if necessary), and comparing with the benchmark dataset.

    The comparison results are then saved to a specified output directory in
    CSV format.

    Parameters
    ----------
    gpkg_batch_path : str or Path
        Path to the batch of GeoPackages containing geometries to process and
        validate.
    benchmark : BenchmarkGeomType, optional, default="GAUL"
        Type of benchmark dataset to use for comparison. Must be one of "GAUL"
        or "GDIS".
    dissolved_units : bool, optional, default=False
        Whether to perform dissolve operation on the geometries before
        validation.
    area_calculation_method : AreaCalculationMethod, optional, default="geodetic"
        Method of calculating areas. Can be "geodetic" (more accurate for
        earth geometries) or another projected area supported method.
    emdat_archive_path : str or Path or None, optional, default=None
        Path to the EM-DAT Archive used for retrieving benchmark DisNo values
        when using "GAUL" as benchmark.
    emdat_gaul_path : str or Path or None, optional, default=None
        Path to the GAUL dataset for benchmarks. Required if "GAUL" is the
        benchmark type.
    gdis_path : str or Path or None, optional, default=None
        Path to the GDIS dataset for benchmarks. Required if "GDIS" is the
        benchmark type.
    gdis_disno_path : str or Path or None, optional, default=None
        Path to a DisNo-specific dataset for GDIS benchmark geometry.
    output_dir : str or Path, optional, default=Path("output")
        Directory where validation results will be saved as a CSV file.

    Raises
    ------
    ValueError
        If an invalid benchmark type is provided.

    Notes
    -----
    1. The function processes and compares geometries corresponding to the
       benchmark dataset's DisNo (disaster numbers) via filtering.
    2. The dissolve operation combines geographical units by specified
       aggregation functions if `dissolved_units` is set to True.
    3. Area calculation during geometry comparison is performed using the
       selected `area_calculation_method`.
    4. The output CSV file contains various metrics computed during geometry
       comparison along with metadata related to disaster numbers, region
       names, and administrative levels.
    """
    logger.info(
        f"Comparing geometries in {gpkg_batch_path} vs. {benchmark} "
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
    gdf_benchmark = filter_by_disnos(gdf_benchmark, disno_list)
    check_geometries(gdf_benchmark["geometry"])
    logger.info(f"{len(gdf_benchmark)} records loaded")

    # Dissolve units
    if dissolved_units:
        aggfunc = {"name": list, "admin_level": list, "admin1": list,
                   "admin2": list}
        gdf_llm = dissolve_units(gdf_llm, aggfunc=aggfunc)

    if benchmark == "GDIS":  # GAUL is already dissolved
        gdf_benchmark = dissolve_units(gdf_benchmark)

    # Perform actual validation
    logger.info(f"Starting geometry validation...")
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

    result_df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, index=False)

    logger.info(f"Validation complete".upper())
    return result_df