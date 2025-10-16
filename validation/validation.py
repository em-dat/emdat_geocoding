import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import geopandas as gpd

from validation.io import check_geometries, load_emdat_archive, load_GAUL
from validation.geom_indices import calculate_geom_indices

logger = logging.getLogger(__name__)

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
    "area_a",
    "area_b",
    "intersection_area",
    "union_area",
    "a_in_b",
    "b_in_a",
    "jaccard",
    "b_contains_a",
    "b_contains_a_properly",
]

LLMGeomType = Literal['gadm', 'osm', 'wiki']
BenchmarkGeomType = Literal['GAUL', 'GDIS']
AreaCalculationMethod = Literal['geodetic', 'equal_area']

def list_disno_in_benchmark(
        benchmark_type: BenchmarkGeomType,
        benchmark_path: str | Path | None = None
) -> list[str]:
    """Return the name of the benchmark geometry column."""
    if benchmark_type == 'GAUL':
        disnos = load_emdat_archive(
            benchmark_path,
            use_columns=["DisNo."],
            geocoded_only=True
        )['DisNo.'].to_list()
    elif benchmark_type == 'GDIS':
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid benchmark type: {benchmark_type}")
    return disnos

def validate_geometries(
        gpkg_subbatch_path: str | Path,
        benchmark: BenchmarkGeomType = 'GAUL',
        dissolve_units: bool = False,
        area_calculation_method: AreaCalculationMethod = 'geodetic',
        emdat_archive_path: str | Path | None = None,
        emdat_gaul_path: str | Path | None = None,
        output_dir: str | Path = Path("output")
):
    """Validate the geometry of the GeoDataFrame."""
    logger.info(
        f"Validating geometries in {gpkg_subbatch_path} vs. {benchmark} "
        f"(Dissolve: {dissolve_units})")

    gpkg_subbatch_path = Path(gpkg_subbatch_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    metadata = gpkg_subbatch_path.stem.split('_')
    geom_type = '_'.join(metadata[:2])
    batch_number = metadata[-1]

    gdf_llm = gpd.read_file(gpkg_subbatch_path)
    logger.info(f"{len(gdf_llm)} records loaded")

    if benchmark == 'GAUL':
        disnos_benchmark = list_disno_in_benchmark(benchmark, emdat_archive_path)
    elif benchmark == 'GDIS':
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid benchmark type: {benchmark}")

    gdf_llm = gdf_llm[gdf_llm['DisNo.'].isin(disnos_benchmark)]
    logger.info(f"{len(gdf_llm)} records filtered based on Dis No.")

    check_geometries(gdf_llm['geometry'])
    disno_list = gdf_llm['DisNo.'].unique().tolist()

    if dissolve_units:
        logger.info("Dissolving units...")
        gdf_llm = gdf_llm.dissolve(
            by='DisNo.',
            aggfunc={
                'name': list,
                'admin_level': list,
                'admin1': list,
                'admin2': list
            }
        )
        gdf_llm.reset_index(inplace=True)

        logger.info(f"{len(gdf_llm)} records after dissolving")

    logger.info(f"Loading {benchmark} geometries...")
    if benchmark == 'GAUL':
        gdf_benchmark = load_GAUL(
            emdat_gaul_path,
            disno=disno_list,
            keep_columns=['DisNo.', 'geometry']
        )
    elif benchmark == 'GDIS':
        raise NotImplementedError

    check_geometries(gdf_benchmark['geometry'])

    logger.info(f"{len(gdf_benchmark)} records loaded")
    logging.info(f"Starting geometry validation...")

    records = []
    geom_dict = dict(zip(gdf_benchmark["DisNo."], gdf_benchmark["geometry"]))
    for ix, row in gdf_llm.iterrows():
        geom_a = row['geometry']
        geom_b = geom_dict.get(row["DisNo."])
        indices: dict[str, float] = calculate_geom_indices(
            geom_a, geom_b,
            method=area_calculation_method,
            shapely_make_valid=False,
            check_geometry=False
        )
        results = [
            row['DisNo.'],
            row['name'],
            row['admin_level'],
            row['admin1'],
            row['admin2'],
            geom_type,
            benchmark,
            batch_number,
            area_calculation_method,
            ] + list(indices.values())
        records.append(results)

    output_filename = (f"{geom_type}_{benchmark.lower()}_batch{batch_number}"
                       f"{'_dissolved' if dissolve_units else ''}"
                       f".csv")
    output_path = output_dir / output_filename
    logging.info(f"Saving results to {output_path}")
    result_df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    result_df.to_csv(output_path, index=False)
    logging.info(f"Validation complete".upper())


