import logging
from pathlib import Path
from .io import load_llm_csv_batch, parse_geometries

logger = logging.getLogger(__name__)


def llm_batch_filenames(
        batch_numbers: set[int] = set([1, 2, 3, 4, 5])
) -> None:
    """Generate LLM-geocoded CSV files from the EM-DAT archive."""
    for bn in batch_numbers:
        yield f"geoemdat_part{bn}.csv"

def check_llm_batch_files(csv_file_dir: Path, batch_numbers: list[int]):
    """Check that the required LLM-geocoded CSV files are present."""
    csv_files = set(i.name for i in csv_file_dir.glob("*.csv"))
    authorized_files = set(llm_batch_filenames(batch_numbers))
    if not csv_files.issuperset(authorized_files):
        raise ValueError(f"Missing one or more required CSV "
                         f"files: {authorized_files}")
    if csv_files != authorized_files:
        logger.warning(f"Found unexpected CSV files: "
                       f"{csv_files - authorized_files}")
    csv_files = list(csv_files)
    csv_files.sort()
    return csv_files


def make_llm_geocoded_subbatches(
        csv_file_dir: str | Path,
        columns_to_keep: list[str],
        batch_numbers: list[int] = [1, 2, 3, 4, 5],
        keep_disno: list[str] | None = None,
        output_dir: str | Path = Path("output"),
        geometry_columns: str | list[str] = "geometry"
) -> None:
    """Create LLM-geocoded GeoPackage batches."""
    logger.info("Starting LLM-geocoded subbatch creation...")

    csv_file_dir = Path(csv_file_dir)
    csv_files = check_llm_batch_files(csv_file_dir, batch_numbers)
    logger.info(f"Files found in {csv_file_dir}: {csv_files}")
    for bn, csv_file in enumerate(csv_files, start=1):
        logger.info(f"Processing batch {bn}/{len(csv_files)}: {csv_file}")

        if isinstance(geometry_columns, str):
            geometry_columns = [geometry_columns]
        for geom_column in geometry_columns:
            df = load_llm_csv_batch(
                csv_file_path=csv_file_dir / csv_file,
                columns_to_keep=columns_to_keep + [geom_column]
            )
            if keep_disno is not None:
                df = df[df["DisNo."].isin(keep_disno)]
            gdf = parse_geometries(df, geom_column)
            suffix = geom_column.split("_")[-1]
            output_path = output_dir / f"llm_{suffix}_{bn}.gpkg"
            gdf.to_file(output_path)
            logger.info(f"Saved: {output_path}")
            del gdf, df

    logger.info("LLM-geocoded subbatch creation complete.")
