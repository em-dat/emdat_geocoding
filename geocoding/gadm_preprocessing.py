"""
Preprocessing utilities for GADM data, including ensuring valid geometries,
and dissolving administrative levels to ADM_1 and ADM_2 primarily to save
memory and improve performance in later steps.
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path

import geopandas as gpd
from shapely.validation import make_valid


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def ensure_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    # Fix invalid geometries robustly
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    return gdf


def dissolve_level(src: gpd.GeoDataFrame, by_field: str, keep_fields: list[str]) -> gpd.GeoDataFrame:
    if by_field not in src.columns:
        raise KeyError(f"Required field '{by_field}' not found in source data. Available: {list(src.columns)}")

    # Build aggregation mapping: take the first non-null for attributes we want to keep
    agg_map = {col: "first" for col in keep_fields if col in src.columns and col != by_field}

    # For dissolve we also want to retain the key itself (as_index=False)
    dissolved = src.dissolve(by=by_field, as_index=False, aggfunc=agg_map)

    # Keep only required columns + geometry
    keep = [c for c in [by_field] + keep_fields if c in dissolved.columns]
    dissolved = dissolved[keep + ["geometry"]]
    return ensure_valid_geometries(dissolved)


essential_fields = [
    # Minimal set used by run_geolocation.py / gadm_projection.py
    "GID_0",      # country alpha-3 per GADM naming
    "COUNTRY",    # country name used to build ADMIN0
    "NAME_1",     # admin1 name
    "NAME_2",     # admin2 name
]


def preprocess_gadm_flat_table(src_path: str, out_path: str) -> None:
    """
    Create a GPKG with layers ADM_1 and ADM_2 from a flat GADM table that contains
    multiple admin levels as columns (GID_1/NAME_1, GID_2/NAME_2, ...).
    This matches what run_geolocation.py expects to read later: layers named 'ADM_1' and 'ADM_2'.
    """
    print(f"Reading source GADM file: {src_path}")

    # Determine layers present
    try:
        layers = gpd.io.file.fiona.listlayers(src_path)
    except Exception:
        layers = None

    if layers and ("ADM_1" in layers and "ADM_2" in layers):
        # Already in expected format — copy out to ensure consistent schema
        print("Detected layers 'ADM_1' and 'ADM_2' — copying to destination with minimal fields.")
        adm1 = gpd.read_file(src_path, layer="ADM_1")
        adm2 = gpd.read_file(src_path, layer="ADM_2")
    else:
        # Single flat layer (or unknown naming) — read the first/only layer
        if layers and len(layers) > 0:
            base_layer = layers[0]
            print(f"Detected non-standard layers: {layers}. Using base layer '{base_layer}'.")
            base = gpd.read_file(src_path, layer=base_layer)
        else:
            print("No explicit layers detected — reading default.")
            base = gpd.read_file(src_path)

        base = ensure_valid_geometries(base)

        missing = [c for c in ["GID_1", "NAME_1", "GID_2", "NAME_2", "GID_0", "COUNTRY"] if c not in base.columns]
        if missing:
            raise KeyError(
                "Source file is missing required GADM columns: " + ", ".join(missing) +
                "\nAvailable columns: " + ", ".join(base.columns)
            )

        # Construct ADM_1 and ADM_2 via dissolve
        print("Dissolving to ADM_1 (by 'GID_1')...")
        adm1 = dissolve_level(
            base,
            by_field="GID_1",
            keep_fields=["GID_0", "COUNTRY", "NAME_1"],
        )
        # Ensure NAME_1 is present
        if "NAME_1" not in adm1.columns and "NAME_1" in base.columns:
            adm1["NAME_1"] = adm1["GID_1"].map(base.set_index("GID_1")["NAME_1"])  # best-effort

        print("Dissolving to ADM_2 (by 'GID_2')...")
        adm2 = dissolve_level(
            base,
            by_field="GID_2",
            keep_fields=["GID_0", "COUNTRY", "NAME_1", "NAME_2"],
        )
        # Ensure NAME_2 exists
        if "NAME_2" not in adm2.columns and "NAME_2" in base.columns:
            adm2["NAME_2"] = adm2["GID_2"].map(base.set_index("GID_2")["NAME_2"])  # best-effort

    # Trim to minimal set expected later
    def trim(gdf: gpd.GeoDataFrame, level: int) -> gpd.GeoDataFrame:
        base_cols = ["GID_0", "COUNTRY", "NAME_1"]
        if level == 2:
            base_cols += ["NAME_2"]
        cols = [c for c in base_cols if c in gdf.columns]
        return gdf[cols + ["geometry"]].copy()

    adm1 = trim(adm1, level=1)
    adm2 = trim(adm2, level=2)

    # Write to GPKG with layers ADM_1 and ADM_2
    print(f"Writing processed GADM layers to: {out_path}")
    # Remove existing file to avoid layer clashes
    if os.path.exists(out_path):
        os.remove(out_path)

    adm1.to_file(out_path, driver="GPKG", layer="ADM_1")
    adm2.to_file(out_path, driver="GPKG", layer="ADM_2")

    print("Done. Layers written:")
    try:
        out_layers = gpd.io.file.fiona.listlayers(out_path)
        print(out_layers)
    except Exception:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare GADM GPKG with ADM_1 and ADM_2 layers expected by the geocoding pipeline.")
    parser.add_argument("--input", "-i", help="Path to source GADM .gpkg (flat table or layered). If omitted, uses geocoding.gadm_path from config.toml")
    parser.add_argument("--output", "-o", help="Path to output processed .gpkg. If omitted, writes alongside input as <name>_adm.gpkg and prints path.")
    parser.add_argument("--config", "-c", default="config.toml", help="Path to config.toml (default: config.toml)")

    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))

    src = args.input or cfg["geocoding"]["gadm_path"]
    src = os.path.abspath(src)

    if not os.path.exists(src):
        print(f"ERROR: Source GADM file not found: {src}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out = args.output
    else:
        p = Path(src)
        out = str(p.with_name(p.stem + "_adm.gpkg"))

    preprocess_gadm_flat_table(src, out)

    print("\nIMPORTANT:")
    print("- Update config.toml -> [geocoding].gadm_path to point to the processed file if needed:")
    print(f"  {out}")


if __name__ == "__main__":
    main()
