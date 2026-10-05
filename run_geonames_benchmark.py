"""GeoNames benchmark driver.

Compares the GeoNames points (geocoding/run_geonames_geocoding.py, Teber et al.
method) with the LLM-GeoDis GADM units, using the two benchmarks of the
validation: EM-DAT GAUL and GDIS. GeoNames gives points, so both methods are
scored with points:

- accuracy: is the location's point inside the benchmark footprint of the same
  event? GeoNames: the GeoNames point. LLM-GeoDis: a point inside each GADM
  polygon (shapely.point_on_surface). GDIS has several unit polygons per event;
  a point counts as inside if it falls in any of them.
- agreement: is the GeoNames point inside an LLM-GeoDis GADM polygon of the same
  event?

Rates are given over all events and over the events located by both methods.
LLM-GeoDis rates are also split by gadm_source when the column is present.

Inputs and paths are read from config.toml. Outputs are written to the
configured validation_output_dir:
- geonames_benchmark_geonames_points.csv: one row per GeoNames location
- geonames_benchmark_llm_points.csv: one row per LLM-GeoDis GADM location
- geonames_benchmark_summary.csv: rates per method, benchmark and subset
"""

import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

logging.basicConfig(
    level=config["logging"]["level"],
    filename=config["logging"]["filename"],
    filemode=config["logging"]["filemode"],
    style=config["logging"]["style"],
    format=config["logging"]["format"],
    datefmt=config["logging"]["datefmt"],
)

CHUNKSIZE = 2000  # rows read at a time; the LLM-GeoDis parts are several GB each
PAIR_BATCH = 20_000  # point-polygon pairs tested at a time, to bound memory
BENCHMARKS = {  # name: (config path key, DisNo. column)
    "gaul": ("emdat_gaul_path", "disno_"),
    "gdis": ("gdis_path", "DisNo."),
}


def points_in_event_polygons(points, poly_disnos, polys, x_col, y_col):
    """For each point, 1 if it lies in any polygon of its own event, else 0.

    NaN where the point has no coordinates or its event has no polygon.
    """
    shapely.prepare(polys)
    poly_index = pd.DataFrame({"DisNo.": poly_disnos, "poly_i": np.arange(len(polys))})
    pts = points.loc[points[x_col].notna(), ["DisNo.", x_col, y_col]]
    pairs = pts.reset_index(names="pt_i").merge(poly_index, on="DisNo.")
    if pairs.empty:
        return pd.Series(np.nan, index=points.index)
    poly_i = pairs["poly_i"].to_numpy()
    xs, ys = pairs[x_col].to_numpy(float), pairs[y_col].to_numpy(float)
    hit = np.zeros(len(pairs), dtype=bool)
    for s in range(0, len(pairs), PAIR_BATCH):
        hit[s:s + PAIR_BATCH] = shapely.intersects_xy(
            polys[poly_i[s:s + PAIR_BATCH]], xs[s:s + PAIR_BATCH], ys[s:s + PAIR_BATCH])
    per_point = pd.Series(hit, index=pairs["pt_i"]).groupby(level=0).max()
    return per_point.astype(float).reindex(points.index)


def read_llm_part(path, events, geonames):
    """Stream one LLM-GeoDis CSV part.

    Returns one point per GADM polygon of the events in scope, and whether each
    GeoNames point of these events falls in one of their GADM polygons.
    """
    columns = ["DisNo.", "name", "admin_level", "admin1", "admin2", "iso3", "geometry_gadm"]
    header = pd.read_csv(path, nrows=0).columns
    if "gadm_source" in header:
        columns.append("gadm_source")
    llm_points, agreement = [], []
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["DisNo."].isin(events) & chunk["geometry_gadm"].notna()]
        polys = shapely.from_wkt(chunk["geometry_gadm"].to_numpy(), on_invalid="ignore")
        valid = ~(shapely.is_missing(polys) | shapely.is_empty(polys))
        chunk, polys = chunk.loc[valid].drop(columns="geometry_gadm"), polys[valid]
        if chunk.empty:
            continue
        rep = shapely.point_on_surface(polys)
        llm_points.append(chunk.assign(x=shapely.get_x(rep), y=shapely.get_y(rep)))

        gn = geonames[geonames["DisNo."].isin(set(chunk["DisNo."]))]
        if not gn.empty:
            hit = points_in_event_polygons(gn, chunk["DisNo."].to_numpy(), polys, "lng", "lat")
            agreement.append(pd.DataFrame({"point_id": gn["point_id"], "in_llm_gadm": hit}))
    return (pd.concat(llm_points, ignore_index=True) if llm_points else pd.DataFrame(),
            pd.concat(agreement, ignore_index=True) if agreement else pd.DataFrame())


def load_benchmark(path, disno_col, events):
    """Read the footprints of the events in scope from a benchmark GeoPackage."""
    values = ",".join(f"'{d}'" for d in sorted(events))
    df = pyogrio.read_dataframe(path, columns=[disno_col], where=f'"{disno_col}" IN ({values})')
    df = df[df.geometry.notna()]
    return df[disno_col].to_numpy(str), df.geometry.to_numpy()


def rates(df, col):
    """Share of points inside, per location (loc_rate) and averaged per event
    (event_rate), and share of events with all points inside (event_all)."""
    d = df[df[col].notna()]
    if d.empty:
        return dict(n_loc=0, n_events=0, loc_rate=np.nan, event_rate=np.nan, event_all=np.nan)
    per_event = d.groupby("DisNo.")[col]
    return dict(n_loc=len(d), n_events=d["DisNo."].nunique(), loc_rate=d[col].mean(),
                event_rate=per_event.mean().mean(), event_all=(per_event.min() == 1).mean())


def summarise(geonames, llm):
    rows = []
    for bm in BENCHMARKS:
        col = f"in_{bm}"
        common = (set(geonames.loc[geonames[col].notna(), "DisNo."])
                  & set(llm.loc[llm[col].notna(), "DisNo."]))
        for subset, keep in [("all", None), ("common", common)]:
            for method, df in [("GeoNames", geonames), ("LLM-GeoDis GADM", llm)]:
                d = df if keep is None else df[df["DisNo."].isin(keep)]
                rows.append(dict(analysis="accuracy", benchmark=bm.upper(), subset=subset,
                                 method=method, **rates(d, col)))
            if "gadm_source" in llm.columns:
                d = llm if keep is None else llm[llm["DisNo."].isin(keep)]
                for source, ds in d.groupby("gadm_source"):
                    rows.append(dict(analysis="accuracy", benchmark=bm.upper(), subset=subset,
                                     method=f"LLM-GeoDis GADM ({source})", **rates(ds, col)))
    rows.append(dict(analysis="agreement", benchmark="LLM-GeoDis GADM", subset="all",
                     method="GeoNames", **rates(geonames, "in_llm_gadm")))
    return pd.DataFrame(rows)


def main():
    logging.info("Running GeoNames benchmark...".upper())
    output_dir = Path(config["path"]["validation_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    geonames = pd.read_csv(config["geonames"]["points_path"], dtype={"DisNo.": str})
    geonames = geonames.reset_index(names="point_id")
    events = set(geonames["DisNo."])
    logging.info(f"GeoNames: {len(geonames)} locations, {geonames['lat'].notna().sum()} points, "
                 f"{len(events)} events")

    llm_parts, agreements = [], []
    csv_dir = Path(config["path"]["csv_file_dir"])
    for bn in config["index"]["batch_numbers"]:
        path = csv_dir / f"LLMGeoDis_part{bn}.csv"
        llm_part, agreement = read_llm_part(path, events, geonames[geonames["lat"].notna()])
        llm_parts.append(llm_part)
        agreements.append(agreement)
        logging.info(f"{path.name}: {len(llm_part)} GADM locations in scope")
    llm = pd.concat(llm_parts, ignore_index=True)
    agreement = pd.concat(agreements, ignore_index=True).groupby("point_id")["in_llm_gadm"].max()
    geonames["in_llm_gadm"] = geonames["point_id"].map(agreement)

    for bm, (path_key, disno_col) in BENCHMARKS.items():
        disnos, polys = load_benchmark(config["path"][path_key], disno_col, events)
        logging.info(f"{bm.upper()}: {len(polys)} footprint polygons")
        geonames[f"in_{bm}"] = points_in_event_polygons(geonames, disnos, polys, "lng", "lat")
        llm[f"in_{bm}"] = points_in_event_polygons(llm, disnos, polys, "x", "y")

    geonames.drop(columns="point_id").to_csv(
        output_dir / "geonames_benchmark_geonames_points.csv", index=False)
    llm.to_csv(output_dir / "geonames_benchmark_llm_points.csv", index=False)
    summarise(geonames, llm).to_csv(output_dir / "geonames_benchmark_summary.csv", index=False)
    logging.info("GeoNames benchmark done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Exception occurred: {e}")
