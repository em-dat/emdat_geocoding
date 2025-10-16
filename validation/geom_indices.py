from typing import Literal

import numpy as np
from pyproj import Transformer, Geod
from shapely import make_valid
from shapely.geometry import Point, Polygon, MultiPolygon, base
from shapely.ops import transform

# WGS84 ellipsoid
GEOD_WGS84 = Geod(ellps="WGS84")


def _geodetic_area(geom: base.BaseGeometry, geod: Geod = GEOD_WGS84) -> float:
    """
    Compute the geodetic (ellipsoidal) area of a geometry in square meters.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Geometry in lon,lat coordinates (EPSG:4326).

    Returns
    -------
    float
        Absolute area in square meters.
    """
    area, _ = geod.geometry_area_perimeter(geom)
    return abs(area)

def _check_geometry(geom: base.BaseGeometry):
    """Basic checks to make sure the input geometry is correctly preprocessed"""
    if geom.is_empty:
        raise ValueError('Geometry is empty.')
    if geom.geom_type not in ["Polygon", "MultiPolygon"]:
        raise TypeError('Only Polygon and MultiPolygon are supported.')
    if not geom.is_valid:
        raise ValueError('Geometry is not valid.')

def calculate_geom_indices(
        geom_a: Polygon | MultiPolygon,
        geom_b: Polygon | MultiPolygon,
        method: Literal["geodetic", "equal_area"] = "geodetic",
        area_crs: str = "EPSG:6933",
        shapely_make_valid: bool = True,
        check_geometry: bool = True,
        float_precision: int | None = None,
) -> dict[str, float]:
    """
    Calculate areas and geometric indices between two geometries with a
    selectable area method.

    Parameters
    ----------
    geom_a, geom_b : Polygon or MultiPolygon
        Input geometries. Must be in EPSG:4326 (lon,lat).
    method : Literal["geodetic", "equal_area"]
        - "geodetic": use ellipsoidal areas on WGS84 via pyproj.Geod
        - "equal_area": reproject to an equal-area CRS and use planar areas
    area_crs : str, default="EPSG:6933"
        Equal-area CRS used for area computations (when method="equal_area").
    float_precision : int or None
        If given, the function returns round results to this many decimals.

    Returns
    -------
    dict : dict[str, float]
        a dictionary with the following keys: 'area_a', 'area_b',
        'intersection_area', 'union_area', 'a_in_b', 'b_in_a', 'jaccard'
    """

    if shapely_make_valid:
        geom_a = make_valid(geom_a)
        geom_b = make_valid(geom_b)

    if check_geometry:
        _check_geometry(geom_a)
        _check_geometry(geom_b)

    if method == "equal_area":
        transformer = Transformer.from_crs("EPSG:4326", area_crs,
                                           always_xy=True)
        project = lambda g: transform(transformer.transform, g)
        ga = project(geom_a)
        gb = project(geom_b)

        area_a = ga.area
        area_b = gb.area
        inter_geom = ga.intersection(gb)
        inter_area = inter_geom.area if not inter_geom.is_empty else 0.0
    else:  # geodetic by default
        inter_geom = geom_a.intersection(geom_b)
        area_a = _geodetic_area(geom_a)
        area_b = _geodetic_area(geom_b)
        inter_area = _geodetic_area(
            inter_geom) if not inter_geom.is_empty else 0.0

    union_area = max(0.0, area_a + area_b - inter_area)
    a_in_b = (inter_area / area_a) if area_a > 0 else 0.0
    b_in_a = (inter_area / area_b) if area_b > 0 else 0.0
    jaccard = (inter_area / union_area) if union_area > 0 else 0.0

    results = {
        "area_a": area_a,
        "area_b": area_b,
        "intersection_area": inter_area,
        "union_area": union_area,
        "a_in_b": a_in_b,
        "b_in_a": b_in_a,
        "jaccard": jaccard,
        "b_contains_a": geom_b.contains(geom_b),
        "b_contains_a_properly": geom_b.contains_properly(geom_b),
    }

    if float_precision is not None:
        results = {k: round(v, float_precision) for k, v in results.items()}

    return results
