from typing import Literal
from dataclasses import dataclass

from pyproj import Transformer, Geod
from shapely import make_valid
from shapely.geometry import Point, Polygon, MultiPolygon, base
from shapely.ops import transform

# WGS84 ellipsoid
GEOD_WGS84 = Geod(ellps="WGS84")
TRANSFORMER_4326_TO_6933 = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

@dataclass(frozen=True)
class GeomIndices:
    # areas (m^2)
    area_a: float
    area_b: float
    intersection_area: float
    union_area: float

    # ratios
    a_in_b: float
    b_in_a: float
    jaccard: float

    # topology flags
    a_contains_b: bool
    b_contains_a: bool
    a_contains_b_properly: bool
    b_contains_a_properly: bool

def _project_to_equal_area(geom: base.BaseGeometry) -> base.BaseGeometry:
    """Project a geometry to equal area projection"""
    return transform(TRANSFORMER_4326_TO_6933.transform, geom)

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
    if geom.geom_type not in ["Point", "Polygon", "MultiPolygon"]:
        raise TypeError('Only Point, Polygon and MultiPolygon are supported.')
    if not geom.is_valid:
        raise ValueError('Geometry is not valid.')


def calculate_geom_indices(
        geom_a: Point | Polygon | MultiPolygon,
        geom_b: Point | Polygon | MultiPolygon,
        method: Literal["geodetic", "equal_area"] = "geodetic",
        shapely_make_valid: bool = True,
        check_geometry: bool = True
) -> dict[str, float | bool]:
    """
    Calculate areas and geometric indices between two geometries with a
    selectable area method. Area calculation returns 0. for Point geometries.
    """

    if shapely_make_valid:
        geom_a = make_valid(geom_a)
        geom_b = make_valid(geom_b)

    if check_geometry:
        _check_geometry(geom_a)
        _check_geometry(geom_b)

    is_any_point = isinstance(geom_a, Point) or isinstance(geom_b, Point)

    a_contains_b = geom_a.contains(geom_b)
    b_contains_a = geom_b.contains(geom_a)
    a_contains_b_properly = geom_a.contains_properly(geom_b)
    b_contains_a_properly = geom_b.contains_properly(geom_a)

    if method == "equal_area":
        geom_a = _project_to_equal_area(geom_a)
        geom_b = _project_to_equal_area(geom_b)
        area_a = geom_a.area
        area_b = geom_b.area
    else:  # geodetic by default
        area_a = _geodetic_area(geom_a)
        area_b = _geodetic_area(geom_b)

    if is_any_point:
        inter_area = 0.0
        union_area = 0.0
        a_in_b = 0.0
        b_in_a = 0.0
        jaccard = 0.0
    else:
        inter_geom = geom_a.intersection(geom_b)
        if method == "equal_area":
            inter_area = inter_geom.area if not inter_geom.is_empty else 0.0
        else:  # geodetic by default
            inter_area = _geodetic_area(
                inter_geom) if not inter_geom.is_empty else 0.0
        union_area = max(0., area_a + area_b - inter_area)
        a_in_b = (inter_area / area_a) if area_a > 0 else 0.0
        b_in_a = (inter_area / area_b) if area_b > 0 else 0.0
        jaccard = (inter_area / union_area) if union_area > 0 else 0.0

    return GeomIndices(
        area_a=area_a,
        area_b=area_b,
        intersection_area=inter_area,
        union_area=union_area,
        a_in_b=a_in_b,
        b_in_a=b_in_a,
        jaccard=jaccard,
        a_contains_b=a_contains_b,
        b_contains_a=b_contains_a,
        a_contains_b_properly=a_contains_b_properly,
        b_contains_a_properly=b_contains_a_properly,
    ).__dict__
