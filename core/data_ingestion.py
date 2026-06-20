"""
AgroSense AI — Data Ingestion Module
Pulls Sentinel-1/2, ERA5, CHIRPS, SRTM data from Google Earth Engine.
All fetching is dynamic — no hardcoded coordinates or dates.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import ee
    GEE_OK = True
except ImportError:
    GEE_OK = False

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    S2_COLLECTION,
    S2_BANDS,
    SCL_CLOUD_VALUES,
    S1_COLLECTION,
    S1_MODE,
    S1_POLARIZATIONS,
    S1_ORBIT,
    S1_SPECKLE_KERNEL,
    ERA5_COLLECTION,
    ERA5_BANDS,
    CHIRPS_COLLECTION,
    SRTM_COLLECTION,
    MODIS_LST_COLLECTION,
    MODIS_LST_BAND,
    MODIS_LST_SCALE,
    GEE_TIMEOUT_S,
    GEE_MAX_PIXELS,
)


def _mask_s2_clouds(image: Any) -> Any:
    """Mask Sentinel-2 clouds using the Scene Classification Layer (SCL) band.

    SCL values corresponding to clouds, cloud shadows, and saturated pixels are
    identified from the SCL_CLOUD_VALUES config constant and masked out so that
    subsequent compositing operations only accumulate clear-sky pixels.

    Args:
        image: An ee.Image from the Sentinel-2 SR Harmonized collection that
               contains an SCL band.

    Returns:
        The input ee.Image with cloud/shadow pixels masked (set to transparent).
    """
    scl = image.select("SCL")
    cloud_mask = scl.neq(SCL_CLOUD_VALUES[0])
    for v in SCL_CLOUD_VALUES[1:]:
        cloud_mask = cloud_mask.And(scl.neq(v))
    return image.updateMask(cloud_mask)


def fetch_sentinel2(
    geometry: Any,
    start_date: str,
    end_date: str,
    cloud_pct: int = 20,
    temporal_resolution: str = "biweekly",
) -> Dict[str, Any]:
    """Fetch Sentinel-2 surface-reflectance composites for a geometry and date range.

    Iterates over non-overlapping time windows of width determined by
    temporal_resolution, collects all qualifying scenes per window, applies
    cloud masking via _mask_s2_clouds, and reduces each window to a median
    composite. When a window contains no qualifying scenes the function expands
    the search by +/-15 days and relaxes the cloud threshold by 20 pp before
    falling back to that wider composite so a gap-free output list is returned.

    Args:
        geometry: ee.Geometry object defining the study area.
        start_date: ISO date string 'YYYY-MM-DD' for the temporal range start.
        end_date: ISO date string 'YYYY-MM-DD' for the temporal range end (exclusive).
        cloud_pct: Maximum scene-level cloud-coverage percentage used to
                   pre-filter the collection before cloud masking (default 20).
        temporal_resolution: Window width - one of 'weekly' (7 days),
                             'biweekly' (15 days), or 'monthly' (30 days).
                             Defaults to 'biweekly'.

    Returns:
        A dictionary with the following keys:
          - 'composites': list[ee.Image] - one median composite per period,
            clipped to geometry, selecting only the spectral bands in S2_BANDS.
          - 'dates': list[tuple[str, str]] - (period_start, period_end) ISO
            date strings for each composite.
          - 'n_composites': int - total number of composites returned.
          - 'scene_counts': list[int] - number of input scenes that contributed
            to each composite (0 when the fallback was used).

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    interval_days: int = {"weekly": 7, "biweekly": 15, "monthly": 30}.get(
        temporal_resolution, 15
    )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    periods: List[Tuple[datetime, datetime]] = []
    current = start_dt
    while current < end_dt:
        period_end = min(current + timedelta(days=interval_days), end_dt)
        periods.append((current, period_end))
        current = period_end

    composites: List[Any] = []
    dates: List[Tuple[str, str]] = []
    scene_counts: List[int] = []

    for p_start, p_end in periods:
        ps = p_start.strftime("%Y-%m-%d")
        pe = p_end.strftime("%Y-%m-%d")

        collection = (
            ee.ImageCollection(S2_COLLECTION)
            .filterBounds(geometry)
            .filterDate(ps, pe)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(_mask_s2_clouds)
            .select(S2_BANDS + ["SCL"])
        )

        count: int = collection.size().getInfo()
        scene_counts.append(count)

        if count == 0:
            logger.warning(
                "No Sentinel-2 scenes for period %s-%s; using +/-15-day fallback.",
                ps,
                pe,
            )
            fallback_start = (p_start - timedelta(days=15)).strftime("%Y-%m-%d")
            fallback_end = (p_end + timedelta(days=15)).strftime("%Y-%m-%d")
            fallback_cloud = min(cloud_pct + 20, 80)

            fallback = (
                ee.ImageCollection(S2_COLLECTION)
                .filterBounds(geometry)
                .filterDate(fallback_start, fallback_end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", fallback_cloud))
                .map(_mask_s2_clouds)
                .select(S2_BANDS)
                .median()
                .clip(geometry)
            )
            composites.append(fallback)
        else:
            composite = collection.select(S2_BANDS).median().clip(geometry)
            composites.append(composite)

        dates.append((ps, pe))

    return {
        "composites": composites,
        "dates": dates,
        "n_composites": len(composites),
        "scene_counts": scene_counts,
    }


def _apply_refined_lee_filter(image: Any) -> Any:
    """Apply a Refined Lee speckle filter to a SAR image in GEE.

    The Refined Lee filter reduces multiplicative speckle noise in SAR imagery
    while preserving edges and point targets. The implementation uses a 3x3
    local statistics window to estimate the local coefficient of variation (CV),
    derives a pixel-wise weighting factor b, and blends the local mean with the
    observed pixel value proportional to b.

    The outer neighbourhood size for computing the scene-level CV2 is controlled
    by the S1_SPECKLE_KERNEL configuration constant (in pixels).

    Args:
        image: ee.Image containing SAR backscatter bands in linear or dB scale.
               Both single-band and multi-band images are supported - the filter
               is applied independently to the full image stack.

    Returns:
        Filtered ee.Image with the same band structure as image.
    """
    kernel_size: int = S1_SPECKLE_KERNEL

    weights_3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel_3 = ee.Kernel.fixed(3, 3, weights_3)

    mean_3 = image.reduceNeighborhood(ee.Reducer.mean(), kernel_3)
    variance_3 = image.reduceNeighborhood(ee.Reducer.variance(), kernel_3)

    cv2_local = variance_3.divide(mean_3.pow(2).add(1e-10))

    half = max(kernel_size // 2, 1)
    cv2_img = (
        image.subtract(mean_3)
        .pow(2)
        .reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(half))
        .divide(mean_3.pow(2).add(1e-10))
    )

    b = cv2_local.divide(cv2_img.add(1e-10)).min(1).max(0)
    filtered = mean_3.add(b.multiply(image.subtract(mean_3)))
    return filtered


def fetch_sentinel1(
    geometry: Any,
    start_date: str,
    end_date: str,
    dates_reference: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Fetch Sentinel-1 GRD SAR composites matching Sentinel-2 temporal windows.

    Filters to IW mode, dual-polarisation (VV+VH), and a single dominant
    relative orbit number for geometric consistency. Each temporal window is
    reduced to a mean composite after applying the Refined Lee speckle filter
    via _apply_refined_lee_filter. If a window contains no scenes the search is
    automatically expanded by +/-7 days.

    Args:
        geometry: ee.Geometry defining the study area.
        start_date: ISO date string 'YYYY-MM-DD' for collection filtering.
        end_date: ISO date string 'YYYY-MM-DD' for collection filtering.
        dates_reference: Optional list of (period_start, period_end) ISO
                         date-string tuples to align SAR composites with
                         Sentinel-2 windows returned by fetch_sentinel2.
                         When None, a single composite spanning the full
                         range is produced.

    Returns:
        A dictionary with:
          - 'composites': list[ee.Image] - one per temporal period,
            bands ['VV', 'VH'] in dB, clipped to geometry.
          - 'dates': list[tuple[str, str]] - period boundaries used.
          - 'dominant_orbit': int | None - the relative orbit number selected
            for consistency (None if no scenes were found).

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", S1_MODE))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", S1_ORBIT))
        .select(["VV", "VH"])
    )

    orbit_numbers: List[int] = (
        collection.aggregate_array("relativeOrbitNumber_start").getInfo()
    )
    dominant_orbit: Optional[int] = None
    if orbit_numbers:
        dominant_orbit = max(set(orbit_numbers), key=orbit_numbers.count)
        collection = collection.filter(
            ee.Filter.eq("relativeOrbitNumber_start", dominant_orbit)
        )
    else:
        logger.warning(
            "No Sentinel-1 scenes found for %s-%s in region.", start_date, end_date
        )

    periods: List[Tuple[str, str]] = dates_reference or [(start_date, end_date)]
    composites: List[Any] = []

    for ps, pe in periods:
        period_col = collection.filterDate(ps, pe)
        n: int = period_col.size().getInfo()

        if n == 0:
            logger.warning(
                "No S1 scenes for %s-%s; expanding window by +/-7 days.", ps, pe
            )
            exp_start = (
                datetime.strptime(ps, "%Y-%m-%d") - timedelta(days=7)
            ).strftime("%Y-%m-%d")
            exp_end = (
                datetime.strptime(pe, "%Y-%m-%d") + timedelta(days=7)
            ).strftime("%Y-%m-%d")
            period_col = collection.filterDate(exp_start, exp_end)

        composite = (
            period_col
            .map(
                lambda img: _apply_refined_lee_filter(img).copyProperties(
                    img, img.propertyNames()
                )
            )
            .mean()
            .clip(geometry)
        )
        composites.append(composite)

    return {
        "composites": composites,
        "dates": periods,
        "dominant_orbit": dominant_orbit,
    }


def fetch_era5(
    geometry: Any,
    start_date: str,
    end_date: str,
    target_scale: int = 100,
) -> Any:
    """Fetch ERA5-LAND daily meteorological data for reference ET0 computation.

    Retrieves temperature, humidity, wind, and radiation variables required by
    the FAO Penman-Monteith equation. Each image is bilinearly resampled to
    target_scale metres so it aligns spatially with satellite imagery.

    Args:
        geometry: ee.Geometry defining the study area.
        start_date: ISO date string 'YYYY-MM-DD'.
        end_date: ISO date string 'YYYY-MM-DD'.
        target_scale: Output pixel size in metres for reprojection (default 100).

    Returns:
        ee.ImageCollection containing the ERA5-LAND bands listed in ERA5_BANDS,
        reprojected to target_scale using bilinear interpolation, clipped to
        geometry.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    collection = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select(ERA5_BANDS)
    )

    def reproject_bilinear(img: Any) -> Any:
        """Bilinearly resample and reproject a single ERA5 image."""
        return (
            img.resample("bilinear")
            .reproject(crs="EPSG:4326", scale=target_scale)
            .clip(geometry)
        )

    return collection.map(reproject_bilinear)


def fetch_chirps(
    geometry: Any,
    start_date: str,
    end_date: str,
) -> Any:
    """Fetch CHIRPS v2.0 daily rainfall data.

    CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
    provides quasi-global, daily precipitation estimates at approximately 5 km
    resolution.

    Args:
        geometry: ee.Geometry defining the study area.
        start_date: ISO date string 'YYYY-MM-DD'.
        end_date: ISO date string 'YYYY-MM-DD'.

    Returns:
        ee.ImageCollection with a single 'precipitation' band (mm/day),
        clipped to geometry.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    return (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select(["precipitation"])
        .map(lambda img: img.clip(geometry))
    )


def fetch_srtm(geometry: Any) -> Dict[str, Any]:
    """Fetch the SRTM 30 m DEM and derive terrain products (slope, aspect).

    Uses ee.Terrain.products to compute slope (degrees from horizontal) and
    aspect (degrees clockwise from north) from the elevation raster.

    Args:
        geometry: ee.Geometry defining the study area.

    Returns:
        Dictionary with three ee.Image values:
          - 'dem': Elevation in metres above sea level.
          - 'slope': Slope angle in degrees (0-90).
          - 'aspect': Aspect in degrees clockwise from north (0-360).

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    dem = ee.Image(SRTM_COLLECTION).clip(geometry)
    terrain = ee.Terrain.products(dem)
    elevation = dem.select("elevation")
    slope = terrain.select("slope")
    aspect = terrain.select("aspect")

    return {"dem": elevation, "slope": slope, "aspect": aspect}


def fetch_modis_lst(
    geometry: Any,
    start_date: str,
    end_date: str,
    dates_reference: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Fetch MODIS Terra Land Surface Temperature (MOD11A1) composites.

    Applies the LST DN-to-Kelvin scale factor defined in MODIS_LST_SCALE and
    renames the output band to 'LST_K'. One mean composite is produced per
    temporal window in dates_reference.

    Args:
        geometry: ee.Geometry defining the study area.
        start_date: ISO date string 'YYYY-MM-DD'.
        end_date: ISO date string 'YYYY-MM-DD'.
        dates_reference: Optional list of (period_start, period_end) tuples to
                         align LST composites with other datasets. When None,
                         a single composite spanning the full range is returned.

    Returns:
        Dictionary with:
          - 'composites': list[ee.Image] - one composite per period, LST in
            Kelvin, clipped to geometry.
          - 'dates': list[tuple[str, str]] - period boundaries.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    collection = (
        ee.ImageCollection(MODIS_LST_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select([MODIS_LST_BAND])
        .map(
            lambda img: img.multiply(MODIS_LST_SCALE)
            .rename("LST_K")
            .clip(geometry)
        )
    )

    periods: List[Tuple[str, str]] = dates_reference or [(start_date, end_date)]
    composites: List[Any] = []

    for ps, pe in periods:
        composite = collection.filterDate(ps, pe).mean()
        composites.append(composite)

    return {"composites": composites, "dates": periods}


def fetch_historical_ndvi_baseline(
    geometry: Any,
    target_month: int,
    baseline_years: List[int],
) -> Dict[str, Any]:
    """Compute per-pixel historical NDVI statistics for a given calendar month.

    Builds a multi-year NDVI baseline used to calculate the Vegetation Condition
    Index (VCI) without hardcoded thresholds. For each year in baseline_years
    the function retrieves all cloud-filtered Sentinel-2 scenes in target_month,
    computes a median NDVI composite, and then aggregates across years to derive
    pixel-wise min, max, and mean.

    Args:
        geometry: ee.Geometry defining the study area.
        target_month: Calendar month number (1=January ... 12=December).
        baseline_years: List of years (e.g. [2018, 2019, 2020, 2021, 2022])
                        to include in the historical baseline.

    Returns:
        Dictionary with three ee.Image values:
          - 'ndvi_min': Per-pixel historical minimum NDVI.
          - 'ndvi_max': Per-pixel historical maximum NDVI.
          - 'ndvi_mean': Per-pixel historical mean NDVI.
        Returns an empty dict and logs a warning when no qualifying imagery
        is found for any of the requested years/months.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("earthengine-api not installed")

    images: List[Any] = []

    for year in baseline_years:
        month_start = f"{year}-{target_month:02d}-01"
        if target_month == 12:
            month_end = f"{year + 1}-01-01"
        else:
            month_end = f"{year}-{target_month + 1:02d}-01"

        col = (
            ee.ImageCollection(S2_COLLECTION)
            .filterBounds(geometry)
            .filterDate(month_start, month_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(_mask_s2_clouds)
            .select(["B8", "B4"])
        )

        n: int = col.size().getInfo()
        if n > 0:
            ndvi = (
                col.map(
                    lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                )
                .median()
            )
            images.append(ndvi)
        else:
            logger.debug(
                "No S2 scenes for historical baseline: year=%d month=%d.",
                year,
                target_month,
            )

    if not images:
        logger.warning(
            "No historical NDVI data found for month %d across years %s.",
            target_month,
            baseline_years,
        )
        return {}

    hist_col = ee.ImageCollection(images)
    return {
        "ndvi_min": hist_col.min(),
        "ndvi_max": hist_col.max(),
        "ndvi_mean": hist_col.mean(),
    }


def get_area_km2(geometry: Any) -> float:
    """Compute the area of a GEE geometry in square kilometres.

    Uses ee.Geometry.area with a 1-metre maximum error to balance accuracy
    with computation speed.

    Args:
        geometry: ee.Geometry whose area is to be measured.

    Returns:
        Area in km2. Returns 0.0 when GEE is not available (offline mode).
    """
    if not GEE_OK:
        return 0.0
    area_m2: float = geometry.area(maxError=1).getInfo()
    return area_m2 / 1_000_000.0
