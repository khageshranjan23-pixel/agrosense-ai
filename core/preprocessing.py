"""
AgroSense AI — Preprocessing Module
Atmospheric correction verification, speckle filtering, compositing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import ee
    GEE_OK = True
except ImportError:
    GEE_OK = False


def validate_s2_data_quality(
    composites: List[Any],
    dates: List[Tuple[str, str]],
    scene_counts: List[int],
    min_scenes: int = 3,
) -> Dict[str, Any]:
    """Assess data quality of Sentinel-2 composites.

    Iterates over the list of composites and assigns a quality flag to each
    period based on the number of contributing scenes. Periods with zero scenes
    are flagged as 'no_data'; periods below min_scenes are flagged 'low_data';
    all others are flagged 'good'. Human-readable warnings are collected for
    downstream logging or reporting.

    Args:
        composites: List of ee.Image composites, one per temporal period.
        dates: List of (period_start, period_end) ISO date-string tuples,
               aligned 1-to-1 with composites.
        scene_counts: Number of input scenes that contributed to each composite.
                      A value of 0 indicates the fallback path was used.
        min_scenes: Minimum number of scenes considered acceptable for a
                    reliable median composite (default 3).

    Returns:
        Dictionary with:
          - 'quality_flags': list[str] - 'good', 'low_data', or 'no_data' per period.
          - 'warnings': list[str] - human-readable descriptions of quality issues.
          - 'good_periods': list[int] - indices of periods with at least one scene.
          - 'coverage_pct': float - percentage of periods rated 'good' or 'low_data'.
          - 'total_periods': int - total number of periods evaluated.
    """
    quality_flags: List[str] = []
    warnings: List[str] = []
    good_periods: List[int] = []

    for i, (count, (ps, pe)) in enumerate(zip(scene_counts, dates)):
        if count == 0:
            quality_flags.append("no_data")
            warnings.append(
                f"Period {ps}-{pe}: No cloud-free scenes - using fallback composite."
            )
        elif count < min_scenes:
            quality_flags.append("low_data")
            warnings.append(
                f"Period {ps}-{pe}: Only {count} scene(s) - composite may be noisy."
            )
            good_periods.append(i)
        else:
            quality_flags.append("good")
            good_periods.append(i)

    coverage_pct = len(good_periods) / max(len(composites), 1) * 100.0

    return {
        "quality_flags": quality_flags,
        "warnings": warnings,
        "good_periods": good_periods,
        "coverage_pct": coverage_pct,
        "total_periods": len(composites),
    }


def scale_s2_reflectance(image: Any) -> Any:
    """Scale Sentinel-2 SR bands from integer DN to float reflectance (0-1).

    The Sentinel-2 SR Harmonized collection stores optical band values as
    16-bit integers in the range 0-10000, representing surface reflectance
    scaled by a factor of 10000. This function applies the inverse scaling so
    that all subsequent index calculations operate on physically meaningful
    reflectance values.

    Args:
        image: ee.Image with Sentinel-2 SR optical bands in integer DN units.

    Returns:
        ee.Image with the same optical bands multiplied by 0.0001, with all
        original image properties preserved.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("GEE not available")

    optical_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    return ee.Image(image.select(optical_bands).multiply(0.0001).copyProperties(
        image, image.propertyNames()
    ))


def scale_s2_composite_list(composites: List[Any]) -> List[Any]:
    """Apply reflectance scaling to every composite in a list.

    Convenience wrapper around scale_s2_reflectance that processes an entire
    list of composites as returned by fetch_sentinel2, returning a new list
    in the same order.

    Args:
        composites: List of ee.Image objects with integer-scaled S2 SR bands.

    Returns:
        New list of ee.Image objects with reflectance-scaled (0-1) bands.
    """
    return [scale_s2_reflectance(img) for img in composites]


def apply_topographic_correction(
    image: Any,
    dem: Any,
    solar_zenith_deg: float = 45.0,
    solar_azimuth_deg: float = 180.0,
) -> Any:
    """Apply C-correction topographic normalization to an optical image.

    Reduces per-pixel illumination variation caused by terrain slope and aspect
    using the empirical C-correction algorithm (Teillet et al., 1982). The
    correction scales each band by (cos_zen + C) / (IL + C) where IL is the
    local illumination angle computed from slope, aspect, and solar geometry.
    A per-band C constant of 0.1 is used as a computationally efficient proxy
    that avoids per-band linear regression while preventing overcorrection in
    low-illumination areas.

    Args:
        image: ee.Image with optical reflectance bands scaled to 0-1.
        dem: ee.Image containing SRTM elevation in metres (used to derive
             slope and aspect internally via ee.Terrain.products).
        solar_zenith_deg: Solar zenith angle in degrees (0=overhead, 90=horizon).
                          Defaults to 45.0 degrees.
        solar_azimuth_deg: Solar azimuth angle in degrees clockwise from north.
                           Defaults to 180.0 (south-facing sun).

    Returns:
        Topographically corrected ee.Image with the same band names as image
        and all original image properties preserved.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("GEE not available")

    terrain = ee.Terrain.products(dem)
    slope_rad = terrain.select("slope").multiply(np.pi / 180.0)
    aspect_rad = terrain.select("aspect").multiply(np.pi / 180.0)

    zen_rad: float = solar_zenith_deg * np.pi / 180.0
    az_rad: float = solar_azimuth_deg * np.pi / 180.0

    cos_zen: float = float(np.cos(zen_rad))
    sin_zen: float = float(np.sin(zen_rad))

    # Cosine of the local illumination angle IL
    cos_il = (
        slope_rad.cos().multiply(cos_zen)
        .add(
            slope_rad.sin()
            .multiply(sin_zen)
            .multiply(aspect_rad.subtract(az_rad).cos())
        )
    )

    def correct_band(band_name: str) -> Any:
        """Apply C-correction to a single named band."""
        band = image.select([band_name])
        c_factor = ee.Number(0.1)
        corrected = band.multiply(cos_zen + 0.1).divide(cos_il.add(c_factor))
        return corrected.rename(band_name)

    bands: List[str] = image.bandNames().getInfo()
    corrected_bands = [correct_band(b) for b in bands]
    return ee.Image(ee.Image.cat(corrected_bands).copyProperties(image, image.propertyNames()))


def compute_pixel_sample(
    image: Any,
    geometry: Any,
    scale: int,
    max_pixels: int = 50000,
    seed: int = 42,
) -> Optional[Any]:
    """Extract a stratified pixel sample from an image for ML training.

    Uses ee.Image.sample to draw up to max_pixels pixels from the image
    within the bounding geometry. Point geometries are included so that the
    resulting FeatureCollection can be exported or used directly as training
    data. tileScale=4 is applied to avoid exceeding GEE memory limits on
    large images.

    Args:
        image: ee.Image to sample. All bands present in the image are included
               in the output features.
        geometry: ee.Geometry bounding the sampling area.
        scale: Nominal pixel resolution in metres at which to sample.
        max_pixels: Maximum number of pixels to include in the sample
                    (default 50000).
        seed: Integer random seed for reproducible sampling (default 42).

    Returns:
        ee.FeatureCollection of sampled point features, each carrying the
        image band values as properties. Returns None if sampling fails and
        logs the exception at ERROR level.
    """
    if not GEE_OK:
        return None
    try:
        sample = image.sample(
            region=geometry,
            scale=scale,
            numPixels=max_pixels,
            seed=seed,
            geometries=True,
            tileScale=4,
        )
        return sample
    except Exception as exc:
        logger.error("Pixel sampling failed: %s", exc)
        return None


def stack_temporal_composites(
    composites: List[Any],
    band_prefix: str = "t",
) -> Any:
    """Stack a list of single-time-step images into a multi-band temporal image.

    Renames each input image's bands with a zero-padded time-step index and
    the given prefix before concatenating all bands into a single ee.Image.
    This is required for feeding temporal feature stacks into classifiers that
    expect all features in a single multi-band image.

    For example, with band_prefix='t' and composites containing bands
    ['NDVI', 'EVI'] at two time steps, the output will have bands:
    ['t01_NDVI', 't01_EVI', 't02_NDVI', 't02_EVI'].

    Args:
        composites: List of ee.Image objects, each representing one time step.
                    All images must share the same band names.
        band_prefix: Prefix string prepended to each renamed band
                     (default 't').

    Returns:
        Single ee.Image with all temporal bands concatenated in time order.

    Raises:
        RuntimeError: If the earthengine-api package is not installed.
    """
    if not GEE_OK:
        raise RuntimeError("GEE not available")

    renamed: List[Any] = []
    for i, img in enumerate(composites):
        suffix = f"{i + 1:02d}"
        band_names: List[str] = img.bandNames().getInfo()
        new_names = [f"{band_prefix}{suffix}_{b}" for b in band_names]
        renamed.append(img.rename(new_names))

    return ee.Image.cat(renamed)


def generate_demo_ndvi_series(n_timesteps: int = 12) -> List[np.ndarray]:
    """Generate synthetic NDVI time series for offline/demo mode.

    Simulates a realistic single-season crop phenology curve using a Gaussian
    bell centred at mid-season (t=0.5). Random spatial variation drawn from
    a normal distribution is added to each 10x10-pixel map to produce
    plausible field heterogeneity. A fixed RNG seed (42) guarantees
    reproducibility across runs.

    Args:
        n_timesteps: Number of equally-spaced time steps to generate
                     (default 12, representing biweekly composites over ~6 months).

    Returns:
        List of n_timesteps 2-D numpy arrays, each of shape (10, 10) and
        dtype float64, with NDVI values clipped to [0.0, 1.0].
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, n_timesteps)

    # Gaussian phenology curve: rises to ~0.8 at mid-season, returns to ~0.2
    ndvi_curve = 0.2 + 0.6 * np.exp(-((t - 0.5) ** 2) / (2.0 * 0.15 ** 2))

    series: List[np.ndarray] = []
    for ndvi_val in ndvi_curve:
        spatial_noise = rng.normal(0.0, 0.05, (10, 10))
        ndvi_map = np.clip(ndvi_val + spatial_noise, 0.0, 1.0)
        series.append(ndvi_map)

    return series


def generate_demo_sar_series(n_timesteps: int = 12) -> Dict[str, List[np.ndarray]]:
    """Generate synthetic SAR backscatter time series for demo mode.

    Simulates Sentinel-1 dual-polarisation backscatter (VV and VH) using
    normally distributed values centred on realistic crop canopy dB levels.
    A fixed RNG seed (123) ensures reproducibility.

    Args:
        n_timesteps: Number of time steps to generate (default 12).

    Returns:
        Dictionary with keys:
          - 'VV': list of n_timesteps (10, 10) float64 arrays in dB.
                  Centred around -12 dB (typical for crop canopies).
          - 'VH': list of n_timesteps (10, 10) float64 arrays in dB.
                  Centred around -18 dB (typically 6 dB below VV for crops).
    """
    rng = np.random.default_rng(123)
    vv_series: List[np.ndarray] = []
    vh_series: List[np.ndarray] = []

    for _ in range(n_timesteps):
        vv = rng.normal(-12.0, 2.0, (10, 10))
        vh = rng.normal(-18.0, 2.0, (10, 10))
        vv_series.append(vv)
        vh_series.append(vh)

    return {"VV": vv_series, "VH": vh_series}
