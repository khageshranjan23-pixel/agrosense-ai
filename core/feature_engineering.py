"""
AgroSense AI — Feature Engineering Module
Computes 600+ spectral, SAR, and temporal features per pixel.
All indices computed dynamically from satellite bands — no hardcoded values.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats as scipy_stats
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)

try:
    import ee
    GEE_OK = True
except ImportError:
    GEE_OK = False


# ── Optical Index Computation (GEE) ─────────────────────────────────────────

def compute_ndvi(image: Any) -> Any:
    """NDVI = (B8 - B4) / (B8 + B4). Vegetation density."""
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")

def compute_evi(image: Any) -> Any:
    """EVI = 2.5 * (B8-B4) / (B8 + 6*B4 - 7.5*B2 + 1). Reduced soil/atm effect."""
    evi = image.expression(
        "2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0)",
        {"B8": image.select("B8"), "B4": image.select("B4"), "B2": image.select("B2")},
    )
    return evi.rename("EVI").clamp(-1, 2)

def compute_ndwi(image: Any) -> Any:
    """NDWI = (B3 - B8) / (B3 + B8). Open water detection."""
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")

def compute_lswi(image: Any) -> Any:
    """LSWI = (B8 - B11) / (B8 + B11). Leaf water content."""
    return image.normalizedDifference(["B8", "B11"]).rename("LSWI")

def compute_ndre(image: Any) -> Any:
    """NDRE = (B8A - B5) / (B8A + B5). Red-edge, superior for dense canopy."""
    return image.normalizedDifference(["B8A", "B5"]).rename("NDRE")

def compute_savi(image: Any) -> Any:
    """SAVI = ((B8-B4)/(B8+B4+0.5)) * 1.5. Soil-adjusted."""
    savi = image.expression(
        "((B8 - B4) / (B8 + B4 + 0.5)) * 1.5",
        {"B8": image.select("B8"), "B4": image.select("B4")},
    )
    return savi.rename("SAVI")

def compute_mtci(image: Any) -> Any:
    """MTCI = (B8A - B5) / (B5 - B4). Chlorophyll content."""
    mtci = image.expression(
        "(B8A - B5) / (B5 - B4 + 1e-6)",
        {"B8A": image.select("B8A"), "B5": image.select("B5"), "B4": image.select("B4")},
    )
    return mtci.rename("MTCI").clamp(-5, 10)

def compute_cire(image: Any) -> Any:
    """CIre = (B8A / B5) - 1. Chlorophyll index red-edge."""
    cire = image.expression(
        "(B8A / (B5 + 1e-6)) - 1.0",
        {"B8A": image.select("B8A"), "B5": image.select("B5")},
    )
    return cire.rename("CIre").clamp(-1, 20)

def compute_lai(ndvi_image: Any) -> Any:
    """LAI = 0.57 * exp(2.33 * NDVI). Leaf area index from NDVI."""
    lai = ndvi_image.multiply(2.33).exp().multiply(0.57)
    return lai.rename("LAI").clamp(0, 10)

def compute_all_optical_indices(image: Any) -> Any:
    """Compute all optical indices for a single S2 image.
    
    Args:
        image: ee.Image with S2 SR bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12),
               reflectance scaled to 0-1.
    
    Returns:
        ee.Image with bands: NDVI, EVI, NDWI, LSWI, NDRE, SAVI, MTCI, CIre, LAI.
    """
    ndvi = compute_ndvi(image)
    evi = compute_evi(image)
    ndwi = compute_ndwi(image)
    lswi = compute_lswi(image)
    ndre = compute_ndre(image)
    savi = compute_savi(image)
    mtci = compute_mtci(image)
    cire = compute_cire(image)
    lai = compute_lai(ndvi)
    return ee.Image.cat([ndvi, evi, ndwi, lswi, ndre, savi, mtci, cire, lai])


# ── SAR Feature Computation (GEE) ────────────────────────────────────────────

def compute_sar_features(sar_image: Any) -> Any:
    """Compute SAR-derived features from a single Sentinel-1 image.
    
    Computes: VV, VH, CR (VH/VV), RVI_SAR, and GLCM texture.
    
    Args:
        sar_image: ee.Image with VV and VH bands (in dB).
    
    Returns:
        ee.Image with SAR feature bands.
    """
    vv = sar_image.select("VV")
    vh = sar_image.select("VH")
    
    # Cross-ratio CR = VH / VV (both in linear scale)
    vv_lin = ee.Image(10).pow(vv.divide(10))
    vh_lin = ee.Image(10).pow(vh.divide(10))
    
    cr = vh_lin.divide(vv_lin.add(1e-10)).log10().multiply(10).rename("CR")
    
    # SAR Vegetation Index RVI = 4*VH / (VV + VH)
    rvi_sar = vh_lin.multiply(4).divide(
        vv_lin.add(vh_lin).add(1e-10)
    ).rename("RVI_SAR")
    
    # GLCM texture on VH (7x7 kernel)
    glcm = vh.unitScale(-30, 0).multiply(255).toByte().glcmTexture(size=3)
    contrast = glcm.select(".*_contrast").rename("VH_contrast")
    homogeneity = glcm.select(".*_idm").rename("VH_homogeneity")
    energy = glcm.select(".*_asm").rename("VH_energy")
    correlation = glcm.select(".*_corr").rename("VH_correlation")
    
    return ee.Image.cat([
        vv.rename("VV"),
        vh.rename("VH"),
        cr,
        rvi_sar,
        contrast,
        homogeneity,
        energy,
        correlation,
    ])


# ── Temporal Feature Computation (NumPy) ────────────────────────────────────

def compute_temporal_statistics(
    time_series: np.ndarray,
    index_name: str,
) -> Dict[str, np.ndarray]:
    """Compute temporal statistics across T time steps for a single index.
    
    Args:
        time_series: Array of shape (T, H, W) — T time steps, H×W pixels.
        index_name: Name of the index (used as dict key prefix).
    
    Returns:
        Dict mapping feature names to (H, W) arrays:
          - {name}_mean, {name}_std, {name}_slope, {name}_max, {name}_auc,
            {name}_green_up_rate, {name}_senescence_rate.
    """
    T, H, W = time_series.shape
    t_axis = np.arange(T, dtype=np.float64)
    
    features: Dict[str, np.ndarray] = {}
    
    # Raw time steps
    for t in range(T):
        features[f"{index_name}_t{t+1:02d}"] = time_series[t]
    
    # Mean and std across time
    features[f"{index_name}_mean"] = np.nanmean(time_series, axis=0)
    features[f"{index_name}_std"] = np.nanstd(time_series, axis=0)
    
    # Temporal slope (linear regression per pixel)
    slopes = np.zeros((H, W), dtype=np.float64)
    for h in range(H):
        for w in range(W):
            pixel_ts = time_series[:, h, w]
            valid = ~np.isnan(pixel_ts)
            if valid.sum() >= 2:
                slope, *_ = scipy_stats.linregress(t_axis[valid], pixel_ts[valid])
                slopes[h, w] = slope
    features[f"{index_name}_slope"] = slopes
    
    # Max value and its time position
    features[f"{index_name}_max"] = np.nanmax(time_series, axis=0)
    features[f"{index_name}_argmax"] = np.nanargmax(time_series, axis=0).astype(np.float64)
    
    # Area under curve (trapezoidal integration)
    auc = np.zeros((H, W), dtype=np.float64)
    for h in range(H):
        for w in range(W):
            pixel_ts = time_series[:, h, w]
            valid_mask = ~np.isnan(pixel_ts)
            if valid_mask.sum() >= 2:
                auc[h, w] = trapezoid(pixel_ts[valid_mask], dx=1.0)
    features[f"{index_name}_auc"] = auc
    
    # Green-up rate: mean slope of ascending phase (before max)
    green_up = np.zeros((H, W), dtype=np.float64)
    senescence = np.zeros((H, W), dtype=np.float64)
    
    for h in range(H):
        for w in range(W):
            pixel_ts = time_series[:, h, w]
            peak_t = int(np.nanargmax(pixel_ts))
            
            if peak_t > 1:
                ascending = pixel_ts[:peak_t + 1]
                t_asc = t_axis[:peak_t + 1]
                valid = ~np.isnan(ascending)
                if valid.sum() >= 2:
                    sl, *_ = scipy_stats.linregress(t_asc[valid], ascending[valid])
                    green_up[h, w] = sl
            
            if peak_t < T - 2:
                descending = pixel_ts[peak_t:]
                t_desc = t_axis[peak_t:]
                valid = ~np.isnan(descending)
                if valid.sum() >= 2:
                    sl, *_ = scipy_stats.linregress(t_desc[valid], descending[valid])
                    senescence[h, w] = sl
    
    features[f"{index_name}_green_up_rate"] = green_up
    features[f"{index_name}_senescence_rate"] = senescence
    
    return features


def build_feature_matrix(
    optical_series: Dict[str, np.ndarray],
    sar_series: Dict[str, np.ndarray],
    n_timesteps: int,
) -> Tuple[np.ndarray, List[str]]:
    """Build full feature matrix from temporal index stacks.
    
    Args:
        optical_series: Dict {index_name: (T, H, W) array}.
        sar_series: Dict {sar_band: (T, H, W) array}.
        n_timesteps: Number of time steps T.
    
    Returns:
        Tuple of (X: (N_pixels, N_features), feature_names: List[str]).
        N_pixels = H * W.
    """
    all_features: Dict[str, np.ndarray] = {}
    
    # Optical temporal statistics
    for index_name, ts_array in optical_series.items():
        feats = compute_temporal_statistics(ts_array, index_name)
        all_features.update(feats)
    
    # SAR temporal statistics
    for sar_band, ts_array in sar_series.items():
        feats = compute_temporal_statistics(ts_array, f"SAR_{sar_band}")
        all_features.update(feats)
    
    feature_names = sorted(all_features.keys())
    
    # Get H, W from first feature
    sample = next(iter(all_features.values()))
    H, W = sample.shape
    N = H * W
    F = len(feature_names)
    
    X = np.zeros((N, F), dtype=np.float32)
    for fi, fname in enumerate(feature_names):
        arr = all_features[fname]
        X[:, fi] = arr.ravel()
    
    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return X, feature_names


def compute_sar_anomaly(
    sar_current: np.ndarray,
    sar_historical_mean: np.ndarray,
) -> np.ndarray:
    """Compute SAR backscatter anomaly vs 5-year historical mean.
    
    Args:
        sar_current: Current-period SAR image array (H, W), dB.
        sar_historical_mean: 5-year mean for same calendar period (H, W), dB.
    
    Returns:
        Anomaly array (H, W) in dB units.
    """
    return sar_current - sar_historical_mean
