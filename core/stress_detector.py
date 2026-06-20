"""
AgroSense AI — Moisture Stress Detection Engine
Multi-source stress assessment: VCI, TCI, VHI, CWSI, SAR-SMI.
All thresholds computed dynamically from data distributions — no hardcoded values.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VHI_ALPHA, HISTORICAL_YEARS,
    NO_STRESS_PERCENTILE, MILD_STRESS_PERCENTILE,
    load_stage_weights,
)


# ---------------------------------------------------------------------------
# Individual Stress Index Computation
# ---------------------------------------------------------------------------

def compute_vci(
    ndvi_current: np.ndarray,
    ndvi_hist_min: np.ndarray,
    ndvi_hist_max: np.ndarray,
) -> np.ndarray:
    """Vegetation Condition Index (VCI).
    
    VCI = 100 × (NDVI_current - NDVI_min_hist) / (NDVI_max_hist - NDVI_min_hist)
    
    Historical min/max computed from 5-year GEE baseline — NOT hardcoded.
    
    Args:
        ndvi_current: Current NDVI array (H, W).
        ndvi_hist_min: 5-year historical min NDVI (H, W).
        ndvi_hist_max: 5-year historical max NDVI (H, W).
    
    Returns:
        VCI array (H, W), range 0–100. Higher = better vegetation condition.
    """
    denom = ndvi_hist_max - ndvi_hist_min
    denom = np.where(denom < 1e-6, 1e-6, denom)
    vci = 100.0 * (ndvi_current - ndvi_hist_min) / denom
    return np.clip(vci, 0.0, 100.0)


def compute_tci(
    lst_current: np.ndarray,
    lst_hist_min: np.ndarray,
    lst_hist_max: np.ndarray,
) -> np.ndarray:
    """Temperature Condition Index (TCI).
    
    TCI = 100 × (LST_max_hist - LST_current) / (LST_max_hist - LST_min_hist)
    
    Args:
        lst_current: Current LST array (H, W), in Kelvin.
        lst_hist_min: Historical minimum LST (H, W).
        lst_hist_max: Historical maximum LST (H, W).
    
    Returns:
        TCI array (H, W), range 0–100. Higher = cooler = better condition.
    """
    denom = lst_hist_max - lst_hist_min
    denom = np.where(denom < 1e-6, 1e-6, denom)
    tci = 100.0 * (lst_hist_max - lst_current) / denom
    return np.clip(tci, 0.0, 100.0)


def compute_vhi(
    vci: np.ndarray,
    tci: np.ndarray,
    alpha: float = VHI_ALPHA,
) -> np.ndarray:
    """Vegetation Health Index (VHI).
    
    VHI = α × VCI + (1-α) × TCI
    
    Args:
        vci: VCI array (H, W).
        tci: TCI array (H, W).
        alpha: Weight for VCI vs TCI. User-adjustable via sidebar.
    
    Returns:
        VHI array (H, W), range 0–100.
    """
    return alpha * vci + (1.0 - alpha) * tci


def compute_cwsi(
    et_actual: np.ndarray,
    et_potential: np.ndarray,
) -> np.ndarray:
    """Crop Water Stress Index (CWSI).
    
    CWSI = 1 - (ETa / ET0)
    0 = no stress (ETa = ET0), 1 = complete stress (ETa = 0).
    
    Args:
        et_actual: Actual evapotranspiration from ERA5 (H, W), mm/day.
        et_potential: Reference ET₀ from Penman-Monteith (H, W), mm/day.
    
    Returns:
        CWSI array (H, W), range 0–1.
    """
    et0_safe = np.where(et_potential < 1e-6, 1e-6, et_potential)
    cwsi = 1.0 - (et_actual / et0_safe)
    return np.clip(cwsi, 0.0, 1.0)


def compute_sar_moisture_index(
    vh_anomaly: np.ndarray,
) -> np.ndarray:
    """SAR-based Soil Moisture Index from VH backscatter anomaly.
    
    SMI_SAR = normalize(VH_anomaly) — negative anomaly = drier than normal.
    
    Args:
        vh_anomaly: VH - VH_5year_mean (H, W), dB.
    
    Returns:
        SMI array (H, W), range 0–100. Higher = wetter than normal.
    """
    # Normalize anomaly: positive = wetter, negative = drier
    anomaly_min = np.nanpercentile(vh_anomaly, 2)
    anomaly_max = np.nanpercentile(vh_anomaly, 98)
    denom = anomaly_max - anomaly_min
    if denom < 1e-6:
        return np.full_like(vh_anomaly, 50.0)
    smi = 100.0 * (vh_anomaly - anomaly_min) / denom
    return np.clip(smi, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Combined Stress Score (PCA-based fusion — no arbitrary weighting)
# ---------------------------------------------------------------------------

def compute_combined_stress_score(
    vci: np.ndarray,
    tci: np.ndarray,
    vhi: np.ndarray,
    cwsi: np.ndarray,
    smi_sar: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fuse all stress indices into a single score via PCA.
    
    PC1 explains maximum variance across all stress signals.
    This is fully data-driven — no manual weighting.
    
    Args:
        vci, tci, vhi: Stress indices 0-100 (H, W each). High = good condition.
        cwsi: Crop water stress 0-1 (H, W). High = stressed.
        smi_sar: SAR moisture 0-100 (H, W). High = wetter.
    
    Returns:
        Tuple of:
          - combined_score: (H, W) array 0-100, higher = MORE stressed.
          - explained_variance: fraction explained by PC1.
    """
    H, W = vci.shape
    N = H * W
    
    # Invert VCI, TCI, VHI so high = more stress (consistent direction)
    stress_vci = 100.0 - vci.ravel()
    stress_tci = 100.0 - tci.ravel()
    stress_vhi = 100.0 - vhi.ravel()
    stress_cwsi = cwsi.ravel() * 100.0
    stress_smi = 100.0 - smi_sar.ravel()  # lower moisture = more stress
    
    X = np.column_stack([stress_vci, stress_tci, stress_vhi, stress_cwsi, stress_smi])
    
    # Replace NaN
    col_means = np.nanmean(X, axis=0)
    for ci in range(X.shape[1]):
        nan_mask = np.isnan(X[:, ci])
        X[nan_mask, ci] = col_means[ci]
    
    # Standardize
    scaler = MinMaxScaler(feature_range=(0, 100))
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).ravel()
    explained_var = float(pca.explained_variance_ratio_[0])
    
    # Rescale to 0-100
    pc1_min, pc1_max = pc1.min(), pc1.max()
    if pc1_max - pc1_min < 1e-6:
        combined = np.full(N, 50.0)
    else:
        combined = 100.0 * (pc1 - pc1_min) / (pc1_max - pc1_min)
    
    return combined.reshape(H, W), explained_var


# ---------------------------------------------------------------------------
# Phenology-Aware Stress Severity
# ---------------------------------------------------------------------------

def compute_phenology_weighted_stress(
    combined_score: np.ndarray,
    stage_map: np.ndarray,
) -> np.ndarray:
    """Weight stress severity by growth stage sensitivity.
    
    Stage weights loaded from stage_weights.json (FAO-56 Table 8).
    Weights: establishment=0.4, vegetative=0.7, reproductive=1.0,
             grain_fill=0.8, maturity=0.3.
    
    Args:
        combined_score: Combined stress score (H, W), 0-100.
        stage_map: Growth stage code map (H, W), int8.
    
    Returns:
        Weighted stress severity (H, W), 0-100.
    """
    stage_weights = load_stage_weights()
    
    # Map stage codes to weights
    code_to_weight = {
        0: stage_weights.get("pre_sowing", 0.0),
        1: stage_weights.get("establishment", 0.4),
        2: stage_weights.get("vegetative", 0.7),
        3: stage_weights.get("reproductive", 1.0),
        4: stage_weights.get("grain_fill", 0.8),
        5: stage_weights.get("maturity", 0.3),
    }
    
    weight_map = np.vectorize(code_to_weight.get)(stage_map)
    return np.clip(combined_score * weight_map, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Dynamic Stress Classification (quantile-based — no fixed thresholds)
# ---------------------------------------------------------------------------

def classify_stress_dynamic(
    weighted_stress: np.ndarray,
    vhi: np.ndarray,
) -> np.ndarray:
    """Classify stress into categories using area-specific quantiles.
    
    Uses quantile thresholds computed from the actual data distribution.
    NOT fixed VCI thresholds.
    
    Args:
        weighted_stress: Phenology-weighted stress score (H, W), 0-100.
        vhi: VHI array for threshold computation (H, W).
    
    Returns:
        Stress class array (H, W):
          0 = no_stress (top 40th percentile of VHI)
          1 = mild_stress (20th–40th)
          2 = moderate_stress (10th–20th)
          3 = severe_stress (below 10th percentile)
    """
    valid_vhi = vhi[~np.isnan(vhi)]
    
    # Dynamic thresholds from actual VHI distribution in the area
    thresh_no_mild = float(np.percentile(valid_vhi, 100 - NO_STRESS_PERCENTILE))
    thresh_mild_mod = float(np.percentile(valid_vhi, 100 - MILD_STRESS_PERCENTILE))
    thresh_mod_severe = float(np.percentile(valid_vhi, 10))
    
    stress_class = np.full_like(vhi, 1, dtype=np.int8)  # default: mild
    stress_class[vhi >= thresh_no_mild] = 0    # no stress
    stress_class[(vhi < thresh_mild_mod) & (vhi >= thresh_mod_severe)] = 2  # moderate
    stress_class[vhi < thresh_mod_severe] = 3  # severe
    
    return stress_class


def get_stress_class_name(code: int) -> str:
    """Convert stress class code to human-readable name."""
    mapping = {0: "No Stress", 1: "Mild Stress", 2: "Moderate Stress", 3: "Severe Stress"}
    return mapping.get(int(code), "Unknown")


# ---------------------------------------------------------------------------
# Full Stress Pipeline
# ---------------------------------------------------------------------------

def run_stress_analysis(
    ndvi_current: np.ndarray,
    ndvi_hist_min: np.ndarray,
    ndvi_hist_max: np.ndarray,
    lst_current: np.ndarray,
    lst_hist_min: np.ndarray,
    lst_hist_max: np.ndarray,
    et_actual: np.ndarray,
    et_potential: np.ndarray,
    vh_anomaly: np.ndarray,
    stage_map: np.ndarray,
    vhi_alpha: float = VHI_ALPHA,
) -> Dict[str, Any]:
    """Run complete multi-source stress detection pipeline.
    
    Args:
        ndvi_current: Current-period NDVI (H, W).
        ndvi_hist_min/max: 5-year historical NDVI bounds (H, W).
        lst_current: Current LST in Kelvin (H, W).
        lst_hist_min/max: Historical LST bounds (H, W).
        et_actual: ERA5 actual evapotranspiration mm/day (H, W).
        et_potential: Penman-Monteith ET₀ mm/day (H, W).
        vh_anomaly: SAR VH anomaly dB (H, W).
        stage_map: Growth stage codes (H, W).
        vhi_alpha: VCI weight in VHI (user-adjustable).
    
    Returns:
        Dict with all stress layers and summary statistics.
    """
    # Individual indices
    vci = compute_vci(ndvi_current, ndvi_hist_min, ndvi_hist_max)
    tci = compute_tci(lst_current, lst_hist_min, lst_hist_max)
    vhi = compute_vhi(vci, tci, alpha=vhi_alpha)
    cwsi = compute_cwsi(et_actual, et_potential)
    smi_sar = compute_sar_moisture_index(vh_anomaly)
    
    # PCA-fused combined score
    combined_score, explained_var = compute_combined_stress_score(
        vci, tci, vhi, cwsi, smi_sar
    )
    
    # Phenology-weighted severity
    weighted_stress = compute_phenology_weighted_stress(combined_score, stage_map)
    
    # Dynamic classification
    stress_class = classify_stress_dynamic(weighted_stress, vhi)
    
    # Summary statistics
    valid = ~np.isnan(weighted_stress)
    summary = {
        "mean_stress": float(np.nanmean(weighted_stress)),
        "pct_no_stress": float(np.mean(stress_class[valid] == 0) * 100),
        "pct_mild_stress": float(np.mean(stress_class[valid] == 1) * 100),
        "pct_moderate_stress": float(np.mean(stress_class[valid] == 2) * 100),
        "pct_severe_stress": float(np.mean(stress_class[valid] == 3) * 100),
        "mean_vci": float(np.nanmean(vci)),
        "mean_vhi": float(np.nanmean(vhi)),
        "mean_cwsi": float(np.nanmean(cwsi)),
        "pca_explained_variance": explained_var,
    }
    
    return {
        "vci": vci,
        "tci": tci,
        "vhi": vhi,
        "cwsi": cwsi,
        "smi_sar": smi_sar,
        "combined_score": combined_score,
        "weighted_stress": weighted_stress,
        "stress_class": stress_class,
        "summary": summary,
    }
