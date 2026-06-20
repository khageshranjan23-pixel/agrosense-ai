"""
AgroSense AI — Phenology Engine
Dynamic growth stage detection using NDVI time series analysis.
No hardcoded NDVI thresholds — all thresholds derived from pixel-specific data.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import label as ndimage_label

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PHENO_SOS_THRESHOLD, PHENO_EOS_THRESHOLD, PHENO_PEAK_DROP,
    SAVITZKY_WINDOW, SAVITZKY_POLY, load_stage_weights,
)


STAGE_CODES = {
    "pre_sowing": 0,
    "establishment": 1,
    "vegetative": 2,
    "reproductive": 3,
    "grain_fill": 4,
    "maturity": 5,
}

STAGE_NAMES = {v: k for k, v in STAGE_CODES.items()}


def smooth_ndvi_series(
    ndvi_series: np.ndarray,
    window: int = SAVITZKY_WINDOW,
    poly: int = SAVITZKY_POLY,
) -> np.ndarray:
    """Apply Savitzky-Golay filter to NDVI time series.
    
    Args:
        ndvi_series: 1D or 2D array. If 2D: (T, N_pixels).
        window: Window length for S-G filter (must be odd).
        poly: Polynomial order for S-G filter.
    
    Returns:
        Smoothed array same shape as input.
    """
    if ndvi_series.ndim == 1:
        T = len(ndvi_series)
        w = min(window, T if T % 2 == 1 else T - 1)
        if w < poly + 1:
            return ndvi_series.copy()
        return savgol_filter(ndvi_series, window_length=w, polyorder=poly)
    
    T, N = ndvi_series.shape
    smoothed = np.zeros_like(ndvi_series)
    w = min(window, T if T % 2 == 1 else T - 1)
    if w < poly + 1:
        return ndvi_series.copy()
    for i in range(N):
        pixel_ts = ndvi_series[:, i]
        valid = ~np.isnan(pixel_ts)
        if valid.sum() >= w:
            pixel_ts_filled = np.where(valid, pixel_ts, np.nanmean(pixel_ts))
            smoothed[:, i] = savgol_filter(pixel_ts_filled, window_length=w, polyorder=poly)
        else:
            smoothed[:, i] = pixel_ts
    return smoothed


def detect_phenological_dates(
    ndvi_smooth: np.ndarray,
) -> Dict[str, int]:
    """Detect SOS, PEAK, EOS_ONSET, EOS for a single pixel NDVI time series.
    
    All thresholds are dynamically computed from the pixel's own NDVI range.
    No hardcoded NDVI values.
    
    Args:
        ndvi_smooth: 1D smoothed NDVI time series of length T.
    
    Returns:
        Dict with 't_sos', 't_peak', 't_eos_onset', 't_eos' as time indices.
        Missing values set to -1.
    """
    T = len(ndvi_smooth)
    
    ndvi_min = np.nanmin(ndvi_smooth)
    ndvi_max = np.nanmax(ndvi_smooth)
    ndvi_range = ndvi_max - ndvi_min
    
    result = {"t_sos": -1, "t_peak": -1, "t_eos_onset": -1, "t_eos": -1}
    
    if ndvi_range < 0.05:
        # Not enough NDVI variation — not a crop pixel
        return result
    
    # Normalize
    ndvi_n = (ndvi_smooth - ndvi_min) / ndvi_range
    
    # Peak: global maximum
    t_peak = int(np.nanargmax(ndvi_smooth))
    result["t_peak"] = t_peak
    
    # SOS: first crossing of SOS_THRESHOLD on ascending limb before peak
    sos_thresh = PHENO_SOS_THRESHOLD
    for t in range(t_peak):
        if ndvi_n[t] >= sos_thresh:
            result["t_sos"] = t
            break
    
    # EOS_ONSET: first date after peak where NDVIn < (1 - PEAK_DROP)
    eos_onset_thresh = 1.0 - PHENO_PEAK_DROP
    for t in range(t_peak + 1, T):
        if ndvi_n[t] < eos_onset_thresh:
            result["t_eos_onset"] = t
            break
    
    # EOS: first date where NDVIn < EOS_THRESHOLD on descending limb
    eos_thresh = PHENO_EOS_THRESHOLD
    start_search = result["t_eos_onset"] if result["t_eos_onset"] >= 0 else t_peak + 1
    for t in range(start_search, T):
        if ndvi_n[t] < eos_thresh:
            result["t_eos"] = t
            break
    
    # Fallback: EOS = last time step if not detected
    if result["t_eos"] < 0 and result["t_peak"] >= 0:
        result["t_eos"] = T - 1
    
    return result


def assign_growth_stages(
    ndvi_series: np.ndarray,
) -> np.ndarray:
    """Assign a growth stage code (0-5) per pixel per time step.
    
    Args:
        ndvi_series: Array of shape (T, H, W) — smoothed NDVI.
    
    Returns:
        Stage code array of shape (T, H, W), dtype int8.
        Codes: 0=pre_sowing, 1=establishment, 2=vegetative,
               3=reproductive, 4=grain_fill, 5=maturity.
    """
    T, H, W = ndvi_series.shape
    stages = np.zeros((T, H, W), dtype=np.int8)  # default: pre_sowing
    
    flat = ndvi_series.reshape(T, -1)  # (T, N)
    N = flat.shape[1]
    
    for n in range(N):
        pixel_ts = flat[:, n]
        pheno = detect_phenological_dates(pixel_ts)
        
        t_sos = pheno["t_sos"]
        t_peak = pheno["t_peak"]
        t_eos_onset = pheno["t_eos_onset"]
        t_eos = pheno["t_eos"]
        
        if t_sos < 0:
            continue  # no crop detected
        
        season_len = max(t_eos - t_sos, 1) if t_eos > 0 else T - t_sos
        veg_end = t_sos + int(0.50 * season_len)
        
        pixel_stages = np.zeros(T, dtype=np.int8)
        for t in range(T):
            if t < t_sos:
                pixel_stages[t] = STAGE_CODES["pre_sowing"]
            elif t < t_sos + max(1, int(0.15 * season_len)):
                pixel_stages[t] = STAGE_CODES["establishment"]
            elif t < veg_end:
                pixel_stages[t] = STAGE_CODES["vegetative"]
            elif t <= t_peak:
                pixel_stages[t] = STAGE_CODES["reproductive"]
            elif t_eos_onset >= 0 and t < t_eos_onset:
                pixel_stages[t] = STAGE_CODES["grain_fill"]
            elif t_eos >= 0 and t <= t_eos:
                pixel_stages[t] = STAGE_CODES["maturity"]
            else:
                pixel_stages[t] = STAGE_CODES["pre_sowing"]
        
        h_idx = n // W
        w_idx = n % W
        stages[:, h_idx, w_idx] = pixel_stages
    
    return stages


def get_current_stage(
    stages: np.ndarray,
    time_index: int,
) -> np.ndarray:
    """Extract growth stage map for a specific time index.
    
    Args:
        stages: Stage array (T, H, W).
        time_index: Time step index.
    
    Returns:
        2D stage code array (H, W).
    """
    return stages[time_index]


def compute_planting_dates(
    ndvi_series: np.ndarray,
    date_list: List[str],
) -> np.ndarray:
    """Compute pixel-specific planting (SOS) dates.
    
    Args:
        ndvi_series: Smoothed NDVI (T, H, W).
        date_list: List of T date strings ('YYYY-MM-DD').
    
    Returns:
        2D string array (H, W) of planting date strings.
        Pixels with no detected crop get 'unknown'.
    """
    T, H, W = ndvi_series.shape
    planting_dates = np.full((H, W), "unknown", dtype=object)
    
    flat = ndvi_series.reshape(T, -1)
    N = flat.shape[1]
    
    for n in range(N):
        pheno = detect_phenological_dates(flat[:, n])
        t_sos = pheno["t_sos"]
        h_idx = n // W
        w_idx = n % W
        if 0 <= t_sos < len(date_list):
            planting_dates[h_idx, w_idx] = date_list[t_sos]
    
    return planting_dates


def compute_days_since_planting(
    planting_dates: np.ndarray,
    current_date: str,
) -> np.ndarray:
    """Compute days since planting per pixel.
    
    Args:
        planting_dates: (H, W) array of date strings or 'unknown'.
        current_date: Target date 'YYYY-MM-DD'.
    
    Returns:
        (H, W) float array of days since planting. -1 where unknown.
    """
    from datetime import datetime
    H, W = planting_dates.shape
    dsp = np.full((H, W), -1.0)
    current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    
    for h in range(H):
        for w in range(W):
            d = planting_dates[h, w]
            if d != "unknown":
                try:
                    plant_dt = datetime.strptime(str(d), "%Y-%m-%d")
                    dsp[h, w] = (current_dt - plant_dt).days
                except ValueError:
                    pass
    return dsp


def get_stage_area_distribution(stage_map: np.ndarray) -> Dict[str, float]:
    """Compute percentage area in each growth stage.
    
    Args:
        stage_map: 2D array (H, W) with stage codes 0-5.
    
    Returns:
        Dict mapping stage name to percentage of total pixels.
    """
    total = stage_map.size
    result = {}
    for code, name in STAGE_NAMES.items():
        pct = float(np.sum(stage_map == code)) / total * 100
        result[name] = round(pct, 2)
    return result
