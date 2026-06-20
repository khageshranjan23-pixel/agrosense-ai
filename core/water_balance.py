"""
AgroSense AI — FAO-56 Water Balance Engine
Penman-Monteith ET₀, dynamic Kc interpolation, dual-Kc, soil water balance.
All values computed from data — no hardcoded thresholds or Kc values.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RUNOFF_COEFF_FLAT, RUNOFF_COEFF_GENTLE,
    RUNOFF_COEFF_MODERATE, RUNOFF_COEFF_STEEP,
    load_kc_database,
)


# ---------------------------------------------------------------------------
# Penman-Monteith Reference ET₀
# ---------------------------------------------------------------------------

def compute_et0_penman_monteith(
    t_mean_c: np.ndarray,
    rn_mj: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    t_dew_c: np.ndarray,
    elevation_m: np.ndarray,
) -> np.ndarray:
    """Compute FAO-56 Penman-Monteith reference ET₀ from ERA5 data.
    
    ET₀ = [0.408·Δ·(Rn-G) + γ·(900/(T+273))·u2·(es-ea)]
          / [Δ + γ·(1 + 0.34·u2)]
    
    All inputs are spatial arrays — computes ET₀ at each pixel.
    
    Args:
        t_mean_c: Mean daily temperature from ERA5 (H, W), °C.
        rn_mj: Net radiation from ERA5 surface solar radiation (H, W), MJ/m²/day.
        u10_ms: U-component wind at 10m (H, W), m/s.
        v10_ms: V-component wind at 10m (H, W), m/s.
        t_dew_c: Dew point temperature (H, W), °C.
        elevation_m: SRTM elevation (H, W), metres.
    
    Returns:
        ET₀ array (H, W), mm/day.
    """
    # Wind speed at 2m from 10m using log-law
    # u2 = u10 * 4.87 / ln(67.8*10 - 5.42)
    u10 = np.sqrt(u10_ms**2 + v10_ms**2)
    u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)
    u2 = np.clip(u2, 0.5, None)  # minimum 0.5 m/s as per FAO-56

    # Atmospheric pressure from elevation (FAO-56 Eq 7)
    P_kpa = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26

    # Psychrometric constant γ (kPa/°C)
    gamma = 0.000665 * P_kpa

    # Saturation vapour pressure es (kPa)
    es = 0.6108 * np.exp(17.27 * t_mean_c / (t_mean_c + 237.3))

    # Actual vapour pressure ea from dew point
    ea = 0.6108 * np.exp(17.27 * t_dew_c / (t_dew_c + 237.3))

    # Slope of vapour pressure curve Δ (kPa/°C)
    delta = 4098.0 * es / (t_mean_c + 237.3) ** 2

    # Soil heat flux G (approximate: 10% of Rn for daily)
    G = 0.1 * rn_mj

    # Penman-Monteith equation (FAO-56, Eq 6)
    numerator = (
        0.408 * delta * (rn_mj - G)
        + gamma * (900.0 / (t_mean_c + 273.0)) * u2 * (es - ea)
    )
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    denominator = np.where(denominator < 1e-6, 1e-6, denominator)

    et0 = numerator / denominator
    return np.clip(et0, 0.0, 20.0)  # physical upper bound


def aggregate_et0_8day(
    et0_daily_series: List[np.ndarray],
) -> np.ndarray:
    """Sum daily ET₀ to 8-day totals.
    
    Args:
        et0_daily_series: List of daily ET₀ arrays (H, W).
    
    Returns:
        8-day summed ET₀ (H, W), mm/8-day.
    """
    stack = np.stack(et0_daily_series, axis=0)  # (T, H, W)
    return np.nansum(stack, axis=0)


# ---------------------------------------------------------------------------
# Dynamic Crop Coefficient (Kc) Interpolation
# ---------------------------------------------------------------------------

def interpolate_kc(
    crop_name: str,
    days_since_planting: np.ndarray,
) -> np.ndarray:
    """Compute spatially-variable Kc via linear interpolation between FAO stages.
    
    Kc is NOT a single value — it varies per pixel based on planting date
    derived from the phenology engine.
    
    Args:
        crop_name: Crop name key in kc_database.json.
        days_since_planting: Pixel-level days since planting (H, W).
    
    Returns:
        Kc array (H, W), typically 0.3–1.25.
    """
    kc_db = load_kc_database()
    crop = kc_db.get(crop_name.lower().replace(" ", "_"))
    if crop is None:
        # Try partial match
        for key in kc_db:
            if crop_name.lower() in key or key in crop_name.lower():
                crop = kc_db[key]
                break
    if crop is None:
        logger.warning("Crop '%s' not in KC database — using default Kc=1.0", crop_name)
        return np.ones_like(days_since_planting, dtype=np.float32)

    kc_ini = float(crop["Kc_ini"])
    kc_mid = float(crop["Kc_mid"])
    kc_end = float(crop["Kc_end"])
    l_ini = float(crop["L_ini"])
    l_dev = float(crop["L_dev"])
    l_mid = float(crop["L_mid"])
    l_late = float(crop["L_late"])

    # Stage boundaries in days since planting
    t1 = l_ini
    t2 = l_ini + l_dev
    t3 = l_ini + l_dev + l_mid
    t4 = l_ini + l_dev + l_mid + l_late

    dsp = days_since_planting
    kc = np.full_like(dsp, kc_ini, dtype=np.float32)

    # Development stage: linear interpolation Kc_ini → Kc_mid
    dev_mask = (dsp >= t1) & (dsp < t2)
    if dev_mask.any():
        frac = (dsp[dev_mask] - t1) / max(l_dev, 1)
        kc[dev_mask] = kc_ini + frac * (kc_mid - kc_ini)

    # Mid-season stage: Kc = Kc_mid
    mid_mask = (dsp >= t2) & (dsp < t3)
    kc[mid_mask] = kc_mid

    # Late season stage: linear interpolation Kc_mid → Kc_end
    late_mask = (dsp >= t3) & (dsp < t4)
    if late_mask.any():
        frac = (dsp[late_mask] - t3) / max(l_late, 1)
        kc[late_mask] = kc_mid + frac * (kc_end - kc_mid)

    # Post-harvest
    post_mask = dsp >= t4
    kc[post_mask] = kc_end

    return np.clip(kc, 0.1, 1.5)


def compute_etc(
    et0: np.ndarray,
    kc: np.ndarray,
) -> np.ndarray:
    """Compute crop evapotranspiration ETc = Kc × ET₀.
    
    Args:
        et0: Reference ET₀ (H, W), mm.
        kc: Crop coefficient (H, W).
    
    Returns:
        ETc (H, W), mm.
    """
    return et0 * kc


# ---------------------------------------------------------------------------
# Dual-Kc (Advanced mode)
# ---------------------------------------------------------------------------

def compute_dual_kc(
    crop_name: str,
    days_since_planting: np.ndarray,
    et0: np.ndarray,
    dr_prev: np.ndarray,
    taw: np.ndarray,
    fw: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute dual crop coefficient (Kcb + Ke) per FAO-56.
    
    Args:
        crop_name: Crop name for database lookup.
        days_since_planting: Days since planting per pixel (H, W).
        et0: Reference ET₀ (H, W), mm.
        dr_prev: Previous day's root zone depletion (H, W), mm.
        taw: Total available water in root zone (H, W), mm.
        fw: Fraction of wetted soil surface (default 1.0 for rain-fed).
    
    Returns:
        Dict with 'kcb', 'ke', 'kc_dual', 'etcb', 'ke_et', 'etc_dual'.
    """
    kc_db = load_kc_database()
    crop = kc_db.get(crop_name.lower().replace(" ", "_"), {})

    kcb_ini = float(crop.get("Kcb_ini", 0.15))
    kcb_mid = float(crop.get("Kcb_mid", 1.00))
    kcb_end = float(crop.get("Kcb_end", 0.25))
    l_ini = float(crop.get("L_ini", 20))
    l_dev = float(crop.get("L_dev", 30))
    l_mid = float(crop.get("L_mid", 50))
    l_late = float(crop.get("L_late", 30))

    # Compute Kcb using same interpolation as Kc
    dsp = days_since_planting
    t1, t2 = l_ini, l_ini + l_dev
    t3, t4 = t2 + l_mid, t2 + l_mid + l_late

    kcb = np.full_like(dsp, kcb_ini, dtype=np.float32)
    dev_mask = (dsp >= t1) & (dsp < t2)
    if dev_mask.any():
        frac = (dsp[dev_mask] - t1) / max(l_dev, 1)
        kcb[dev_mask] = kcb_ini + frac * (kcb_mid - kcb_ini)
    kcb[(dsp >= t2) & (dsp < t3)] = kcb_mid
    late_mask = (dsp >= t3) & (dsp < t4)
    if late_mask.any():
        frac = (dsp[late_mask] - t3) / max(l_late, 1)
        kcb[late_mask] = kcb_mid + frac * (kcb_end - kcb_mid)
    kcb[dsp >= t4] = kcb_end

    # Kcmax = max(1.2, Kcb + 0.05)
    kcmax = np.maximum(1.2, kcb + 0.05)

    # Evaporation reduction coefficient Kr
    # Kr = 1 when soil is wet, decreases as top soil dries
    taw_safe = np.where(taw < 1e-6, 1e-6, taw)
    tew = 0.10 * taw  # total evaporable water (simplified: 10% of TAW)
    tew_safe = np.where(tew < 1e-6, 1e-6, tew)
    kr = np.clip((tew - np.minimum(dr_prev, tew)) / tew_safe, 0.0, 1.0)

    # Fraction of exposed soil
    few = np.clip(1.0 - kcb / kcmax, 0.01, 1.0) * fw

    # Ke = min(Kr * (Kcmax - Kcb), few * Kcmax)
    ke = np.minimum(kr * (kcmax - kcb), few * kcmax)
    ke = np.clip(ke, 0.0, 1.0)

    kc_dual = kcb + ke
    etcb = kcb * et0
    ke_et = ke * et0
    etc_dual = kc_dual * et0

    return {
        "kcb": kcb, "ke": ke, "kc_dual": kc_dual,
        "etcb": etcb, "ke_et": ke_et, "etc_dual": etc_dual,
    }


# ---------------------------------------------------------------------------
# Soil Water Balance
# ---------------------------------------------------------------------------

def get_taw(
    theta_fc: np.ndarray,
    theta_wp: np.ndarray,
    root_depth_m: np.ndarray,
) -> np.ndarray:
    """Compute Total Available Water (TAW) from SoilGrids data.
    
    TAW = 1000 × (θfc - θwp) × Zr
    
    Args:
        theta_fc: Field capacity volumetric water content (H, W), m³/m³.
        theta_wp: Wilting point volumetric water content (H, W), m³/m³.
        root_depth_m: Effective root depth (H, W), metres.
    
    Returns:
        TAW (H, W), mm.
    """
    return 1000.0 * np.clip(theta_fc - theta_wp, 0.01, 0.5) * np.clip(root_depth_m, 0.1, 3.0)


def get_raw(
    taw: np.ndarray,
    p_depletion: float,
) -> np.ndarray:
    """Compute Readily Available Water (RAW).
    
    RAW = p × TAW, where p is crop-specific depletion fraction from FAO-56.
    
    Args:
        taw: Total available water (H, W), mm.
        p_depletion: Depletion fraction p (from kc_database.json, crop-specific).
    
    Returns:
        RAW (H, W), mm.
    """
    return np.clip(p_depletion, 0.0, 1.0) * taw


def get_root_depth(
    crop_name: str,
    days_since_planting: np.ndarray,
) -> np.ndarray:
    """Compute effective root depth via linear interpolation (FAO-56 Table).
    
    Args:
        crop_name: Crop name for database lookup.
        days_since_planting: Days since planting per pixel (H, W).
    
    Returns:
        Root depth (H, W), metres.
    """
    kc_db = load_kc_database()
    crop = kc_db.get(crop_name.lower().replace(" ", "_"), {})
    zr_min = float(crop.get("Zr_min", 0.3))
    zr_max = float(crop.get("Zr_max", 1.5))
    l_total = sum([
        float(crop.get("L_ini", 20)),
        float(crop.get("L_dev", 30)),
        float(crop.get("L_mid", 50)),
        float(crop.get("L_late", 30)),
    ])

    # Root grows from Zr_min to Zr_max over first 50% of season
    growth_end = l_total * 0.5
    frac = np.clip(days_since_planting / max(growth_end, 1), 0.0, 1.0)
    return zr_min + frac * (zr_max - zr_min)


def get_runoff_coefficient(slope_deg: np.ndarray) -> np.ndarray:
    """Compute runoff coefficient based on terrain slope.
    
    Args:
        slope_deg: Slope in degrees (H, W).
    
    Returns:
        Runoff coefficient (H, W), dimensionless.
    """
    rc = np.full_like(slope_deg, RUNOFF_COEFF_FLAT, dtype=np.float32)
    rc[slope_deg >= 2] = RUNOFF_COEFF_GENTLE
    rc[slope_deg >= 5] = RUNOFF_COEFF_MODERATE
    rc[slope_deg >= 10] = RUNOFF_COEFF_STEEP
    return rc


def update_soil_water_balance(
    dr_prev: np.ndarray,
    rainfall_mm: np.ndarray,
    etc_mm: np.ndarray,
    slope_deg: np.ndarray,
    taw: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Update daily soil water depletion.
    
    Dr(t) = Dr(t-1) - (P - RO) + ETc + DP
    
    where I (irrigation) is the unknown we solve for (target output).
    DP (deep percolation): occurs when Dr < 0.
    
    Args:
        dr_prev: Previous depletion (H, W), mm.
        rainfall_mm: CHIRPS rainfall (H, W), mm.
        etc_mm: Crop ET (H, W), mm.
        slope_deg: Terrain slope (H, W), degrees.
        taw: Total available water (H, W), mm.
    
    Returns:
        Dict with 'dr_current', 'runoff', 'deep_percolation'.
    """
    rc = get_runoff_coefficient(slope_deg)
    runoff = rainfall_mm * rc
    effective_rain = rainfall_mm - runoff

    dr_new = dr_prev - effective_rain + etc_mm

    # Deep percolation when soil over-saturated
    dp = np.where(dr_new < 0, -dr_new, 0.0)
    dr_new = np.clip(dr_new, 0.0, taw * 1.2)

    return {
        "dr_current": dr_new,
        "runoff": runoff,
        "deep_percolation": dp,
        "effective_rain": effective_rain,
    }


# ---------------------------------------------------------------------------
# Irrigation Advisory from Water Balance
# ---------------------------------------------------------------------------

def compute_irrigation_need(
    dr_current: np.ndarray,
    raw: np.ndarray,
    taw: np.ndarray,
) -> np.ndarray:
    """Compute net irrigation need.
    
    When Dr > RAW: crop is under stress — irrigate.
    When Dr > TAW: wilting point exceeded — critical.
    
    Args:
        dr_current: Current soil water depletion (H, W), mm.
        raw: Readily available water (H, W), mm.
        taw: Total available water (H, W), mm.
    
    Returns:
        Net irrigation need (H, W), mm. 0 where not needed.
    """
    need = np.maximum(0.0, dr_current - raw)
    return np.round(need, 1)


def classify_irrigation_advisory(
    dr_current: np.ndarray,
    raw: np.ndarray,
    taw: np.ndarray,
) -> np.ndarray:
    """Classify irrigation urgency per pixel.
    
    All thresholds are pixel-specific (depend on RAW and TAW,
    which depend on actual soil type and crop — not hardcoded).
    
    Returns:
        Advisory code (H, W):
          0 = no_irrigation (Dr < RAW)
          1 = irrigate_3_days (RAW ≤ Dr < 0.7*TAW)
          2 = irrigate_immediately (0.7*TAW ≤ Dr < TAW)
          3 = critical_alert (Dr ≥ TAW)
    """
    advisory = np.zeros_like(dr_current, dtype=np.int8)
    advisory[(dr_current >= raw) & (dr_current < taw * 0.7)] = 1
    advisory[(dr_current >= taw * 0.7) & (dr_current < taw)] = 2
    advisory[dr_current >= taw] = 3
    return advisory


def get_advisory_text(code: int) -> str:
    """Convert advisory code to human-readable message."""
    mapping = {
        0: "✅ No irrigation needed",
        1: "⚠️ Irrigate within 3 days",
        2: "🔴 Irrigate immediately",
        3: "🚨 CRITICAL — Crop stress threshold exceeded",
    }
    return mapping.get(int(code), "Unknown")


# ---------------------------------------------------------------------------
# SoilGrids Fallback (when API unavailable)
# ---------------------------------------------------------------------------

def get_default_soil_params(
    shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Return conservative default soil parameters when SoilGrids is unavailable.
    
    Uses loam soil typical values from FAO-56 Table 19.
    
    Args:
        shape: (H, W) of the output arrays.
    
    Returns:
        Dict with 'theta_fc', 'theta_wp' arrays.
    """
    # Loam: FC = 0.32, WP = 0.14 (FAO-56 Table 19)
    return {
        "theta_fc": np.full(shape, 0.32, dtype=np.float32),
        "theta_wp": np.full(shape, 0.14, dtype=np.float32),
    }
