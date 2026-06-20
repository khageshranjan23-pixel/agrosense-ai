"""
AgroSense AI — Irrigation Advisory Engine
Integrates water balance, stress detection, phenology, and crop classification
to generate spatially-explicit, 8-day period irrigation advisory maps.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_kc_database

from core.water_balance import (
    compute_et0_penman_monteith,
    compute_etc,
    interpolate_kc,
    get_taw,
    get_raw,
    get_root_depth,
    update_soil_water_balance,
    compute_irrigation_need,
    classify_irrigation_advisory,
    get_advisory_text,
    get_default_soil_params,
)
from core.stress_detector import run_stress_analysis
from core.phenology import STAGE_NAMES


def run_advisory_pipeline(
    # ET₀ inputs (from ERA5)
    t_mean_c: np.ndarray,
    rn_mj: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    t_dew_c: np.ndarray,
    elevation_m: np.ndarray,
    # Rainfall
    rainfall_mm: np.ndarray,
    # Crop info
    crop_map: np.ndarray,
    crop_classes: List[str],
    days_since_planting: np.ndarray,
    stage_map: np.ndarray,
    # Stress inputs
    ndvi_current: np.ndarray,
    ndvi_hist_min: np.ndarray,
    ndvi_hist_max: np.ndarray,
    lst_current: np.ndarray,
    lst_hist_min: np.ndarray,
    lst_hist_max: np.ndarray,
    et_actual: np.ndarray,
    vh_anomaly: np.ndarray,
    slope_deg: np.ndarray,
    # Soil (from SoilGrids or fallback)
    theta_fc: Optional[np.ndarray] = None,
    theta_wp: Optional[np.ndarray] = None,
    # Options
    use_dual_kc: bool = False,
    vhi_alpha: float = 0.5,
    period_days: int = 8,
) -> Dict[str, Any]:
    """Run the complete 8-day irrigation advisory pipeline.
    
    Combines ET₀ computation, dynamic Kc, soil water balance, and
    multi-source stress detection into spatially-explicit advisory maps.
    
    Args:
        t_mean_c: Mean temperature (H, W), °C.
        rn_mj: Net radiation (H, W), MJ/m²/day.
        u10_ms, v10_ms: Wind components at 10m (H, W), m/s.
        t_dew_c: Dew point temperature (H, W), °C.
        elevation_m: SRTM elevation (H, W), m.
        rainfall_mm: CHIRPS rainfall for period (H, W), mm.
        crop_map: Classified crop type codes (H, W), int.
        crop_classes: List of crop class names (index = code).
        days_since_planting: Per-pixel days since SOS (H, W).
        stage_map: Growth stage codes (H, W).
        ndvi_current: Current NDVI (H, W).
        ndvi_hist_min/max: Historical NDVI bounds (H, W).
        lst_current: Current LST in K (H, W).
        lst_hist_min/max: Historical LST bounds (H, W).
        et_actual: ERA5 actual ET (H, W), mm/day.
        vh_anomaly: SAR VH anomaly (H, W), dB.
        slope_deg: Terrain slope (H, W), degrees.
        theta_fc: Field capacity (H, W), m³/m³. Uses defaults if None.
        theta_wp: Wilting point (H, W), m³/m³. Uses defaults if None.
        use_dual_kc: If True, use dual-Kc; else single Kc.
        vhi_alpha: VCI weight in VHI.
        period_days: Advisory period length in days.
    
    Returns:
        Dict containing all advisory layers and summary statistics.
    """
    H, W = ndvi_current.shape
    
    # Use default soil parameters if SoilGrids data unavailable
    if theta_fc is None or theta_wp is None:
        soil_params = get_default_soil_params((H, W))
        theta_fc = soil_params["theta_fc"]
        theta_wp = soil_params["theta_wp"]
    
    # --- Step 1: Compute ET₀ ---
    et0_daily = compute_et0_penman_monteith(
        t_mean_c, rn_mj, u10_ms, v10_ms, t_dew_c, elevation_m
    )
    et0_period = et0_daily * period_days  # scale to period
    
    # --- Step 2: Compute ETc per crop type ---
    etc_map = np.zeros((H, W), dtype=np.float32)
    root_depth_map = np.zeros((H, W), dtype=np.float32)
    kc_map = np.zeros((H, W), dtype=np.float32)
    
    for code, crop_name in enumerate(crop_classes):
        crop_mask = crop_map == code
        if not crop_mask.any():
            continue
        
        dsp_crop = days_since_planting.copy()
        dsp_crop[~crop_mask] = 0.0
        
        kc = interpolate_kc(crop_name, dsp_crop)
        rd = get_root_depth(crop_name, dsp_crop)
        
        etc = compute_etc(et0_period, kc)
        
        etc_map[crop_mask] = etc[crop_mask]
        kc_map[crop_mask] = kc[crop_mask]
        root_depth_map[crop_mask] = rd[crop_mask]
    
    # --- Step 3: Compute TAW and RAW per crop type ---
    taw_map = get_taw(theta_fc, theta_wp, root_depth_map)
    
    kc_db = load_kc_database()
    p_map = np.full((H, W), 0.5, dtype=np.float32)  # default p
    for code, crop_name in enumerate(crop_classes):
        crop_mask = crop_map == code
        if not crop_mask.any():
            continue
        crop_key = crop_name.lower().replace(" ", "_")
        p_val = float(kc_db.get(crop_key, {}).get("p", 0.5))
        p_map[crop_mask] = p_val
    
    raw_map = get_raw(taw_map, p_map.mean())  # simplified: use mean p
    # Pixel-specific RAW
    raw_map = taw_map * p_map
    
    # --- Step 4: Soil water balance ---
    dr_prev = np.zeros((H, W), dtype=np.float32)  # start from field capacity
    rainfall_period = rainfall_mm * period_days
    
    balance = update_soil_water_balance(
        dr_prev, rainfall_period, etc_map, slope_deg, taw_map
    )
    dr_current = balance["dr_current"]
    
    # --- Step 5: Irrigation advisory ---
    irr_need_mm = compute_irrigation_need(dr_current, raw_map, taw_map)
    advisory_code = classify_irrigation_advisory(dr_current, raw_map, taw_map)
    
    # --- Step 6: Stress analysis ---
    stress_results = run_stress_analysis(
        ndvi_current=ndvi_current,
        ndvi_hist_min=ndvi_hist_min,
        ndvi_hist_max=ndvi_hist_max,
        lst_current=lst_current,
        lst_hist_min=lst_hist_min,
        lst_hist_max=lst_hist_max,
        et_actual=et_actual,
        et_potential=et0_daily,
        vh_anomaly=vh_anomaly,
        stage_map=stage_map,
        vhi_alpha=vhi_alpha,
    )
    
    # --- Step 7: Compute area statistics ---
    pixel_area_ha = 0.09  # 30m × 30m = 0.09 ha
    total_area_ha = H * W * pixel_area_ha
    
    # Advisory area breakdown
    advisory_areas = {}
    advisory_texts = {}
    for code_val in range(4):
        n_pixels = int(np.sum(advisory_code == code_val))
        advisory_areas[get_advisory_text(code_val)] = round(n_pixels * pixel_area_ha, 1)
        advisory_texts[code_val] = get_advisory_text(code_val)
    
    # Total irrigation volume (ML = 10^6 L = 1000 m³)
    irr_volume_mm_ha = float(np.nanmean(irr_need_mm))
    total_irr_volume_ML = irr_volume_mm_ha * 10 * total_area_ha / 1e3  # mm/ha → ML
    
    # Crop-wise summary
    crop_summary = []
    for code, crop_name in enumerate(crop_classes):
        crop_mask = crop_map == code
        n = int(crop_mask.sum())
        if n == 0:
            continue
        area_ha = round(n * pixel_area_ha, 1)
        mean_irr = float(np.nanmean(irr_need_mm[crop_mask]))
        mean_stress = float(np.nanmean(stress_results["weighted_stress"][crop_mask]))
        dominant_advisory = int(np.bincount(
            advisory_code[crop_mask].ravel(),
            minlength=4
        ).argmax())
        crop_summary.append({
            "crop": crop_name,
            "area_ha": area_ha,
            "mean_irr_need_mm": round(mean_irr, 1),
            "mean_stress_score": round(mean_stress, 1),
            "advisory": get_advisory_text(dominant_advisory),
            "urgency_code": dominant_advisory,
        })
    
    crop_summary = sorted(crop_summary, key=lambda x: x["urgency_code"], reverse=True)
    
    return {
        # Spatial layers
        "et0_daily": et0_daily,
        "et0_period": et0_period,
        "etc_period": etc_map,
        "kc_map": kc_map,
        "root_depth_map": root_depth_map,
        "taw_map": taw_map,
        "raw_map": raw_map,
        "dr_current": dr_current,
        "irr_need_mm": irr_need_mm,
        "advisory_code": advisory_code,
        # Stress layers
        **{f"stress_{k}": v for k, v in stress_results.items()},
        # Summary stats
        "total_area_ha": total_area_ha,
        "total_irr_volume_ML": round(total_irr_volume_ML, 2),
        "advisory_areas": advisory_areas,
        "advisory_texts": advisory_texts,
        "crop_summary": crop_summary,
        "stress_summary": stress_results["summary"],
        "period_days": period_days,
    }


def generate_block_summary(
    advisory_code: np.ndarray,
    crop_map: np.ndarray,
    crop_classes: List[str],
    irr_need_mm: np.ndarray,
    block_size: int = 50,
) -> pd.DataFrame:
    """Generate block/district-level summary of advisory.
    
    Divides the area into spatial blocks and computes aggregate statistics.
    
    Args:
        advisory_code: Advisory code map (H, W).
        crop_map: Crop type code map (H, W).
        crop_classes: List of crop class names.
        irr_need_mm: Irrigation need map (H, W), mm.
        block_size: Block size in pixels.
    
    Returns:
        DataFrame with columns: Block, Area_ha, Dominant_Crop,
        Mean_Deficit_mm, Priority, Advisory.
    """
    H, W = advisory_code.shape
    rows = []
    block_id = 1
    pixel_area_ha = 0.09
    
    for r in range(0, H, block_size):
        for c in range(0, W, block_size):
            block_adv = advisory_code[r:r + block_size, c:c + block_size]
            block_crop = crop_map[r:r + block_size, c:c + block_size]
            block_irr = irr_need_mm[r:r + block_size, c:c + block_size]
            
            n_pixels = block_adv.size
            area_ha = round(n_pixels * pixel_area_ha, 1)
            
            # Dominant crop
            crop_counts = np.bincount(block_crop.ravel(), minlength=len(crop_classes))
            dominant_crop_idx = int(np.argmax(crop_counts))
            dominant_crop = crop_classes[dominant_crop_idx] if dominant_crop_idx < len(crop_classes) else "Unknown"
            
            # Advisory urgency
            adv_counts = np.bincount(block_adv.ravel(), minlength=4)
            dominant_adv = int(np.argmax(adv_counts))
            
            rows.append({
                "Block": f"Block-{block_id:03d}",
                "Area_ha": area_ha,
                "Dominant_Crop": dominant_crop.capitalize(),
                "Mean_Deficit_mm": round(float(np.nanmean(block_irr)), 1),
                "Priority": dominant_adv,
                "Advisory": get_advisory_text(dominant_adv),
            })
            block_id += 1
    
    df = pd.DataFrame(rows)
    return df.sort_values("Priority", ascending=False).reset_index(drop=True)
