"""
AgroSense AI — Global Configuration
All user-configurable parameters. Zero hardcoded thresholds.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# ── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UI_DIR = BASE_DIR / "ui"
CORE_DIR = BASE_DIR / "core"

KC_DB_PATH = DATA_DIR / "kc_database.json"
STAGE_WEIGHTS_PATH = DATA_DIR / "stage_weights.json"
WRIS_COMMANDS_PATH = DATA_DIR / "wris_commands.json"
FALLBACK_DIR = DATA_DIR / "fallback"
CACHE_DIR = DATA_DIR / "cache"

# ── Sentinel-2 Configuration ─────────────────────────────────────────────
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_SCALE_M = 10  # native resolution in metres

# SCL cloud mask values to remove
SCL_CLOUD_VALUES = [3, 8, 9, 10, 11]  # shadows, medium/high cloud, cirrus, snow

# ── Sentinel-1 Configuration ─────────────────────────────────────────────
S1_COLLECTION = "COPERNICUS/S1_GRD"
S1_MODE = "IW"
S1_POLARIZATIONS = ["VV", "VH"]
S1_ORBIT = "DESCENDING"
S1_SPECKLE_KERNEL = 7

# ── ERA5 Configuration ─────────────────────────────────────────────────────
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_BANDS = [
    "total_evaporation_sum",
    "total_precipitation_sum",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
]

# ── CHIRPS Rainfall ────────────────────────────────────────────────────────
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"

# ── SRTM DEM ──────────────────────────────────────────────────────────────
SRTM_COLLECTION = "USGS/SRTMGL1_003"

# ── MODIS LST ─────────────────────────────────────────────────────────────
MODIS_LST_COLLECTION = "MODIS/061/MOD11A1"
MODIS_LST_BAND = "LST_Day_1km"
MODIS_LST_SCALE = 0.02  # Kelvin scale factor

# ── Season Definitions (DOY ranges) ───────────────────────────────────────
SEASONS: Dict[str, Dict[str, Any]] = {
    "Kharif": {
        "months": list(range(6, 12)),
        "start_month": 6,
        "end_month": 11,
        "description": "June – November (Monsoon season)",
        "typical_crops": ["rice", "cotton", "maize", "soybean", "groundnut", "sorghum", "pearl_millet"],
    },
    "Rabi": {
        "months": [11, 12, 1, 2, 3, 4],
        "start_month": 11,
        "end_month": 4,
        "description": "November – April (Winter season)",
        "typical_crops": ["wheat", "mustard", "chickpea", "lentil", "barley", "sunflower"],
    },
    "Zaid": {
        "months": list(range(3, 7)),
        "start_month": 3,
        "end_month": 6,
        "description": "March – June (Summer season)",
        "typical_crops": ["watermelon", "cucumber", "mungbean", "groundnut"],
    },
}

# ── Phenology Thresholds (normalized NDVI fractions) ─────────────────────
PHENO_SOS_THRESHOLD = 0.20    # 20% of NDVI range = green-up onset
PHENO_EOS_THRESHOLD = 0.20    # 20% of NDVI range = end of season
PHENO_PEAK_DROP = 0.20        # 20% drop from peak = senescence onset
SAVITZKY_WINDOW = 5
SAVITZKY_POLY = 3

# ── ML Model Defaults ──────────────────────────────────────────────────────
N_OPTUNA_TRIALS = 50
N_CV_FOLDS = 5
TRAIN_SPLIT = 0.70
MIN_SAMPLES_PER_CLASS = 30
LOW_CONFIDENCE_THRESHOLD = 0.60
ACCURACY_WARNING_THRESHOLD = 0.70
N_SHAP_FEATURES = 80

# ── Stress Detection ───────────────────────────────────────────────────────
VHI_ALPHA = 0.5              # weight of VCI in VHI (user adjustable)
HISTORICAL_YEARS = list(range(2019, 2024))  # baseline years for VCI/TCI
NO_STRESS_PERCENTILE = 40    # top 40th = no stress
MILD_STRESS_PERCENTILE = 20  # 20-40th = mild stress
# below 20th = severe stress

# ── Water Balance ──────────────────────────────────────────────────────────
RUNOFF_COEFF_FLAT = 0.10    # slope < 2%
RUNOFF_COEFF_GENTLE = 0.20  # slope 2-5%
RUNOFF_COEFF_MODERATE = 0.30  # slope 5-10%
RUNOFF_COEFF_STEEP = 0.45   # slope > 10%

# ── GEE Export Settings ────────────────────────────────────────────────────
GEE_EXPORT_SCALE_SMALL = 30   # metres, for area < 100 km²
GEE_EXPORT_SCALE_LARGE = 100  # metres, for area >= 100 km²
GEE_AREA_THRESHOLD_KM2 = 100
GEE_TIMEOUT_S = 30
GEE_MAX_PIXELS = 1e9

# ── GeoTIFF Export ────────────────────────────────────────────────────────
GEOTIFF_COMPRESS = "LZW"
GEOTIFF_CRS = "EPSG:4326"

# ── Streamlit Cache TTL ────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600  # 1 hour

# ── Colormap Definitions (for maps) ───────────────────────────────────────
CROP_COLORS: Dict[str, str] = {
    "wheat": "#FFD700",
    "rice": "#228B22",
    "cotton": "#FFFACD",
    "maize": "#FFA500",
    "soybean": "#6B8E23",
    "sugarcane": "#2E8B57",
    "groundnut": "#DEB887",
    "mustard": "#FFFF00",
    "sunflower": "#FFD700",
    "sorghum": "#CD853F",
    "pearl_millet": "#D2B48C",
    "chickpea": "#F5DEB3",
    "other": "#808080",
    "uncertain": "#C0C0C0",
}

STRESS_COLORS: Dict[str, str] = {
    "no_stress": "#00C853",
    "mild_stress": "#FFD600",
    "moderate_stress": "#FF6D00",
    "severe_stress": "#D50000",
}

IRRIGATION_COLORS: Dict[str, str] = {
    "no_irrigation": "#1B5E20",
    "irrigate_3_days": "#F9A825",
    "irrigate_immediately": "#E65100",
    "critical_alert": "#B71C1C",
}

# ── Loaders ────────────────────────────────────────────────────────────────
def load_kc_database() -> Dict[str, Any]:
    """Load FAO-56 crop coefficient database from JSON.

    Returns:
        Dict mapping crop names to their FAO-56 Kc stage values and metadata.

    Raises:
        FileNotFoundError: If kc_database.json does not exist at KC_DB_PATH.
        json.JSONDecodeError: If the file contains malformed JSON.
    """
    with open(KC_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stage_weights() -> Dict[str, float]:
    """Load growth stage stress impact weights.

    Each stage entry is expected to have a ``weight`` key (float 0–1)
    indicating how severely water stress at that stage penalises yield.

    Returns:
        Dict mapping stage name strings to their float weight values.

    Raises:
        FileNotFoundError: If stage_weights.json does not exist.
        KeyError: If any stage entry is missing the required ``weight`` key.
        json.JSONDecodeError: If the file contains malformed JSON.
    """
    with open(STAGE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v["weight"] for k, v in data.items()}


def load_wris_commands() -> Dict[str, Any]:
    """Load WRIS India canal command areas.

    Returns:
        Dict mapping command area names/IDs to their geometry and metadata.

    Raises:
        FileNotFoundError: If wris_commands.json does not exist.
        json.JSONDecodeError: If the file contains malformed JSON.
    """
    with open(WRIS_COMMANDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
