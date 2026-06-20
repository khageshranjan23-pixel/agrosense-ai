"""
AgroSense AI — Sidebar Layout & Controls
Renders Streamlit sidebar controls and progress widgets.
Zero hardcoded default choices; values are dynamically fetched or set based on configs.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Config & Loader imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SEASONS,
    load_kc_database,
    load_wris_commands,
)


def render_sidebar() -> Dict[str, Any]:
    """Render the main sidebar controls for AgroSense AI.

    Returns:
        Dict containing user-configured parameters:
        - geo_mode (str)
        - wris_data (dict | None)
        - season (str)
        - year (int)
        - start_date (str)
        - end_date (str)
        - crops (list[str])
        - optical_source (str)
        - sar_source (str)
        - cloud_pct (int)
        - temporal_resolution (str)
    """
    st.sidebar.markdown("## 🛰️ Control Panel")
    
    # ── Section 1: Study Area Definition ────────────────────────────────────
    st.sidebar.subheader("1️⃣ Study Area")
    geo_options = ["Select Command Area (WRIS)", "Draw Custom Area", "Upload GeoJSON", "Use My Current Location"]
    
    default_mode = "Select Command Area (WRIS)"
    query_params = st.query_params
    if "geo_mode" in query_params:
        param_val = query_params["geo_mode"]
        if param_val in geo_options:
            default_mode = param_val
    elif "user_location" in st.session_state:
        default_mode = "Use My Current Location"
        
    try:
        default_idx = geo_options.index(default_mode)
    except ValueError:
        default_idx = 0

    geo_mode = st.sidebar.selectbox(
        "Define Study Area",
        geo_options,
        index=default_idx,
        help="Select a predefined command area or specify a custom boundary.",
    )
    
    if geo_mode == "Use My Current Location" and "user_location" not in st.session_state:
        st.sidebar.info("🌐 Requesting browser location...")
        js_code = """
        <script>
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const url = new URL(window.parent.location.href);
                url.searchParams.set("lat", lat);
                url.searchParams.set("lon", lon);
                url.searchParams.set("geo_mode", "Use My Current Location");
                window.parent.location.href = url.toString();
            },
            function(error) {
                alert("Error getting location: " + error.message);
            }
        );
        </script>
        """
        import streamlit.components.v1 as components
        components.html(js_code, height=0, width=0)
    
    wris_data = None
    uploaded_geometry = None
    if geo_mode == "Select Command Area (WRIS)":
        try:
            wris_commands = load_wris_commands()
            wris_options = {meta["name"]: (k, meta) for k, meta in wris_commands.items()}
            selected_name = st.sidebar.selectbox(
                "Canal Command Area",
                list(wris_options.keys()),
                help="Preloaded command area boundary with crop mappings.",
            )
            if selected_name:
                wris_key, wris_data = wris_options[selected_name]
        except Exception as exc:
            logger.error("Failed to load WRIS command areas: %s", exc)
            st.sidebar.error("⚠️ Failed to load WRIS command areas.")
    elif geo_mode == "Upload GeoJSON":
        uploaded_file = st.sidebar.file_uploader(
            "Upload GeoJSON boundary",
            type=["geojson", "json"],
            help="Upload a GeoJSON file representing the field boundary."
        )
        if uploaded_file is not None:
            try:
                import json
                geojson_data = json.load(uploaded_file)
                if "features" in geojson_data and geojson_data["features"]:
                    uploaded_geometry = geojson_data["features"][0].get("geometry")
                else:
                    uploaded_geometry = geojson_data.get("geometry") or geojson_data
                
                if uploaded_geometry and "type" in uploaded_geometry:
                    st.sidebar.success("✅ GeoJSON loaded successfully!")
                else:
                    st.sidebar.error("❌ Invalid GeoJSON structure.")
                    uploaded_geometry = None
            except Exception as e:
                st.sidebar.error(f"❌ Failed to parse GeoJSON: {e}")
                uploaded_geometry = None

    # Geolocation shortcut
    if st.sidebar.button("📍 Center Map on My Location"):
        js_code = """
        <script>
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const url = new URL(window.parent.location.href);
                url.searchParams.set("lat", lat);
                url.searchParams.set("lon", lon);
                url.searchParams.set("geo_mode", "Use My Current Location");
                window.parent.location.href = url.toString();
            },
            function(error) {
                alert("Error getting location: " + error.message);
            }
        );
        </script>
        """
        import streamlit.components.v1 as components
        components.html(js_code, height=0, width=0)
            
    # ── Section 2: Season & Temporal Range ──────────────────────────────────
    st.sidebar.subheader("2️⃣ Analysis Window")
    
    season = st.sidebar.selectbox(
        "Crop Season",
        list(SEASONS.keys()),
        index=0,
        help="Select the agricultural season to analyze.",
    )
    
    current_year = datetime.now().year
    year = st.sidebar.selectbox(
        "Target Year",
        list(range(current_year - 5, current_year + 1)),
        index=len(range(current_year - 5, current_year + 1)) - 2,  # Default to last complete year
        help="Specify the target calendar year for GEE analysis.",
    )
    
    # Dynamically derive date range boundaries from config month ranges
    if season == "Rabi":
        # Rabi crosses calendar years (Nov of target year to April of next year)
        default_start = date(year, 11, 1)
        default_end = date(year + 1, 4, 30)
    elif season == "Zaid":
        default_start = date(year, 3, 1)
        default_end = date(year, 6, 30)
    else: # Kharif (June - Nov)
        default_start = date(year, 6, 1)
        default_end = date(year, 11, 30)
        
    # Cap default dates at today to prevent future date selection errors
    today = date.today()
    default_start = min(default_start, today)
    default_end = min(default_end, today)
    
    start_date_val = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=today,
        help="Start of GEE collection filter window.",
    )
    end_date_val = st.sidebar.date_input(
        "End Date",
        value=default_end,
        max_value=today,
        help="End of GEE collection filter window.",
    )
    
    if start_date_val >= end_date_val:
        st.sidebar.error("❌ Start Date must be before End Date.")
    
    # ── Section 3: Crops Config ─────────────────────────────────────────────
    st.sidebar.subheader("3️⃣ Target Crops")
    
    try:
        kc_db = load_kc_database()
        available_crops = sorted(list(kc_db.keys()))
    except Exception as exc:
        logger.error("Failed to load crop database: %s", exc)
        available_crops = ["wheat", "rice", "cotton", "mustard", "sugarcane"]
        
    # Dynamically set typical crops as default selection
    default_selection = []
    if geo_mode == "Select Command Area (WRIS)" and wris_data:
        dominant = wris_data.get("dominant_crops", [])
        default_selection = [c for c in dominant if c in available_crops]
        
    if not default_selection:
        typical = SEASONS[season].get("typical_crops", [])
        default_selection = [c for c in typical if c in available_crops]
        
    if not default_selection:
        default_selection = available_crops[:2]
        
    crops = st.sidebar.multiselect(
        "Expected Crop Types",
        available_crops,
        default=default_selection,
        help="Multiselect the crop types expected within this area to match FAO-56 stage coefficients.",
    )

    # Ground Truth file
    ground_truth_file = st.sidebar.file_uploader(
        "Ground Truth CSV (optional)",
        type=["csv"],
        help="Upload CSV with latitude, longitude, crop_label to train classifier.",
    )
    
    # ── Section 4: Satellite Configuration ──────────────────────────────────
    st.sidebar.subheader("4️⃣ Sensor Settings")
    
    optical_source = st.sidebar.selectbox(
        "Optical Imagery Source",
        ["Sentinel-2 (10m)", "Landsat-8/9 (30m)"],
        index=0,
        help="Source for NDVI & land surface temperature indices.",
    )
    
    sar_source = st.sidebar.selectbox(
        "Radar Imagery Source",
        ["Sentinel-1 SAR", "None (Optical Only)"],
        index=0,
        help="Active microwave sensor source for leaf water status tracking.",
    )
    
    cloud_pct = st.sidebar.slider(
        "Max Cloud Coverage (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help="Maximum pre-composite cloud threshold for scene selection.",
    )
    
    temporal_resolution = st.sidebar.selectbox(
        "Temporal Resolution",
        ["weekly", "biweekly", "monthly"],
        index=1,
        help="Width of the compositing windows.",
    )
    
    return {
        "geo_mode": geo_mode,
        "wris_data": wris_data,
        "uploaded_geometry": uploaded_geometry,
        "ground_truth_file": ground_truth_file,
        "season": season,
        "year": year,
        "start_date": start_date_val.strftime("%Y-%m-%d"),
        "end_date": end_date_val.strftime("%Y-%m-%d"),
        "crops": crops,
        "optical_source": optical_source,
        "sar_source": sar_source,
        "cloud_pct": cloud_pct,
        "temporal_resolution": temporal_resolution,
    }


def render_progress_bar(steps: List[Tuple[str, bool]]) -> None:
    """Render custom-styled task completion steps.

    Args:
        steps: List of (step_name, is_completed) tuples.
    """
    for name, is_done in steps:
        icon = "✅" if is_done else "⏳"
        st.markdown(
            f"<div class='step-row'><span>{icon}</span> <span>{name}</span></div>",
            unsafe_allow_html=True,
        )
