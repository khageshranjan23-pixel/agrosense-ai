# -*- coding: utf-8 -*-
"""
AgroSense AI — Redesigned Sidebar Layout & Controls
Bilingual, with collapsible expanders, custom emojis, and modern UI controls.
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
from translations import translate

def render_sidebar() -> Dict[str, Any]:
    """Render the redesigned sidebar controls for AgroSense AI.

    Returns:
        Dict containing user-configured parameters.
    """
    st.sidebar.markdown(
        "<div style='text-align: center; margin-bottom: 12px;'>"
        "<h2 style='font-family: \"Outfit\", sans-serif; color: #1B5E20; margin: 0;'>🛰️ AgroSense AI</h2>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # ── Language Selector (Top Priority) ──────────────────────────────────
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
        
    lang_choice = st.sidebar.radio(
        "🌐 Language / भाषा चुनें",
        options=["English", "हिंदी"],
        index=0 if st.session_state["lang"] == "en" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state["lang"] = "en" if lang_choice == "English" else "hi"
    lang = st.session_state["lang"]
    
    # Collapsible 1: Study Area
    with st.sidebar.expander(f"📍 {translate('sb_location_header', lang)}", expanded=True):
        geo_mode_mapping = {
            translate("option_wris", lang): "Select Command Area (WRIS)",
            translate("option_map", lang): "Draw Custom Area",
            translate("option_upload", lang): "Upload GeoJSON",
            "🌐 Use Location" if lang == "en" else "🌐 वर्तमान लोकेशन": "Use My Current Location"
        }
        
        default_mode = "Select Command Area (WRIS)"
        query_params = st.query_params
        if "geo_mode" in query_params:
            param_val = query_params["geo_mode"]
            if param_val in geo_mode_mapping.values():
                default_mode = param_val
        elif "user_location" in st.session_state:
            default_mode = "Use My Current Location"
            
        default_idx = 0
        for idx, (disp, internal) in enumerate(geo_mode_mapping.items()):
            if internal == default_mode:
                default_idx = idx
                break
                
        selected_disp = st.selectbox(
            "Define Study Area" if lang == "en" else "क्षेत्र का प्रकार",
            list(geo_mode_mapping.keys()),
            index=default_idx
        )
        geo_mode = geo_mode_mapping[selected_disp]
        
        if geo_mode == "Use My Current Location" and "user_location" not in st.session_state:
            st.info("🌐 Requesting browser location..." if lang == "en" else "🌐 आपकी लोकेशन खोजी जा रही है...")
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
                selected_name = st.selectbox(
                    "Canal Command Area" if lang == "en" else "सिंचाई कमांड क्षेत्र",
                    list(wris_options.keys()),
                    help="Preloaded command area boundary." if lang == "en" else "पहले से सेव की गई सिंचाई क्षेत्र की सीमा।"
                )
                if selected_name:
                    wris_key, wris_data = wris_options[selected_name]
            except Exception as exc:
                logger.error("Failed to load WRIS command areas: %s", exc)
                st.error("⚠️ Failed to load WRIS command areas.")
        elif geo_mode == "Upload GeoJSON":
            uploaded_file = st.file_uploader(
                "Upload GeoJSON boundary" if lang == "en" else "GeoJSON बाउंड्री अपलोड करें",
                type=["geojson", "json"],
                help="Upload a GeoJSON file representing the field boundary." if lang == "en" else "अपने खेत की जियो-बाउंड्री फाइल अपलोड करें।"
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
                        st.success("✅ GeoJSON loaded successfully!" if lang == "en" else "✅ जियो-बाउंड्री लोड हो गई!")
                    else:
                        st.error("❌ Invalid GeoJSON structure." if lang == "en" else "❌ अमान्य GeoJSON फाइल।")
                        uploaded_geometry = None
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    uploaded_geometry = None

        if st.button("📍 " + ("Center Map on My Location" if lang == "en" else "मेरी लोकेशन पर नक्शा लाएं"), use_container_width=True):
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
                    alert("Error: " + error.message);
                }
            );
            </script>
            """
            import streamlit.components.v1 as components
            components.html(js_code, height=0, width=0)

    # Collapsible 2: Season & Crops
    with st.sidebar.expander(f"🌾 {translate('sb_crops_header', lang)}", expanded=True):
        season_mapping = {
            "🌧️ Kharif" if lang == "en" else "🌧️ खरीफ": "Kharif",
            "❄️ Rabi" if lang == "en" else "❄️ रबी": "Rabi",
            "☀️ Zaid" if lang == "en" else "☀️ जायद": "Zaid"
        }
        
        selected_season_disp = st.selectbox(
            "Select Season" if lang == "en" else "फसल का मौसम",
            list(season_mapping.keys()),
            index=0
        )
        season = season_mapping[selected_season_disp]
        
        current_year = datetime.now().year
        year = st.selectbox(
            "Target Year" if lang == "en" else "साल चुनें",
            list(range(current_year - 5, current_year + 1)),
            index=len(range(current_year - 5, current_year + 1)) - 2,
        )
        
        if season == "Rabi":
            default_start = date(year, 11, 1)
            default_end = date(year + 1, 4, 30)
        elif season == "Zaid":
            default_start = date(year, 3, 1)
            default_end = date(year, 6, 30)
        else: # Kharif
            default_start = date(year, 6, 1)
            default_end = date(year, 11, 30)
            
        today = date.today()
        default_start = min(default_start, today)
        default_end = min(default_end, today)
        
        start_date_val = st.date_input("Start Date" if lang == "en" else "शुरू की तारीख", value=default_start, max_value=today)
        end_date_val = st.date_input("End Date" if lang == "en" else "आखिरी तारीख", value=default_end, max_value=today)
        
        if start_date_val >= end_date_val:
            st.error("❌ Start Date must be before End Date." if lang == "en" else "❌ शुरू की तारीख आखिरी तारीख से पहले होनी चाहिए।")
            
        try:
            kc_db = load_kc_database()
            available_crops = sorted(list(kc_db.keys()))
        except Exception as exc:
            logger.error("Failed to load crop database: %s", exc)
            available_crops = ["wheat", "rice", "cotton", "mustard", "sugarcane"]
            
        default_selection = []
        if geo_mode == "Select Command Area (WRIS)" and wris_data:
            dominant = wris_data.get("dominant_crops", [])
            default_selection = [c for c in dominant if c in available_crops]
            
        if not default_selection:
            typical = SEASONS[season].get("typical_crops", [])
            default_selection = [c for c in typical if c in available_crops]
            
        if not default_selection:
            default_selection = available_crops[:2]
            
        crops = st.multiselect(
            "Expected Crop Types" if lang == "en" else "उगाई गई फसलें",
            available_crops,
            default=default_selection
        )
        
        ground_truth_file = st.file_uploader(
            "Ground Truth CSV (optional)" if lang == "en" else "ग्राउंड ट्रुथ डेटा (ऑप्शनल)",
            type=["csv"],
            help="CSV with latitude, longitude, crop_label to train classifier." if lang == "en" else "क्लासीफायर को ट्रेन करने के लिए अक्षांश और देशांतर वाली CSV फाइल।"
        )

    # Collapsible 3: Irrigation & Sensitivity Controls
    with st.sidebar.expander(f"💧 {translate('sb_method_header', lang)}", expanded=True):
        if "irrigation_method" not in st.session_state:
            st.session_state["irrigation_method"] = "Flood"
            
        st.markdown(f"**{translate('sb_method_header', lang)}:**")
        cols = st.columns(3)
        with cols[0]:
            if st.button("🌊 Flood" if lang == "en" else "🌊 बहाव", use_container_width=True, type="primary" if st.session_state["irrigation_method"] == "Flood" else "secondary"):
                st.session_state["irrigation_method"] = "Flood"
        with cols[1]:
            if st.button("🌧️ Sprink." if lang == "en" else "🌧️ फव्वारा", use_container_width=True, type="primary" if st.session_state["irrigation_method"] == "Sprinkler" else "secondary"):
                st.session_state["irrigation_method"] = "Sprinkler"
        with cols[2]:
            if st.button("💧 Drip" if lang == "en" else "💧 टपकन", use_container_width=True, type="primary" if st.session_state["irrigation_method"] == "Drip" else "secondary"):
                st.session_state["irrigation_method"] = "Drip"
                
        # Sensitivity Selectbox
        sens_mapping = {
            translate("sensitivity_early", lang): "early",
            translate("sensitivity_late", lang): "late"
        }
        selected_sens = st.selectbox(
            translate("sb_sensitivity", lang),
            list(sens_mapping.keys()),
            index=0
        )
        st.session_state["stress_sensitivity"] = sens_mapping[selected_sens]

    # Collapsible 4: Advanced Sensors
    with st.sidebar.expander("⚙️ Advanced Sensors" if lang == "en" else "⚙️ एडवांस सेंसर सेटिंग्स", expanded=False):
        optical_source = st.selectbox(
            "Optical Source" if lang == "en" else "ऑप्टिकल सैटेलाइट",
            ["Sentinel-2 (10m)", "Landsat-8/9 (30m)"],
            index=0
        )
        sar_source = st.selectbox(
            "Radar Source" if lang == "en" else "राडार सैटेलाइट",
            ["Sentinel-1 SAR", "None (Optical Only)"],
            index=0
        )
        cloud_pct = st.slider(
            "Max Clouds (%)" if lang == "en" else "बादल की सीमा (%)",
            0, 100, 20, 5
        )
        temp_res_mapping = {
            "Weekly" if lang == "en" else "साप्ताहिक": "weekly",
            "Bi-weekly" if lang == "en" else "पाक्षिक (15 दिन)": "biweekly",
            "Monthly" if lang == "en" else "मासिक": "monthly"
        }
        selected_temp_res = st.selectbox(
            "Temporal Resolution" if lang == "en" else "डेटा फ्रीक्वेंसी",
            list(temp_res_mapping.keys()),
            index=1
        )
        temporal_resolution = temp_res_mapping[selected_temp_res]

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
    """Render custom-styled task completion steps."""
    for name, is_done in steps:
        icon = "✅" if is_done else "⏳"
        st.markdown(
            f"<div class='step-row'><span>{icon}</span> <span>{name}</span></div>",
            unsafe_allow_html=True,
        )
