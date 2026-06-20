"""
AgroSense AI — Plotly Visualization Library
All charts are dynamic, data-driven, and styled for hackathon-winning aesthetics.
Colorful, vibrant, and professional — no plain gray charts.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from translations import translate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette & shared theme
# ---------------------------------------------------------------------------

# Colorful palette for AgroSense AI
AGRO_PALETTE: List[str] = [
    "#00C853", "#FFD600", "#FF6D00", "#D50000",
    "#0091EA", "#AA00FF", "#00B0FF", "#64DD17",
    "#FFAB00", "#DD2C00", "#1DE9B6", "#F50057",
]

CHART_TEMPLATE: Dict[str, Any] = dict(
    plot_bgcolor="rgba(248,250,246,0.95)",
    paper_bgcolor="rgba(255,255,255,1)",
    font=dict(family="Outfit, Inter, sans-serif", size=13, color="#1B301E"),
    title_font=dict(size=18, color="#123E1C", family="Outfit, sans-serif"),
    colorway=AGRO_PALETTE,
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
    ),
)

def translate_crop(crop_name: str, lang: str) -> str:
    """Helper to translate crop names to Hindi/English."""
    crop_map = {
        "wheat": {"en": "Wheat", "hi": "गेहूं (Wheat)"},
        "rice": {"en": "Rice", "hi": "धान (Rice)"},
        "cotton": {"en": "Cotton", "hi": "कपास (Cotton)"},
        "mustard": {"en": "Mustard", "hi": "सरसों (Mustard)"},
        "sugarcane": {"en": "Sugarcane", "hi": "गन्ना (Sugarcane)"},
        "maize": {"en": "Maize", "hi": "मक्का (Maize)"},
        "soybean": {"en": "Soybean", "hi": "सोयाबीन (Soybean)"},
        "chickpea": {"en": "Chickpea", "hi": "चना (Chickpea)"},
        "groundnut": {"en": "Groundnut", "hi": "मूंगफली (Groundnut)"},
    }
    key = crop_name.lower().strip()
    if key in crop_map:
        return crop_map[key][lang]
    return crop_name.capitalize()



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_agro_theme(fig: go.Figure) -> go.Figure:
    """Apply the AgroSense AI chart theme to an existing figure.

    Args:
        fig: Any Plotly Figure object.

    Returns:
        The same figure with the AgroSense theme applied in-place.
    """
    fig.update_layout(**CHART_TEMPLATE)
    return fig


def _safe_index(lst: List[Any], value: Any) -> Optional[int]:
    """Return the index of *value* in *lst*, or None if not found.

    Args:
        lst: Source list.
        value: Value to locate.

    Returns:
        Integer index or None.
    """
    try:
        return lst.index(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plot_ndvi_timeseries(
    dates: List[str],
    ndvi_values: List[float],
    ndvi_smoothed: Optional[List[float]] = None,
    sos_date: Optional[str] = None,
    peak_date: Optional[str] = None,
    eos_date: Optional[str] = None,
    crop_label: str = "Field Average",
) -> go.Figure:
    """Plot NDVI time series with phenological stage markers and shaded regions."""
    lang = st.session_state.get("lang", "en")
    
    if not dates:
        raise ValueError("dates list must not be empty.")
    if len(dates) != len(ndvi_values):
        raise ValueError(
            f"dates ({len(dates)}) and ndvi_values ({len(ndvi_values)}) must have the same length."
        )
    if ndvi_smoothed is not None and len(ndvi_smoothed) != len(dates):
        raise ValueError("ndvi_smoothed must have the same length as dates.")

    fig = go.Figure()

    # Raw NDVI scatter-line
    raw_name = "NDVI (raw)" if lang == "en" else "फसल हरियाली (कच्चा)"
    fig.add_trace(go.Scatter(
        x=dates,
        y=ndvi_values,
        mode="markers+lines",
        name=raw_name,
        line=dict(color="#81C784", width=1, dash="dot"),
        marker=dict(size=5, color="#4CAF50"),
        opacity=0.7,
        hovertemplate="Date: %{x}<br>NDVI: %{y:.3f}<extra>Raw</extra>",
    ))

    # Smoothed NDVI with area fill
    if ndvi_smoothed is not None:
        smooth_name = "NDVI (smoothed)" if lang == "en" else "फसल हरियाली (साफ)"
        fig.add_trace(go.Scatter(
            x=dates,
            y=ndvi_smoothed,
            mode="lines",
            name=smooth_name,
            line=dict(color="#2E7D32", width=3),
            fill="tozeroy",
            fillcolor="rgba(46,125,50,0.12)",
            hovertemplate="Date: %{x}<br>NDVI: %{y:.3f}<extra>Smoothed</extra>",
        ))

    # Phenological stage background regions
    veg_label = "Vegetative" if lang == "en" else "बढ़ने की स्टेज (Vegetative)"
    grain_label = "Grain Fill" if lang == "en" else "दाना भरने की स्टेज (Grain Fill)"
    
    if sos_date and peak_date:
        fig.add_vrect(
            x0=sos_date,
            x1=peak_date,
            fillcolor="rgba(0,200,83,0.08)",
            line_width=0,
            annotation_text=veg_label,
            annotation_position="top left",
            annotation_font=dict(color="#2E7D32", size=11),
        )
    if peak_date and eos_date:
        fig.add_vrect(
            x0=peak_date,
            x1=eos_date,
            fillcolor="rgba(255,214,0,0.08)",
            line_width=0,
            annotation_text=grain_label,
            annotation_position="top left",
            annotation_font=dict(color="#E65100", size=11),
        )

    # Phenology event markers
    marker_specs: List[Tuple[Optional[str], str, str, str]] = [
        (sos_date, "SOS", "#00C853", "triangle-up"),
        (peak_date, "PEAK", "#FFD600", "star"),
        (eos_date, "EOS", "#FF6D00", "triangle-down"),
    ]
    for event_date, label, color, symbol in marker_specs:
        if event_date is None:
            continue
        idx = _safe_index(dates, event_date)
        if idx is None:
            logger.warning("Phenology date %s not found in dates list; skipping.", event_date)
            continue
        val = ndvi_values[idx]
        fig.add_trace(go.Scatter(
            x=[event_date],
            y=[val],
            mode="markers+text",
            name=label,
            marker=dict(size=14, color=color, symbol=symbol,
                        line=dict(color="white", width=1)),
            text=[label],
            textposition="top center",
            textfont=dict(size=11, color=color, family="Inter, sans-serif"),
            showlegend=True,
            hovertemplate=f"{label}: %{{x}}<br>NDVI: %{{y:.3f}}<extra></extra>",
        ))

    crop_disp = translate_crop(crop_label, lang)
    title_text = f"NDVI Time Series - {crop_disp}" if lang == "en" else f"फसल स्वास्थ्य ग्राफ (NDVI) - {crop_disp}"
    xaxis_lbl = "Date" if lang == "en" else "तारीख (Date)"
    yaxis_lbl = "NDVI Vigor" if lang == "en" else "हरियाली सूचकांक (NDVI)"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=xaxis_lbl,
        yaxis_title=yaxis_lbl,
        yaxis_range=[0.0, 1.0],
        hovermode="x unified",
        height=420,
        margin=dict(t=60, b=60, l=60, r=40),
        **CHART_TEMPLATE,
    )
    return fig


def plot_stress_trend(
    dates: List[str],
    vci_values: List[float],
    tci_values: List[float],
    vhi_values: List[float],
    cwsi_values: Optional[List[float]] = None,
) -> go.Figure:
    """Plot Vegetation Health Index components and optional CWSI as an area/line chart."""
    lang = st.session_state.get("lang", "en")
    
    if not dates:
        raise ValueError("dates list must not be empty.")
    for name, arr in [("vci_values", vci_values), ("tci_values", tci_values),
                      ("vhi_values", vhi_values)]:
        if len(arr) != len(dates):
            raise ValueError(f"{name} length {len(arr)} must match dates length {len(dates)}.")
    if cwsi_values is not None and len(cwsi_values) != len(dates):
        raise ValueError("cwsi_values must have the same length as dates.")

    fig = go.Figure()

    # VHI - primary filled area
    vhi_lbl = "VHI (Health Index)" if lang == "en" else "फसल स्वास्थ्य सूचकांक (VHI)"
    fig.add_trace(go.Scatter(
        x=dates,
        y=vhi_values,
        mode="lines",
        name=vhi_lbl,
        line=dict(color="#00C853", width=3),
        fill="tozeroy",
        fillcolor="rgba(0,200,83,0.15)",
        hovertemplate="Date: %{x}<br>VHI: %{y:.1f}<extra></extra>",
    ))

    # VCI - dashed overlay
    vci_lbl = "VCI (Vegetation Condition)" if lang == "en" else "फसल हरियाली स्थिति (VCI)"
    fig.add_trace(go.Scatter(
        x=dates,
        y=vci_values,
        mode="lines",
        name=vci_lbl,
        line=dict(color="#4CAF50", width=2, dash="dash"),
        hovertemplate="Date: %{x}<br>VCI: %{y:.1f}<extra></extra>",
    ))

    # TCI - dashed overlay
    tci_lbl = "TCI (Temperature Condition)" if lang == "en" else "तापमान स्थिति (TCI)"
    fig.add_trace(go.Scatter(
        x=dates,
        y=tci_values,
        mode="lines",
        name=tci_lbl,
        line=dict(color="#FF6D00", width=2, dash="dash"),
        hovertemplate="Date: %{x}<br>TCI: %{y:.1f}<extra></extra>",
    ))

    # Optional CWSI (rescaled x100 for comparison)
    if cwsi_values:
        cwsi_scaled = [v * 100.0 for v in cwsi_values]
        cwsi_lbl = "CWSI x 100" if lang == "en" else "फसल जल तनाव सूचकांक (CWSI) x 100"
        fig.add_trace(go.Scatter(
            x=dates,
            y=cwsi_scaled,
            mode="lines",
            name=cwsi_lbl,
            line=dict(color="#D50000", width=2),
            hovertemplate="Date: %{x}<br>CWSI x100: %{y:.1f}<extra></extra>",
        ))

    # Threshold reference lines
    t1_txt = "Stress threshold (VHI < 40)" if lang == "en" else "तनाव सीमा (VHI < 40)"
    t2_txt = "Severe stress (VHI < 25)" if lang == "en" else "गंभीर तनाव सीमा (VHI < 25)"
    fig.add_hline(
        y=40,
        line_dash="dot",
        line_color="#FF6D00",
        line_width=1.5,
        annotation_text=t1_txt,
        annotation_position="right",
        annotation_font=dict(color="#FF6D00", size=11),
    )
    fig.add_hline(
        y=25,
        line_dash="dot",
        line_color="#D50000",
        line_width=1.5,
        annotation_text=t2_txt,
        annotation_position="right",
        annotation_font=dict(color="#D50000", size=11),
    )

    title_txt = "Moisture Stress Trend (8-day Rolling)" if lang == "en" else "जल तनाव सूचकांक (8-दिवसीय रोलिंग)"
    xaxis_lbl = "Date" if lang == "en" else "तारीख (Date)"
    yaxis_lbl = "Index Value (0-100)" if lang == "en" else "सूचकांक मान (0-100)"

    fig.update_layout(
        title=title_txt,
        xaxis_title=xaxis_lbl,
        yaxis_title=yaxis_lbl,
        yaxis_range=[0, 105],
        hovermode="x unified",
        height=400,
        margin=dict(t=60, b=60, l=60, r=120),
        **CHART_TEMPLATE,
    )
    return fig


def plot_crop_area_pie(
    crop_names: List[str],
    areas_ha: List[float],
    colors: Optional[List[str]] = None,
) -> go.Figure:
    """Render crop area distribution as an interactive donut chart."""
    lang = st.session_state.get("lang", "en")
    
    if not crop_names:
        raise ValueError("crop_names must not be empty.")
    if len(crop_names) != len(areas_ha):
        raise ValueError("crop_names and areas_ha must have the same length.")
    if any(a < 0 for a in areas_ha):
        raise ValueError("All area values must be non-negative.")

    if colors is None:
        colors = AGRO_PALETTE[: len(crop_names)]

    translated_labels = [translate_crop(c, lang) for c in crop_names]
    hover_lbl = "<b>%{label}</b><br>Area: %{value:.1f} ha<br>Share: %{percent}" if lang == "en" else "<b>%{label}</b><br>रकबा: %{value:.1f} हेक्टेयर<br>हिस्सा: %{percent}"

    fig = go.Figure(go.Pie(
        labels=translated_labels,
        values=areas_ha,
        marker=dict(
            colors=colors,
            line=dict(color="white", width=2),
        ),
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(size=13, family="Outfit, sans-serif"),
        hovertemplate=hover_lbl,
        pull=[0.03] * len(crop_names),
    ))

    total_ha = sum(areas_ha)
    total_text = "Total" if lang == "en" else "कुल"
    fig.update_layout(
        title="Crop Area Distribution" if lang == "en" else "फसलों का क्षेत्र वितरण (रकबा)",
        annotations=[dict(
            text=f"<b>{total_ha:,.0f} ha</b><br>{total_text}",
            x=0.5,
            y=0.5,
            font_size=14,
            font_color="#2E7D32",
            font_family="Outfit, sans-serif",
            showarrow=False,
        )],
        height=420,
        margin=dict(t=60, b=20, l=20, r=20),
        **CHART_TEMPLATE,
    )
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
) -> go.Figure:
    """Render an interactive confusion matrix as an annotated heatmap."""
    lang = st.session_state.get("lang", "en")
    
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be a square 2-D array; got shape {cm.shape}.")
    if len(class_names) != cm.shape[0]:
        raise ValueError(
            f"class_names length ({len(class_names)}) must match cm dimension ({cm.shape[0]})."
        )

    # Row-normalise (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cm_norm: np.ndarray = cm / row_sums

    # Build annotation text matrix
    text_matrix: List[List[str]] = [
        [f"{int(cm[i, j])}<br>({cm_norm[i, j]:.1%})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ]

    nice_names = [translate_crop(c, lang) for c in class_names]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=nice_names,
        y=nice_names,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11, family="Outfit, sans-serif"),
        colorscale=[
            [0.0, "#FFFFFF"],
            [0.3, "#C8E6C9"],
            [0.6, "#66BB6A"],
            [1.0, "#1B5E20"],
        ],
        showscale=True,
        colorbar=dict(
            title="Recall" if lang == "en" else "शुद्धता",
            tickformat=".0%",
            len=0.8,
        ),
        hoverongaps=False,
        hovertemplate=(
            "Predicted: <b>%{x}</b><br>"
            "True: <b>%{y}</b><br>"
            "Recall: %{z:.1%}<extra></extra>"
        ) if lang == "en" else (
            "अनुमानित: <b>%{x}</b><br>"
            "वास्तविक: <b>%{y}</b><br>"
            "शुद्धता: %{z:.1%}<extra></extra>"
        ),
    ))

    # Highlight diagonal
    n = cm.shape[0]
    for i in range(n):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color="#00C853", width=2),
        )

    fig.update_layout(
        title="Classification Accuracy - Confusion Matrix" if lang == "en" else "वर्गीकरण शुद्धता - कन्फ्यूजन मैट्रिक्स",
        xaxis_title="Predicted Label" if lang == "en" else "अनुमानित फसल (Predicted)",
        yaxis_title="True Label" if lang == "en" else "वास्तविक फसल (True)",
        height=max(450, n * 55),
        margin=dict(t=70, b=70, l=100, r=80),
        **CHART_TEMPLATE,
    )
    return fig


def plot_irrigation_summary_bar(
    crop_names: List[str],
    irr_needs_mm: List[float],
    stress_scores: List[float],
) -> go.Figure:
    """Dual-axis bar + line chart of irrigation requirement and stress per crop."""
    lang = st.session_state.get("lang", "en")
    
    if not crop_names:
        raise ValueError("crop_names must not be empty.")
    if len(irr_needs_mm) != len(crop_names) or len(stress_scores) != len(crop_names):
        raise ValueError("All input lists must have the same length.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    nice_names = [translate_crop(c, lang) for c in crop_names]

    # Bars - irrigation need with diverging colorscale
    bar_name = "Irrigation Need (mm)" if lang == "en" else "सिंचाई की आवश्यकता (mm)"
    fig.add_trace(
        go.Bar(
            name=bar_name,
            x=nice_names,
            y=irr_needs_mm,
            marker=dict(
                color=irr_needs_mm,
                colorscale=[[0.0, "#00C853"], [0.5, "#FFD600"], [1.0, "#D50000"]],
                showscale=True,
                colorbar=dict(title="mm", x=1.08, len=0.7),
                line=dict(color="rgba(255,255,255,0.3)", width=1),
            ),
            opacity=0.88,
            text=[f"{v:.0f} mm" for v in irr_needs_mm],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>Irrigation: %{y:.1f} mm<extra></extra>",
        ),
        secondary_y=False,
    )

    # Line - stress score
    line_name = "Stress Score" if lang == "en" else "तनाव स्कोर"
    fig.add_trace(
        go.Scatter(
            name=line_name,
            x=nice_names,
            y=stress_scores,
            mode="lines+markers",
            marker=dict(size=10, color="#FF6D00", symbol="diamond",
                        line=dict(color="white", width=1.5)),
            line=dict(color="#FF6D00", width=2.5),
            hovertemplate="<b>%{x}</b><br>Stress: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Irrigation Need (mm)" if lang == "en" else "सिंचाई की आवश्यकता (mm)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(
        title_text="Stress Score (0-100)" if lang == "en" else "तनाव स्कोर (0-100)",
        secondary_y=True,
        range=[0, 110],
        showgrid=False,
    )
    fig.update_layout(
        title="Irrigation Need & Stress by Crop" if lang == "en" else "फसलों के अनुसार सिंचाई आवश्यकता और तनाव",
        hovermode="x unified",
        height=440,
        margin=dict(t=70, b=60, l=60, r=100),
        **CHART_TEMPLATE,
    )
    return fig


def plot_phenology_gantt(
    crop_names: List[str],
    sos_dates: List[str],
    peak_dates: List[str],
    eos_dates: List[str],
) -> go.Figure:
    """Gantt-style horizontal bar chart of crop growth stage timelines."""
    lang = st.session_state.get("lang", "en")
    
    if not (len(crop_names) == len(sos_dates) == len(peak_dates) == len(eos_dates)):
        raise ValueError("All input lists must have the same length.")

    stages_data: List[Dict[str, str]] = []

    # Map stage display names based on language
    stage_names_map = {
        "Vegetative": "Vegetative" if lang == "en" else "वानस्पतिक (बढ़ना)",
        "Reproductive": "Reproductive" if lang == "en" else "प्रजनन (फूल/बालियां)",
        "Grain Fill": "Grain Fill" if lang == "en" else "दाना भरना"
    }

    stage_colors: Dict[str, str] = {
        stage_names_map["Vegetative"]: "#4CAF50",
        stage_names_map["Reproductive"]: "#FFD600",
        stage_names_map["Grain Fill"]: "#FF8F00",
    }

    for crop, sos, peak, eos in zip(crop_names, sos_dates, peak_dates, eos_dates):
        if not all([sos, peak, eos]):
            logger.debug("Skipping %s: incomplete phenology dates.", crop)
            continue
        try:
            sos_ts = pd.Timestamp(sos)
            peak_ts = pd.Timestamp(peak)
            eos_ts = pd.Timestamp(eos)
        except Exception as exc:
            logger.warning("Could not parse dates for %s: %s", crop, exc)
            continue

        mid_ts = sos_ts + (peak_ts - sos_ts) / 2

        for start, end, stage_key in [
            (sos_ts, mid_ts, "Vegetative"),
            (mid_ts, peak_ts, "Reproductive"),
            (peak_ts, eos_ts, "Grain Fill"),
        ]:
            stages_data.append({
                "Task": translate_crop(crop, lang),
                "Start": str(start.date()),
                "Finish": str(end.date()),
                "Stage": stage_names_map[stage_key],
            })

    if not stages_data:
        fig = go.Figure()
        err_msg = "No phenology data available" if lang == "en" else "फसल चक्र की जानकारी उपलब्ध नहीं है"
        fig.update_layout(
            title=err_msg,
            height=250,
            **CHART_TEMPLATE,
        )
        return fig

    df = pd.DataFrame(stages_data)
    color_map = {s: stage_colors.get(s, "#9E9E9E") for s in df["Stage"].unique()}

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Stage",
        color_discrete_map=color_map,
        title="Crop Growth Stage Timeline" if lang == "en" else "फसल वृद्धि का समय चक्र (Timeline)",
        hover_data={"Start": True, "Finish": True, "Stage": True},
    )
    fig.update_yaxes(autorange="reversed")
    
    # Translate axes labels
    fig.update_xaxes(title_text="Date" if lang == "en" else "तारीख")
    fig.update_yaxes(title_text="Crop" if lang == "en" else "फसल")
    
    fig.update_layout(
        height=max(320, len(crop_names) * 65),
        margin=dict(t=70, b=60, l=100, r=40),
        **CHART_TEMPLATE,
    )
    return fig


def plot_stress_by_stage_bar(
    stage_names: List[str],
    stage_areas_pct: List[float],
    stage_stress_means: List[float],
) -> go.Figure:
    """Dual-axis bar + line chart of area coverage and mean stress per growth stage."""
    lang = st.session_state.get("lang", "en")
    
    if not stage_names:
        raise ValueError("stage_names must not be empty.")
    if len(stage_areas_pct) != len(stage_names) or len(stage_stress_means) != len(stage_names):
        raise ValueError("All input lists must have the same length.")

    stage_colors: Dict[str, str] = {
        "pre_sowing": "#ECEFF1",
        "establishment": "#A5D6A7",
        "vegetative": "#4CAF50",
        "reproductive": "#FFD600",
        "grain_fill": "#FF8F00",
        "maturity": "#BF360C",
    }
    
    stage_translations = {
        "pre_sowing": "Pre-sowing" if lang == "en" else "बुआई से पहले",
        "establishment": "Establishment" if lang == "en" else "अंकुरण/शुरुआत",
        "vegetative": "Vegetative" if lang == "en" else "बढ़ने की स्टेज",
        "reproductive": "Reproductive" if lang == "en" else "फूल/बालियां",
        "grain_fill": "Grain Fill" if lang == "en" else "दाना भरना",
        "maturity": "Maturity" if lang == "en" else "फसल पकाव"
    }
    
    bar_colors = [stage_colors.get(s.lower(), "#9E9E9E") for s in stage_names]
    nice_names = [stage_translations.get(s.lower(), s.replace("_", " ").title()) for s in stage_names]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_name = "Area (%)" if lang == "en" else "रकबा (%)"
    fig.add_trace(
        go.Bar(
            x=nice_names,
            y=stage_areas_pct,
            name=bar_name,
            marker_color=bar_colors,
            marker_line=dict(color="rgba(255,255,255,0.4)", width=1),
            opacity=0.88,
            text=[f"{v:.1f}%" for v in stage_areas_pct],
            textposition="inside",
            textfont=dict(size=12, color="#1B1B1B"),
            hovertemplate="<b>%{x}</b><br>Area: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    line_name = "Mean Stress" if lang == "en" else "औसत जल तनाव"
    fig.add_trace(
        go.Scatter(
            x=nice_names,
            y=stage_stress_means,
            name=line_name,
            mode="lines+markers",
            marker=dict(size=12, color="#D50000", symbol="circle",
                        line=dict(color="white", width=1.5)),
            line=dict(color="#D50000", width=2.5),
            hovertemplate="<b>%{x}</b><br>Stress: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Area (%)" if lang == "en" else "क्षेत्र प्रतिशत (%)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(
        title_text="Mean Stress Score (0-100)" if lang == "en" else "औसत तनाव स्कोर (0-100)",
        secondary_y=True,
        range=[0, 110],
        showgrid=False,
    )
    fig.update_layout(
        title="Stress by Growth Stage" if lang == "en" else "फसल चक्र के चरणों के अनुसार जल तनाव",
        height=400,
        margin=dict(t=70, b=60, l=60, r=80),
        **CHART_TEMPLATE,
    )
    return fig


def plot_et0_validation_scatter(
    observed: List[float],
    computed: List[float],
    r2: float,
    rmse: float,
) -> go.Figure:
    """Scatter plot comparing observed vs. Penman-Monteith computed ET0.

    Includes a 1:1 reference line, R-squared and RMSE annotations in the title,
    and colour-coded deviation from the 1:1 diagonal.

    Args:
        observed: Observed (station) ET0 values in mm/day.
        computed: FAO-56 PM computed ET0 values in mm/day.
        r2: Coefficient of determination (dimensionless, 0-1).
        rmse: Root mean squared error in mm/day.

    Returns:
        Fully configured plotly.graph_objects.Figure.

    Raises:
        ValueError: If *observed* and *computed* differ in length or are empty.
    """
    if not observed:
        raise ValueError("observed list must not be empty.")
    if len(observed) != len(computed):
        raise ValueError(
            f"observed ({len(observed)}) and computed ({len(computed)}) must have the same length."
        )

    residuals = [c - o for c, o in zip(computed, observed)]
    max_val = max(max(observed, default=0), max(computed, default=0)) * 1.1
    max_val = max(max_val, 0.1)

    fig = go.Figure()

    # Scatter coloured by residual
    fig.add_trace(go.Scatter(
        x=observed,
        y=computed,
        mode="markers",
        marker=dict(
            size=8,
            color=residuals,
            colorscale=[[0.0, "#D50000"], [0.5, "#FFFFFF"], [1.0, "#0091EA"]],
            cmid=0.0,
            showscale=True,
            colorbar=dict(title="Residual<br>(mm/d)", len=0.7),
            opacity=0.80,
            line=dict(color="rgba(0,0,0,0.15)", width=0.5),
        ),
        name="ET0 pairs",
        hovertemplate=(
            "Observed: %{x:.2f} mm/d<br>"
            "Computed: %{y:.2f} mm/d<br>"
            "Residual: %{marker.color:.3f} mm/d<extra></extra>"
        ),
    ))

    # 1:1 reference line
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color="#2E7D32", width=2, dash="dash"),
        name="1:1 Line",
        hoverinfo="skip",
    ))

    # +10% envelope
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val * 1.1],
        mode="lines",
        line=dict(color="#81C784", width=1, dash="dot"),
        name="+10%",
        hoverinfo="skip",
        showlegend=True,
    ))
    # -10% envelope
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val * 0.9],
        mode="lines",
        line=dict(color="#EF9A9A", width=1, dash="dot"),
        name="-10%",
        hoverinfo="skip",
        showlegend=True,
    ))

    fig.update_layout(
        title=f"ET0 Validation - R2 = {r2:.3f}  |  RMSE = {rmse:.2f} mm/d",
        xaxis_title="Observed ET0 (mm/day)",
        yaxis_title="Computed ET0 (mm/day)",
        xaxis_range=[0, max_val],
        yaxis_range=[0, max_val],
        height=440,
        margin=dict(t=70, b=70, l=70, r=80),
        **CHART_TEMPLATE,
    )
    return fig


def plot_shap_importance(
    feature_names: List[str],
    shap_values: List[float],
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of SHAP mean absolute feature importances."""
    lang = st.session_state.get("lang", "en")
    
    if len(feature_names) != len(shap_values):
        raise ValueError("feature_names and shap_values must have the same length.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    df = pd.DataFrame({"Feature": feature_names, "SHAP": shap_values})
    df = df.sort_values("SHAP", ascending=True).tail(top_n).reset_index(drop=True)
    df["Feature"] = df["Feature"].str.replace("_", " ").str.title()

    fig = go.Figure(go.Bar(
        x=df["SHAP"],
        y=df["Feature"],
        orientation="h",
        marker=dict(
            color=df["SHAP"],
            colorscale=[[0.0, "#C8E6C9"], [0.5, "#43A047"], [1.0, "#1B5E20"]],
            showscale=True,
            colorbar=dict(title="Mean |SHAP|", len=0.6),
            line=dict(color="rgba(255,255,255,0.3)", width=0.5),
        ),
        text=[f"{v:.4f}" for v in df["SHAP"]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))

    num_f = min(top_n, len(df))
    title_text = f"Top {num_f} Feature Importances (SHAP)" if lang == "en" else f"मुख्य {num_f} महत्वपूर्ण पैरामीटर्स (AI)"
    fig.update_layout(
        title=title_text,
        xaxis_title="Mean |SHAP Value|" if lang == "en" else "औसत प्रभाव (SHAP)",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        height=max(420, len(df) * 26),
        margin=dict(t=70, b=60, l=180, r=80),
        **CHART_TEMPLATE,
    )
    return fig


def plot_metric_cards(
    metrics: Dict[str, Any],
) -> go.Figure:
    """Render key performance indicator tiles as Plotly Indicator gauges.

    Displays four headline metrics side-by-side in a compact single-row
    grid suitable for dashboard headers.

    Args:
        metrics: Dictionary with any subset of keys:
            - total_area_ha: Total area analysed (hectares).
            - total_irr_volume_ML: Total irrigation volume (mega-litres).
            - pct_severe_stress: Percentage of area under severe stress.
            - mean_et0: Mean reference ET0 (mm/day).

    Returns:
        Fully configured plotly.graph_objects.Figure.
    """
    indicators: List[Tuple[str, float, str, str]] = [
        ("Area Analyzed", float(metrics.get("total_area_ha", 0.0)), "ha", "#00C853"),
        ("Irrigation Need", float(metrics.get("total_irr_volume_ML", 0.0)), "ML", "#0091EA"),
        ("% Area Stressed", float(metrics.get("pct_severe_stress", 0.0)), "%", "#D50000"),
        ("Mean ET0", float(metrics.get("mean_et0", 0.0)), "mm/d", "#FF8F00"),
    ]

    fig = go.Figure()

    for i, (label, value, suffix, color) in enumerate(indicators):
        fig.add_trace(go.Indicator(
            mode="number",
            value=value,
            number={
                "suffix": f" {suffix}",
                "font": {"size": 30, "color": color, "family": "Inter, sans-serif"},
                "valueformat": ",.1f",
            },
            title={
                "text": f"<b>{label}</b>",
                "font": {"size": 13, "color": "#37474F", "family": "Inter, sans-serif"},
            },
            domain={"row": 0, "column": i},
        ))

    fig.update_layout(
        grid={"rows": 1, "columns": 4, "pattern": "independent"},
        height=160,
        paper_bgcolor="rgba(241,248,233,0.6)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10, l=20, r=20),
        font=dict(family="Inter, Helvetica, sans-serif"),
    )
    return fig


def plot_soil_moisture_heatmap(
    dates: List[str],
    depths_cm: List[int],
    moisture_matrix: List[List[float]],
) -> go.Figure:
    """Render a time by depth soil moisture heatmap.

    Colour scale runs from dry (red) through moderate (yellow) to
    saturated (blue-green), matching agronomic conventions.

    Args:
        dates: Date strings on the time axis (columns).
        depths_cm: Soil depth intervals in cm (rows), e.g. [10, 30, 60, 90].
        moisture_matrix: 2-D array of shape (len(depths_cm), len(dates))
            with volumetric water content (m3/m3).

    Returns:
        Fully configured plotly.graph_objects.Figure.

    Raises:
        ValueError: If matrix dimensions are inconsistent with axis lists.
    """
    if not dates or not depths_cm:
        raise ValueError("dates and depths_cm must not be empty.")
    if len(moisture_matrix) != len(depths_cm):
        raise ValueError(
            f"moisture_matrix rows ({len(moisture_matrix)}) must equal len(depths_cm) ({len(depths_cm)})."
        )
    for i, row in enumerate(moisture_matrix):
        if len(row) != len(dates):
            raise ValueError(
                f"moisture_matrix row {i} has {len(row)} values; expected {len(dates)}."
            )

    z = np.array(moisture_matrix, dtype=float)
    depth_labels = [f"{d} cm" for d in depths_cm]

    fig = go.Figure(go.Heatmap(
        x=dates,
        y=depth_labels,
        z=z,
        colorscale=[
            [0.0, "#D50000"],
            [0.3, "#FFD600"],
            [0.6, "#4CAF50"],
            [1.0, "#0091EA"],
        ],
        zmin=0.0,
        zmax=0.5,
        colorbar=dict(title="VWC (m3/m3)", tickformat=".2f", len=0.7),
        hovertemplate=(
            "Date: %{x}<br>"
            "Depth: %{y}<br>"
            "VWC: %{z:.3f} m3/m3<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Soil Moisture Profile (Volumetric Water Content)",
        xaxis_title="Date",
        yaxis_title="Soil Depth",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(t=70, b=60, l=80, r=80),
        **CHART_TEMPLATE,
    )
    return fig


def plot_yield_forecast_gauge(
    predicted_yield: float,
    potential_yield: float,
    crop_name: str,
    unit: str = "t/ha",
) -> go.Figure:
    """Render a gauge chart comparing predicted vs. potential yield.

    The gauge needle points to the predicted yield; the track is colour-coded
    green (>80% attainment), yellow (50-80%), and red (<50%).

    Args:
        predicted_yield: ML-predicted yield in *unit*.
        potential_yield: Maximum achievable (potential) yield in *unit*.
        crop_name: Crop name for the chart title.
        unit: Yield unit string for display (default "t/ha").

    Returns:
        Fully configured plotly.graph_objects.Figure.

    Raises:
        ValueError: If *potential_yield* <= 0.
    """
    if potential_yield <= 0:
        raise ValueError("potential_yield must be > 0.")

    attainment_pct = min(predicted_yield / potential_yield * 100.0, 100.0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_yield,
        number={"suffix": f" {unit}", "font": {"size": 28, "color": "#1B5E20"}},
        delta={
            "reference": potential_yield,
            "relative": True,
            "valueformat": ".1%",
            "decreasing": {"color": "#D50000"},
            "increasing": {"color": "#00C853"},
        },
        title={
            "text": (
                f"<b>{crop_name.capitalize()} Yield Forecast</b><br>"
                f"<span style='font-size:13px;color:#555'>"
                f"Attainment: {attainment_pct:.1f}% of potential</span>"
            ),
            "font": {"size": 16, "color": "#2E7D32"},
        },
        gauge={
            "axis": {
                "range": [0, potential_yield * 1.05],
                "tickwidth": 1,
                "tickcolor": "#555",
                "ticksuffix": f" {unit}",
            },
            "bar": {"color": "#2E7D32", "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#C8E6C9",
            "steps": [
                {"range": [0, potential_yield * 0.5], "color": "#FFCDD2"},
                {"range": [potential_yield * 0.5, potential_yield * 0.8], "color": "#FFF9C4"},
                {"range": [potential_yield * 0.8, potential_yield * 1.05], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "#FF6D00", "width": 3},
                "thickness": 0.75,
                "value": predicted_yield,
            },
        },
    ))

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(248,255,248,0.95)",
        margin=dict(t=100, b=30, l=40, r=40),
        font=dict(family="Inter, Helvetica, sans-serif"),
    )
    return fig
