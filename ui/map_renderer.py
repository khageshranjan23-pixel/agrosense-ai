"""
AgroSense AI — Dynamic Folium Map Builder
Builds interactive, layer-toggling maps with satellite imagery base layer.
All styling driven by config colormaps — no hardcoded hex colors.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import folium
from folium import plugins
from folium.plugins import MeasureControl, MiniMap, MousePosition, Draw, LocateControl

logger = logging.getLogger(__name__)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROP_COLORS, STRESS_COLORS, IRRIGATION_COLORS


# ---------------------------------------------------------------------------
# Base Map
# ---------------------------------------------------------------------------


def create_base_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 12,
) -> folium.Map:
    """Create a folium map with dual base layers (satellite + OSM).

    Adds three tile providers (ESRI Satellite, OSM, Google Hybrid), a
    collapsible MiniMap, a live MousePosition readout, and a Measure tool so
    the user can calculate distances / areas directly on the map.

    Args:
        center_lat: Map center latitude in decimal degrees.
        center_lon: Map center longitude in decimal degrees.
        zoom_start: Initial zoom level (1-18).

    Returns:
        Configured ``folium.Map`` instance with base layers and controls
        attached.  Callers should add overlay layers then call
        :func:`finalize_map` before rendering.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,          # We add our own tile layers below
        control_scale=True,  # Scale bar in bottom-left
    )

    # ------------------------------------------------------------------
    # Base tile layers (radio-button — only one active at a time)
    # ------------------------------------------------------------------
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="ESRI World Imagery",
        name="Satellite (ESRI)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map (OSM)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Hybrid",
        overlay=False,
        control=True,
    ).add_to(m)

    # ------------------------------------------------------------------
    # UI controls
    # ------------------------------------------------------------------
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coords: ",
        lat_formatter="function(num) {return L.Util.formatNum(num, 5);}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5);}",
    ).add_to(m)

    MeasureControl(
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
        primary_area_unit="hectares",
        secondary_area_unit="sqmeters",
        position="topleft",
    ).add_to(m)

    LocateControl(
        position="topleft",
        drawCircle=True,
        follow=False,
        keepCurrentZoomLevel=False,
    ).add_to(m)

    logger.debug(
        "Base map created at (%.5f, %.5f) zoom=%d",
        center_lat,
        center_lon,
        zoom_start,
    )
    return m


# ---------------------------------------------------------------------------
# Raster overlay helper
# ---------------------------------------------------------------------------


def array_to_image_overlay(
    data: np.ndarray,
    bounds: List[List[float]],
    colormap_fn: Any,
    name: str,
    opacity: float = 0.7,
) -> folium.raster_layers.ImageOverlay:
    """Convert a 2-D numpy array to a folium ImageOverlay with a colormap.

    The array is min-max normalised to [0, 1] before passing through
    *colormap_fn*.  NaN pixels are rendered as fully transparent.

    Args:
        data: 2-D data array of shape (H, W).
        bounds: [[south, west], [north, east]] geographic bounding box.
        colormap_fn: Callable f(normalized: np.ndarray) -> np.ndarray
            where the output has shape (H, W, 4) with RGBA values in
            [0, 1].  Any matplotlib colormap instance is compatible.
        name: Layer name shown in the layer-control widget.
        opacity: Layer opacity, 0 (transparent) to 1 (opaque).

    Returns:
        A folium.raster_layers.ImageOverlay ready to be added to a
        folium.Map or folium.FeatureGroup.

    Raises:
        ValueError: If *data* is not a 2-D array.
    """
    from PIL import Image  # Imported here to keep top-level imports minimal

    if data.ndim != 2:
        raise ValueError(
            f"array_to_image_overlay expects a 2-D array, got shape {data.shape}"
        )

    # ------------------------------------------------------------------
    # Normalise to [0, 1] — NaN -> 0 (transparent after RGBA conversion)
    # ------------------------------------------------------------------
    d_min = float(np.nanmin(data))
    d_max = float(np.nanmax(data))

    if (d_max - d_min) < 1e-9:
        normalized = np.zeros_like(data, dtype=np.float64)
    else:
        normalized = (data.astype(np.float64) - d_min) / (d_max - d_min)

    nan_mask = np.isnan(data)
    normalized = np.nan_to_num(normalized, nan=0.0)

    # ------------------------------------------------------------------
    # Apply colormap -> RGBA [0, 1]
    # ------------------------------------------------------------------
    rgba_float = colormap_fn(normalized)  # (H, W, 4)
    rgba_float = np.asarray(rgba_float, dtype=np.float64)

    # Force NaN pixels fully transparent
    rgba_float[nan_mask, 3] = 0.0

    # Clip to valid range before converting to uint8
    rgba_float = np.clip(rgba_float, 0.0, 1.0)
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Encode as base64 PNG URL
    # ------------------------------------------------------------------
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    url = f"data:image/png;base64,{encoded}"

    return folium.raster_layers.ImageOverlay(
        image=url,
        bounds=bounds,
        opacity=opacity,
        name=name,
        cross_origin=False,
        zindex=1,
    )


# ---------------------------------------------------------------------------
# Thematic overlay layers
# ---------------------------------------------------------------------------


def add_crop_type_layer(
    m: folium.Map,
    crop_map: np.ndarray,
    bounds: List[List[float]],
    crop_classes: List[str],
) -> folium.Map:
    """Add a discrete crop-type classification overlay to the map.

    Each unique integer code in *crop_map* (0, 1, ... N-1) is assigned the
    colour defined in config.CROP_COLORS for the corresponding class
    name in *crop_classes*.  An HTML legend is injected into the map.

    Args:
        m: Target folium map.
        crop_map: 2-D integer array (H, W) with class indices [0, N).
        bounds: [[south, west], [north, east]].
        crop_classes: Ordered list of class names matching index 0..N-1.

    Returns:
        The updated map with the crop-type overlay and legend appended.
    """
    import matplotlib.colors as mcolors

    n_classes = len(crop_classes)
    if n_classes == 0:
        logger.warning("add_crop_type_layer called with empty crop_classes list")
        return m

    colors_list: List[str] = [
        CROP_COLORS.get(cls.lower(), CROP_COLORS.get("other", "#808080"))
        for cls in crop_classes
    ]

    # Pre-convert to RGBA once so the closure is O(N) not O(H*W*N)
    rgba_table = np.array(
        [mcolors.to_rgba(c) for c in colors_list], dtype=np.float64
    )  # (N, 4)

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        indices = np.round(normalized * (n_classes - 1)).astype(int)
        indices = np.clip(indices, 0, n_classes - 1)
        return rgba_table[indices]  # vectorised index -> (H, W, 4)

    overlay = array_to_image_overlay(
        crop_map.astype(np.float64),
        bounds,
        colormap_fn,
        name="Crop Types",
        opacity=0.75,
    )
    overlay.add_to(m)

    legend_html = _build_legend_html(
        "Crop Types", list(zip(crop_classes, colors_list)), left_offset="10px"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    logger.debug("Crop-type layer added (%d classes)", n_classes)
    return m


def add_stress_layer(
    m: folium.Map,
    stress_class: np.ndarray,
    bounds: List[List[float]],
) -> folium.Map:
    """Add a 4-class moisture-stress classification overlay to the map.

    Stress codes 0-3 map to: No Stress, Mild, Moderate, Severe.
    Colours are read from config.STRESS_COLORS.

    Args:
        m: Target folium map.
        stress_class: 2-D integer array (H, W) with values 0-3.
        bounds: [[south, west], [north, east]].

    Returns:
        The updated map with the stress overlay and legend appended.
    """
    import matplotlib.colors as mcolors

    stress_color_values: List[str] = [
        STRESS_COLORS["no_stress"],
        STRESS_COLORS["mild_stress"],
        STRESS_COLORS["moderate_stress"],
        STRESS_COLORS["severe_stress"],
    ]
    stress_labels = ["No Stress", "Mild Stress", "Moderate Stress", "Severe Stress"]

    rgba_table = np.array(
        [mcolors.to_rgba(c) for c in stress_color_values], dtype=np.float64
    )

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        codes = np.round(normalized * 3).astype(int)
        codes = np.clip(codes, 0, 3)
        return rgba_table[codes]

    overlay = array_to_image_overlay(
        stress_class.astype(np.float64),
        bounds,
        colormap_fn,
        name="Moisture Stress",
        opacity=0.70,
    )
    overlay.add_to(m)

    legend_html = _build_legend_html(
        "Moisture Stress",
        list(zip(stress_labels, stress_color_values)),
        left_offset="10px",
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    logger.debug("Stress layer added")
    return m


def add_irrigation_advisory_layer(
    m: folium.Map,
    advisory_code: np.ndarray,
    bounds: List[List[float]],
) -> folium.Map:
    """Add a 4-class irrigation advisory overlay to the map.

    Advisory codes 0-3 map to: No Irrigation, Irrigate in 3 Days,
    Irrigate Immediately, Critical Alert.  Colours are drawn from
    config.IRRIGATION_COLORS.

    Args:
        m: Target folium map.
        advisory_code: 2-D integer array (H, W) with values 0-3.
        bounds: [[south, west], [north, east]].

    Returns:
        The updated map with the advisory overlay and legend appended.
    """
    import matplotlib.colors as mcolors

    adv_color_values: List[str] = [
        IRRIGATION_COLORS["no_irrigation"],
        IRRIGATION_COLORS["irrigate_3_days"],
        IRRIGATION_COLORS["irrigate_immediately"],
        IRRIGATION_COLORS["critical_alert"],
    ]
    adv_labels = [
        "No Irrigation",
        "Irrigate in 3 Days",
        "Irrigate Now!",
        "Critical Alert",
    ]

    rgba_table = np.array(
        [mcolors.to_rgba(c) for c in adv_color_values], dtype=np.float64
    )

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        codes = np.round(normalized * 3).astype(int)
        codes = np.clip(codes, 0, 3)
        return rgba_table[codes]

    overlay = array_to_image_overlay(
        advisory_code.astype(np.float64),
        bounds,
        colormap_fn,
        name="Irrigation Advisory",
        opacity=0.75,
    )
    overlay.add_to(m)

    legend_html = _build_legend_html(
        "Irrigation Advisory",
        list(zip(adv_labels, adv_color_values)),
        left_offset="10px",
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    logger.debug("Irrigation advisory layer added")
    return m


def add_water_deficit_layer(
    m: folium.Map,
    irr_need_mm: np.ndarray,
    bounds: List[List[float]],
) -> folium.Map:
    """Add a continuous water-deficit (mm) overlay using the RdYlGn_r colormap.

    Red pixels indicate high irrigation deficit; green pixels indicate
    adequately watered areas.

    Args:
        m: Target folium map.
        irr_need_mm: 2-D float array (H, W) of irrigation need in mm.
        bounds: [[south, west], [north, east]].

    Returns:
        The updated map with the water-deficit overlay appended.
    """
    import matplotlib.cm as cm

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        return cm.RdYlGn_r(normalized)  # type: ignore[return-value]

    overlay = array_to_image_overlay(
        irr_need_mm,
        bounds,
        colormap_fn,
        name="Water Deficit (mm)",
        opacity=0.70,
    )
    overlay.add_to(m)
    logger.debug("Water-deficit layer added")
    return m


def add_ndvi_layer(
    m: folium.Map,
    ndvi: np.ndarray,
    bounds: List[List[float]],
) -> folium.Map:
    """Add an NDVI heatmap overlay using the RdYlGn colormap.

    Low NDVI (stressed / bare soil) -> red; high NDVI (dense vegetation) ->
    green.

    Args:
        m: Target folium map.
        ndvi: 2-D float array (H, W) with NDVI values typically in [-1, 1].
        bounds: [[south, west], [north, east]].

    Returns:
        The updated map with the NDVI overlay appended.
    """
    import matplotlib.cm as cm

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        return cm.RdYlGn(normalized)  # type: ignore[return-value]

    overlay = array_to_image_overlay(
        ndvi,
        bounds,
        colormap_fn,
        name="NDVI",
        opacity=0.70,
    )
    overlay.add_to(m)
    logger.debug("NDVI layer added")
    return m


def add_vhi_layer(
    m: folium.Map,
    vhi: np.ndarray,
    bounds: List[List[float]],
) -> folium.Map:
    """Add a Vegetation Health Index (VHI) overlay.

    VHI combines NDVI and Land Surface Temperature anomalies. Values range
    from 0 (extreme drought) to 100 (optimal health).

    Args:
        m: Target folium map.
        vhi: 2-D float array (H, W) with VHI values in [0, 100].
        bounds: [[south, west], [north, east]].

    Returns:
        The updated map with the VHI overlay appended.
    """
    import matplotlib.cm as cm

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        return cm.RdYlGn(normalized)  # type: ignore[return-value]

    overlay = array_to_image_overlay(
        vhi,
        bounds,
        colormap_fn,
        name="VHI",
        opacity=0.68,
    )
    overlay.add_to(m)
    logger.debug("VHI layer added")
    return m


def add_et0_layer(
    m: folium.Map,
    et0_daily: np.ndarray,
    bounds: List[List[float]],
) -> folium.Map:
    """Add a daily reference evapotranspiration (ET0) overlay.

    Uses a YlOrRd colormap — brighter orange/red indicates higher atmospheric
    demand for water.

    Args:
        m: Target folium map.
        et0_daily: 2-D float array (H, W) of ET0 in mm/day.
        bounds: [[south, west], [north, east]].

    Returns:
        The updated map with the ET0 overlay appended.
    """
    import matplotlib.cm as cm

    def colormap_fn(normalized: np.ndarray) -> np.ndarray:
        return cm.YlOrRd(normalized)  # type: ignore[return-value]

    overlay = array_to_image_overlay(
        et0_daily,
        bounds,
        colormap_fn,
        name="ET0 (mm/day)",
        opacity=0.65,
    )
    overlay.add_to(m)
    logger.debug("ET0 layer added")
    return m


# ---------------------------------------------------------------------------
# Boundary / vector layers
# ---------------------------------------------------------------------------


def add_study_area_boundary(
    m: folium.Map,
    geojson: Dict[str, Any],
    area_name: str = "Study Area",
) -> folium.Map:
    """Add a study-area boundary polygon to the map as a dashed GeoJSON overlay.

    The boundary is styled with a cyan dashed stroke and no fill so it does
    not obscure the raster layers underneath.

    Args:
        m: Target folium map.
        geojson: GeoJSON dict for the boundary geometry (Polygon or
            MultiPolygon).  Accepts either a Feature or a bare Geometry.
        area_name: Human-readable label for the tooltip and layer control.

    Returns:
        The updated map with the boundary overlay appended.
    """
    folium.GeoJson(
        geojson,
        name=f"{area_name} Boundary",
        style_function=lambda _: {
            "fillColor": "none",
            "color": "#00E5FF",
            "weight": 2.5,
            "dashArray": "6 4",
            "fillOpacity": 0.0,
        },
        highlight_function=lambda _: {
            "color": "#FFFFFF",
            "weight": 3.5,
        },
        tooltip=area_name,
    ).add_to(m)
    logger.debug("Study-area boundary added: %s", area_name)
    return m


def add_field_markers(
    m: folium.Map,
    field_records: List[Dict[str, Any]],
) -> folium.Map:
    """Add individual field-level popup markers to the map.

    Each record must contain ``lat``, ``lon``, ``name``, and optionally
    ``advisory``, ``crop``, ``area_ha``, and ``stress_score`` keys.

    Args:
        m: Target folium map.
        field_records: List of field metadata dicts.

    Returns:
        The updated map with markers attached to a named FeatureGroup.
    """
    fg = folium.FeatureGroup(name="Field Markers", show=True)

    for rec in field_records:
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is None or lon is None:
            logger.warning("Skipping field record without lat/lon: %s", rec)
            continue

        popup_html = (
            f"<b>{rec.get('name', 'Unknown Field')}</b><br>"
            f"Crop: {rec.get('crop', 'N/A')}<br>"
            f"Area: {rec.get('area_ha', 'N/A')} ha<br>"
            f"Stress: {rec.get('stress_score', 'N/A')}/100<br>"
            f"Advisory: {rec.get('advisory', 'N/A')}"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=rec.get("name", "Field"),
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(fg)

    fg.add_to(m)
    logger.debug("Added %d field markers", len(field_records))
    return m


# ---------------------------------------------------------------------------
# Map finalisation
# ---------------------------------------------------------------------------


def finalize_map(m: folium.Map) -> folium.Map:
    """Attach layer control and draw tool then return the completed map.

    This should be called *last*, after all overlays have been added, so
    that the LayerControl lists every layer.

    Args:
        m: Fully-overlaid folium map.

    Returns:
        The map with LayerControl and Draw appended.
    """
    Draw(
        export=True,
        draw_options={
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "polyline": False,
            "marker": True,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # LayerControl must be added after all layers
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    logger.debug("Map finalised — layer control and draw tool added")
    return m


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def get_bounds_from_geometry(geojson: Dict[str, Any]) -> List[List[float]]:
    """Extract [[south, west], [north, east]] bounds from a GeoJSON dict.

    Handles GeoJSON Feature wrappers and nested MultiPolygon coordinate
    arrays by recursively flattening until [lon, lat] pairs are reached.

    Args:
        geojson: GeoJSON dict — may be a Feature, FeatureCollection
            Geometry, Polygon, or MultiPolygon.

    Returns:
        [[min_lat, min_lon], [max_lat, max_lon]].  Falls back to global
        extents if no coordinates can be extracted.
    """
    # Unwrap Feature -> Geometry
    geometry = geojson.get("geometry", geojson)
    coords_root = geometry.get("coordinates", [])

    if not coords_root:
        logger.warning(
            "get_bounds_from_geometry: no coordinates found, returning global extent"
        )
        return [[-90.0, -180.0], [90.0, 180.0]]

    flat_coords: List[List[float]] = []

    def _flatten(obj: Any) -> None:
        """Recursively flatten nested coordinate arrays to [lon, lat] pairs."""
        if not obj:
            return
        if isinstance(obj[0], (int, float)):
            flat_coords.append(obj)  # type: ignore[arg-type]
        else:
            for child in obj:
                _flatten(child)

    _flatten(coords_root)

    if not flat_coords:
        logger.warning(
            "get_bounds_from_geometry: coordinate flattening yielded nothing"
        )
        return [[-90.0, -180.0], [90.0, 180.0]]

    lons = [c[0] for c in flat_coords]
    lats = [c[1] for c in flat_coords]

    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def get_center_from_bounds(bounds: List[List[float]]) -> Tuple[float, float]:
    """Compute the geographic centre from a bounding box.

    Args:
        bounds: [[south, west], [north, east]].

    Returns:
        (center_lat, center_lon) tuple.
    """
    center_lat = (bounds[0][0] + bounds[1][0]) / 2.0
    center_lon = (bounds[0][1] + bounds[1][1]) / 2.0
    return center_lat, center_lon


def compute_zoom_for_bounds(
    bounds: List[List[float]],
    map_width_px: int = 800,
    map_height_px: int = 600,
) -> int:
    """Estimate an appropriate starting zoom level for given bounds.

    Uses the Mercator tile math to find the largest integer zoom where the
    bounding box fits within map_width_px x map_height_px.

    Args:
        bounds: [[south, west], [north, east]].
        map_width_px: Display width of the map widget in pixels.
        map_height_px: Display height of the map widget in pixels.

    Returns:
        Recommended zoom level, clamped to [1, 18].
    """
    import math

    south, west = bounds[0]
    north, east = bounds[1]

    lat_span = abs(north - south)
    lon_span = abs(east - west)

    if lat_span < 1e-9 or lon_span < 1e-9:
        return 15  # Tiny area — zoom in

    lat_zoom = math.log2(180.0 * map_height_px / (256.0 * lat_span))
    lon_zoom = math.log2(360.0 * map_width_px / (256.0 * lon_span))

    zoom = int(min(lat_zoom, lon_zoom))
    return max(1, min(18, zoom))


# ---------------------------------------------------------------------------
# Legend builder
# ---------------------------------------------------------------------------


def _build_legend_html(
    title: str,
    items: List[Tuple[str, str]],
    left_offset: str = "10px",
    bottom_offset: str = "30px",
) -> str:
    """Render an HTML legend div for injection into a folium map.

    Args:
        title: Legend heading text.
        items: List of (label, hex_color) pairs.
        left_offset: CSS left value positioning the legend.
        bottom_offset: CSS bottom value positioning the legend.

    Returns:
        Raw HTML string suitable for wrapping in folium.Element.
    """
    rows = "".join(
        f"""<div style="display:flex;align-items:center;margin:3px 0;">
                <div style="width:16px;height:16px;background:{color};
                    border-radius:3px;margin-right:8px;
                    border:1px solid rgba(0,0,0,0.3);flex-shrink:0;"></div>
                <span style="font-size:11px;color:#333;">{label}</span>
            </div>"""
        for label, color in items
    )

    return f"""
    <div style="
        position: fixed;
        bottom: {bottom_offset};
        left: {left_offset};
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 10px 14px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        font-family: 'Segoe UI', Arial, sans-serif;
        min-width: 160px;
        pointer-events: none;
    ">
        <b style="font-size:12px;display:block;margin-bottom:6px;color:#111;">{title}</b>
        {rows}
    </div>
    """
