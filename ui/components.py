# -*- coding: utf-8 -*-
"""
AgroSense AI — UI Component Library
Provides premium, styled HTML/CSS component injections for the dashboard.
"""
from typing import Dict, List, Any
import streamlit as st
from translations import TRANSLATIONS

def load_custom_css(css_file_path: str = "assets/styles.css"):
    """Load the custom CSS stylesheet and inject it into the streamlit page."""
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

def get_txt(key: str, lang: str) -> str:
    """Safely retrieve translation value for key and language."""
    try:
        return TRANSLATIONS[key][lang]
    except KeyError:
        return key

def render_hero_banner(lang: str):
    """Render the premium full-width welcoming banner."""
    title = get_txt("app_title", lang)
    sub = get_txt("app_subtitle", lang)
    tagline = get_txt("tagline", lang)
    powered = get_txt("powered_by", lang)
    
    badge_free = get_txt("trust_free", lang)
    badge_indian = get_txt("trust_indian", lang)
    badge_speed = get_txt("trust_speed", lang)
    badge_simple = get_txt("trust_simple", lang)
    
    html = f"""
    <div class="agro-hero">
        <div class="agro-badge" style="background: rgba(255,255,255,0.2); margin-bottom: 12px; font-size: 0.9rem;">
            {powered}
        </div>
        <h1 class="agro-hero-title">{title}</h1>
        <p style="font-size: 1.6rem; font-weight: 600; margin: 4px 0 16px 0; color: #E8F5E9;">{sub}</p>
        <p class="agro-hero-sub">{tagline}</p>
        <div class="agro-badges-row">
            <span class="agro-badge">{badge_indian}</span>
            <span class="agro-badge">{badge_free}</span>
            <span class="agro-badge">{badge_speed}</span>
            <span class="agro-badge">{badge_simple}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(title: str, subtitle: str = ""):
    """Render a styled section header with optional subtitle."""
    html = f"""
    <div class="section-header">
        <h2>{title}</h2>
    </div>
    """
    if subtitle:
        html += f'<div class="bilingual-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_step_indicator(current_step: int, total_steps: int = 5):
    """Render the visual step indicator for the wizard."""
    nodes = []
    for i in range(1, total_steps + 1):
        if i < current_step:
            nodes.append(f'<div class="wizard-step-node completed">{i}</div>')
        elif i == current_step:
            nodes.append(f'<div class="wizard-step-node active">{i}</div>')
        else:
            nodes.append(f'<div class="wizard-step-node">{i}</div>')
            
    steps_html = "".join(nodes)
    html = f"""
    <div class="wizard-steps">
        {steps_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_how_it_works(lang: str):
    """Render the 3-step cards explainer."""
    title = get_txt("how_it_works_title", lang)
    s1_t = get_txt("step_1_title", lang)
    s1_d = get_txt("step_1_desc", lang)
    s2_t = get_txt("step_2_title", lang)
    s2_d = get_txt("step_2_desc", lang)
    s3_t = get_txt("step_3_title", lang)
    s3_d = get_txt("step_3_desc", lang)
    
    html = f"""
    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; margin: 32px 0 16px 0; color: #123E1C; text-align: center;">{title}</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px;">
        <div style="background: #FFFFFF; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #EBF1EA; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📍</div>
            <h4 style="margin: 0 0 8px 0; font-weight: 700; font-size: 1.15rem; color: #1A301E;">{s1_t}</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #5C7060; line-height: 1.4;">{s1_d}</p>
        </div>
        <div style="background: #FFFFFF; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #EBF1EA; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🛰️</div>
            <h4 style="margin: 0 0 8px 0; font-weight: 700; font-size: 1.15rem; color: #1A301E;">{s2_t}</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #5C7060; line-height: 1.4;">{s2_d}</p>
        </div>
        <div style="background: #FFFFFF; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #EBF1EA; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📋</div>
            <h4 style="margin: 0 0 8px 0; font-weight: 700; font-size: 1.15rem; color: #1A301E;">{s3_t}</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #5C7060; line-height: 1.4;">{s3_d}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, unit: str, icon: str, category: str, footer_text: str) -> str:
    """Generate HTML string for a premium metric card."""
    return f"""
    <div class="metric-card {category}">
        <div class="metric-header">
            <span class="metric-label">{label}</span>
            <span class="metric-icon">{icon}</span>
        </div>
        <div class="metric-value-container">
            <span class="metric-value">{value}</span>
            <span class="metric-unit">{unit}</span>
        </div>
        <div class="metric-footer">
            <span>{footer_text}</span>
        </div>
    </div>
    """

def render_farmer_alert(status_type: str, title: str, description: str):
    """Render a bold, simple farmer-friendly alert block."""
    # status_type: danger, warning, success, info
    icon_map = {
        "danger": "🚨",
        "warning": "⚠️",
        "success": "✅",
        "info": "💡"
    }
    icon = icon_map.get(status_type, "💡")
    html = f"""
    <div class="farmer-alert {status_type}">
        <div class="alert-icon">{icon}</div>
        <div class="alert-content">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_loading_screen(lang: str, fact: str):
    """Render the animated loading screen for GEE processing."""
    wait_title = get_txt("loading_title", lang)
    wait_subtitle = get_txt("loading_subtitle", lang)
    fact_label = get_txt("loading_fact_label", lang)
    
    html = f"""
    <div class="satellite-loading">
        <div class="satellite-icon-container">
            🛰️
            <div class="satellite-beams"></div>
        </div>
        <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; margin: 0 0 8px 0; color: #123E1C;">{wait_title}</h3>
        <p style="font-size: 1.05rem; color: #5C7060; margin: 0 0 24px 0;">{wait_subtitle}</p>
        
        <div class="loading-fact-card">
            <strong style="color: #1B5E20; display: block; margin-bottom: 6px; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">{fact_label}</strong>
            "{fact}"
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
