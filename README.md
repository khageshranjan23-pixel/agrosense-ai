# 🌱 AgroSense AI — Precision Agriculture Intelligence System

> **AI-Driven Automated Crop Type, Moisture Stress Detection and Irrigation Advisory Across Growth Stages Using Optical & Microwave Satellite Data**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit)](https://streamlit.io)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-0.1.390-4285F4?logo=google)](https://earthengine.google.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Problem Statement 6 — ISRO/SAC Hackathon

This solution addresses **AI-driven automated methodology for crop type identification, stage-wise phenological mapping, moisture stress detection, and 8-day crop water deficit estimation** using multi-source satellite data fusion.

### Key Capabilities
| Feature | Technology |
|---|---|
| Crop Type Classification | XGBoost + RF + LightGBM stacked ensemble |
| Phenological Stage Detection | Dynamic Savitzky-Golay + NDVI normalization |
| Moisture Stress Assessment | VCI + TCI + VHI + CWSI + SAR-SMI (PCA fusion) |
| Reference ET₀ Computation | Penman-Monteith from ERA5 (all variables computed) |
| Irrigation Advisory | FAO-56 soil water balance, pixel-specific |
| Optical Data | Sentinel-2 SR Harmonized (10m, 10 bands) |
| SAR Data | Sentinel-1 GRD (IW, VV+VH, Refined Lee filter) |
| Climate Data | ERA5-LAND + CHIRPS daily rainfall |
| Terrain | SRTM DEM (slope, aspect, runoff) |
| Soil Parameters | ISRIC SoilGrids (field capacity, wilting point) |

---

## 🏗️ Architecture

```
agrosense-ai/
├── app.py                      # Streamlit entry point (colorful dashboard)
├── config.py                   # All configurable parameters (no hardcoding)
├── core/
│   ├── gee_auth.py             # GEE authentication (service account + fallback)
│   ├── data_ingestion.py       # Sentinel-1/2, ERA5, CHIRPS, SRTM ingestion
│   ├── preprocessing.py        # Cloud masking, speckle filtering, compositing
│   ├── feature_engineering.py  # 600+ spectral + SAR + temporal features
│   ├── phenology.py            # Dynamic SOS/PEAK/EOS detection
│   ├── crop_classifier.py      # XGBoost+RF+LightGBM stacking + Optuna
│   ├── stress_detector.py      # VCI, TCI, VHI, CWSI, SAR-SMI + PCA fusion
│   ├── water_balance.py        # FAO-56 ET₀, Kc, soil water balance
│   ├── advisory_engine.py      # 8-day irrigation advisory pipeline
│   └── validation.py           # Accuracy metrics + bootstrap CIs
├── ui/
│   ├── map_renderer.py         # Folium interactive map builder
│   ├── charts.py               # All Plotly visualizations
│   ├── sidebar.py              # Dynamic sidebar controls
│   └── export.py               # GeoTIFF + CSV + PDF export
├── data/
│   ├── kc_database.json        # FAO-56 Kc for 80+ crops
│   ├── stage_weights.json      # Growth stage stress weights
│   └── wris_commands.json      # 50 Indian canal command areas
└── .streamlit/
    ├── config.toml             # Theme + server config
    └── secrets.toml.example    # GEE service account template
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-team/agrosense-ai.git
cd agrosense-ai
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure GEE (Optional — system works in demo mode without it)

```bash
# Option A: Interactive auth (local development)
earthengine authenticate

# Option B: Service account (Streamlit Cloud deployment)
# Copy .streamlit/secrets.toml.example → .streamlit/secrets.toml
# Paste your GEE service account JSON
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Fork this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set main file to `app.py`
4. Add GEE service account in **Secrets**:
   ```toml
   [gee_service_account]
   type = "service_account"
   project_id = "..."
   # ... full service account JSON
   ```
5. Deploy!

---

## 📡 Data Sources

| Dataset | Collection ID | Resolution | Use |
|---|---|---|---|
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | 10m | Optical indices |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 10m | SAR backscatter |
| ERA5-LAND | `ECMWF/ERA5_LAND/DAILY_AGGR` | 9km | ET₀, climate |
| CHIRPS | `UCSB-CHG/CHIRPS/DAILY` | 5.5km | Rainfall |
| SRTM DEM | `USGS/SRTMGL1_003` | 30m | Terrain |
| MODIS LST | `MODIS/061/MOD11A1` | 1km | Land temperature |

---

## 🧮 Scientific Methods

### Spectral Indices
- **NDVI** = (B8−B4)/(B8+B4) — vegetation density
- **EVI** = 2.5×(B8−B4)/(B8+6B4−7.5B2+1) — soil/atmosphere corrected
- **NDWI** = (B3−B8)/(B3+B8) — open water
- **LSWI** = (B8−B11)/(B8+B11) — leaf water content
- **NDRE** = (B8A−B5)/(B8A+B5) — red-edge, stress-sensitive
- **SAVI** = ((B8−B4)/(B8+B4+0.5))×1.5 — soil-adjusted
- **MTCI** = (B8A−B5)/(B5−B4) — chlorophyll content
- **CIre** = (B8A/B5)−1 — chlorophyll red-edge

### Stress Indices
- **VCI** = 100×(NDVIcurrent−NDVImin)/(NDVImax−NDVImin) [5-year baseline]
- **TCI** = 100×(LSTmax−LSTcurrent)/(LSTmax−LSTmin)
- **VHI** = α×VCI + (1−α)×TCI [α user-adjustable]
- **CWSI** = 1 − ETa/ET₀ [ERA5 actual ET vs Penman-Monteith]
- **SAR-SMI** = normalize(VHcurrent − VH5yearmean)
- **Combined** = PC1 of all 5 indices [PCA, no arbitrary weights]

### FAO-56 Penman-Monteith ET₀
```
ET₀ = [0.408·Δ·(Rn−G) + γ·(900/(T+273))·u₂·(es−ea)]
      / [Δ + γ·(1 + 0.34·u₂)]
```
All variables computed from ERA5 — no lookup tables.

### Crop Coefficient (Dynamic)
Kc interpolated per pixel based on days-since-planting (from phenology engine):
- `Kc_ini → Kc_mid` during development stage
- `Kc_mid` during mid-season
- `Kc_mid → Kc_end` during late season
- Values from FAO-56 Table 12 via `kc_database.json`

---

## 🤖 ML Pipeline

```
Ground Truth Labels
        │
        ▼
Feature Engineering (600+ features)
   ├── NDVI, EVI, NDWI, LSWI, NDRE, SAVI, MTCI, CIre, LAI
   ├── VV, VH, CR, RVI_SAR, GLCM texture (4 metrics)
   └── Temporal: mean, std, slope, max, AUC, green-up rate, senescence rate
        │
        ▼
SHAP Feature Selection (top 80)
        │
        ▼
Stacked Ensemble
   ├── XGBoost (Optuna-tuned, 50 trials)
   ├── Random Forest (300 trees)
   └── LightGBM (Optuna-tuned)
        │
        ▼
Meta-Learner: Logistic Regression on base predictions
        │
        ▼
Crop Type Map + Confidence Map + Uncertainty Mask
```

---

## 📊 Expected Outputs

| Output | Format | Description |
|---|---|---|
| Crop Type Map | GeoTIFF + Map | Per-pixel classification + confidence |
| Stress Class Map | GeoTIFF + Map | No/Mild/Moderate/Severe stress |
| Irrigation Advisory | GeoTIFF + Map | 4-class urgency |
| Water Deficit | GeoTIFF | Soil depletion in mm |
| NDVI Time Series | Plotly chart | Phenology curve + stage markers |
| Advisory CSV | CSV | Block-level advisory summary |
| PDF Report | PDF | Printable farmer advisory |

---

## 📦 Demo Mode

When GEE is not connected, AgroSense AI automatically runs in **demo mode** using synthetic data that faithfully replicates real satellite data patterns. This allows full demonstration of the dashboard during the hackathon without requiring GEE credentials.

---

## 📚 References

1. Allen, R.G. et al. (1998). *FAO-56: Crop Evapotranspiration*. FAO Irrigation Paper 56.
2. Doorenbos, J. & Kassam, A.H. (1979). *Yield Response to Water*. FAO Paper 33.
3. Kogan, F.N. (1995). Application of Vegetation Index and Brightness Temperature for Drought Detection. *Advances in Space Research*, 15(11), 91–100.
4. Drusch, M. et al. (2012). Sentinel-2: ESA's Optical High-Resolution Mission. *Remote Sensing of Environment*, 120, 25–36.
5. Torres, R. et al. (2012). GMES Sentinel-1 Mission. *Remote Sensing of Environment*, 120, 9–24.

---

## 👥 Team

Built for **ISRO/SAC Hackathon — Problem Statement 6**  
*AI-Driven Crop Type, Moisture Stress Detection and Irrigation Advisory Using Moderate Resolution Spectral Signatures*

---

*AgroSense AI — Empowering Indian agriculture with satellite intelligence* 🇮🇳
