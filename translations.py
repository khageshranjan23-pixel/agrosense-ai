# -*- coding: utf-8 -*-
"""
AgroSense AI — Bilingual Translation Database
Provides clean dictionary lookup for English and friendly conversational Hindi.
"""
from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Global / App Identity
    "app_title": {
        "en": "AgroSense AI",
        "hi": "एग्रोसेंस AI"
    },
    "app_subtitle": {
        "en": "Your Crop's Digital Doctor",
        "hi": "आपकी फसल का डिजिटल डॉक्टर 🩺"
    },
    "powered_by": {
        "en": "🛰️ Powered by Sentinel-1 & Sentinel-2 Satellites",
        "hi": "🛰️ स्पेस से सीधे: सेंटिनल-1 और सेंटिनल-2 सैटेलाइट द्वारा संचालित"
    },
    "tagline": {
        "en": "Satellite se seedha aapke khet ki puri jaankari — kaun si fasal hai, stress hai ya nahi, aur kab paani dena hai — sab kuch ek jagah",
        "hi": "सैटेलाइट से सीधा आपके खेत की पूरी जानकारी — कौन सी फसल है, पानी की कमी है या नहीं, और कब पानी देना है — सब कुछ एक जगह"
    },
    "btn_start_analysis": {
        "en": "🚀 Analyse My Field",
        "hi": "🚀 अपना खेत एनालाइज करें"
    },
    
    # Trust Indicators
    "trust_free": {
        "en": "📡 Free Satellite Data",
        "hi": "📡 मुफ्त सैटेलाइट डेटा"
    },
    "trust_indian": {
        "en": "🇮🇳 Made for Indian Farmers",
        "hi": "🇮🇳 भारतीय किसानों के लिए निर्मित"
    },
    "trust_speed": {
        "en": "⚡ Results in Minutes",
        "hi": "⚡ कुछ ही मिनटों में परिणाम"
    },
    "trust_simple": {
        "en": "🎯 No Technical Knowledge Needed",
        "hi": "🎯 किसी तकनीकी ज्ञान की आवश्यकता नहीं"
    },

    # How it works
    "how_it_works_title": {
        "en": "🤔 How Does This Work?",
        "hi": "🤔 यह कैसे काम करता है?"
    },
    "step_1_title": {
        "en": "📍 Show your field",
        "hi": "📍 अपना खेत दिखाओ"
    },
    "step_1_desc": {
        "en": "Write your village name or click on your field boundary on the interactive map.",
        "hi": "अपने गाँव का नाम चुनें या सीधे मैप पर अपने खेत पर क्लिक करके बाउंड्री बनाएं।"
    },
    "step_2_title": {
        "en": "🛰️ Satellite looks from space",
        "hi": "🛰️ सैटेलाइट देखेगा"
    },
    "step_2_desc": {
        "en": "Our system fetches space imagery to analyse your crop vigor and soil moisture.",
        "hi": "हमारा सिस्टम अंतरिक्ष से आपके खेत की लाइव फोटो लेकर मिट्टी की नमी की जांच करेगा।"
    },
    "step_3_title": {
        "en": "📋 Get direct answers",
        "hi": "📋 सीधा जवाब मिलेगा"
    },
    "step_3_desc": {
        "en": "Get clear advice: how the crop is doing, if it needs water, and what action to take.",
        "hi": "आपको साफ़-साफ़ सलाह मिलेगी: फसल कैसी है, पानी चाहिए या नहीं, और आगे क्या करना है।"
    },

    # Setup Wizard
    "wizard_title": {
        "en": "⚙️ Configure Your Analysis",
        "hi": "⚙️ अपने खेत की जानकारी भरें"
    },
    "wizard_step_1": {
        "en": "📍 Where is your field?",
        "hi": "📍 आपका खेत कहाँ है?"
    },
    "option_map": {
        "en": "🗺️ Click on Map",
        "hi": "🗺️ मैप पर क्लिक करें"
    },
    "option_map_desc": {
        "en": "Easiest way - zoom into your area on the map and draw your field boundary.",
        "hi": "सबसे आसान तरीका — नीचे मैप पर अपने खेत की बाउंड्री बनाएं।"
    },
    "option_upload": {
        "en": "📁 Upload GeoJSON File",
        "hi": "📁 फाइल अपलोड करें"
    },
    "option_upload_desc": {
        "en": "If you have a digital boundary file (.geojson or .json), upload it here.",
        "hi": "अगर आपके पास खेत की बाउंड्री फाइल है तो यहाँ अपलोड करें।"
    },
    "option_wris": {
        "en": "🏘️ Select Command Area (WRIS)",
        "hi": "🏘️ सिचाई कमांड एरिया चुनें"
    },
    "option_wris_desc": {
        "en": "Choose from registered government canal command areas in India.",
        "hi": "सरकारी सिचाई क्षेत्र की सूची में से अपना क्षेत्र चुनें।"
    },
    "map_instruction": {
        "en": "👇 Find your village on the map below and draw your field boundary:",
        "hi": "👇 नीचे अपने गाँव को ढूंढें और अपने खेत की बाउंड्री (ड्रॉ टूल से) मार्क करें:"
    },
    "btn_next": {
        "en": "Next Step →",
        "hi": "आगे बढ़ें →"
    },
    "btn_prev": {
        "en": "← Go Back",
        "hi": "← पीछे जाएं"
    },

    "wizard_step_2": {
        "en": "📅 Which crop season and year?",
        "hi": "📅 फसल का मौसम और साल चुनें"
    },
    "season_kharif_title": {
        "en": "🌧️ Kharif (Monsoon Season)",
        "hi": "🌧️ खरीफ (बरसात की फसल)"
    },
    "season_kharif_desc": {
        "en": "June - November (Rice, Cotton, Maize, Soybean)",
        "hi": "जून से नवंबर (धान, कपास, मक्का, सोयाबीन)"
    },
    "season_rabi_title": {
        "en": "❄️ Rabi (Winter Season)",
        "hi": "❄️ रबी (सर्दियों की फसल)"
    },
    "season_rabi_desc": {
        "en": "November - April (Wheat, Mustard, Chickpea)",
        "hi": "नवंबर से अप्रैल (गेहूं, सरसों, चना, जौ)"
    },
    "season_zaid_title": {
        "en": "☀️ Zaid (Summer Season)",
        "hi": "☀️ जायद (गर्मियों की फसल)"
    },
    "season_zaid_desc": {
        "en": "March - June (Watermelon, Cucumber, Pulses)",
        "hi": "मार्च से जून (तरबूज, ककड़ी, मूंग, मूंगफली)"
    },
    "year_label": {
        "en": "📅 Target Year",
        "hi": "📅 कौन सा साल?"
    },

    "wizard_step_3": {
        "en": "🌾 What crops did you plant?",
        "hi": "🌾 आपके खेत में क्या उगा है?"
    },
    "crop_help_note": {
        "en": "💡 Don't remember? No worries, our satellite will automatically identify them.",
        "hi": "💡 याद नहीं? कोई बात नहीं — सैटेलाइट खुद पता कर लेगा!"
    },

    "wizard_step_4": {
        "en": "📸 Upload a field photo (Optional)",
        "hi": "📸 खेत की फोटो अपलोड करें (ऑप्शनल)"
    },
    "optional_badge": {
        "en": "⭐ Optional - Not Mandatory",
        "hi": "⭐ ऑप्शनल — ज़रूरी नहीं"
    },
    "upload_dashed_box": {
        "en": "Drop your field photo here or click to upload. High quality phone photos work best!",
        "hi": "अपने खेत या पत्तियों की फोटो यहाँ अपलोड करें। फ़ोन से ली हुई फोटो भी चलेगी! (JPG, PNG)"
    },
    "btn_gps_photo": {
        "en": "📍 Get via GPS location",
        "hi": "📍 जीपीएस से ऑटोमैटिक पता करें"
    },
    "btn_skip": {
        "en": "Will Do Later →",
        "hi": "बाद में करेंगे →"
    },

    "wizard_step_5": {
        "en": "✅ Everything is Ready!",
        "hi": "✅ सब कुछ तैयार है!"
    },
    "summary_title": {
        "en": "Selected Summary",
        "hi": "भरी गई जानकारी"
    },
    "summary_loc": {
        "en": "📍 Location: ",
        "hi": "📍 क्षेत्र: "
    },
    "summary_season": {
        "en": "📅 Season: ",
        "hi": "📅 मौसम: "
    },
    "summary_crops": {
        "en": "🌾 Crops: ",
        "hi": "🌾 फसलें: "
    },
    "summary_photo": {
        "en": "📸 Photo: ",
        "hi": "📸 फोटो: "
    },
    "photo_yes": {
        "en": "Uploaded ✓",
        "hi": "अपलोड हो गई ✓"
    },
    "photo_no": {
        "en": "Not uploaded (Will skip)",
        "hi": "अपलोड नहीं की (छोड़ दिया)"
    },
    "btn_run_analysis": {
        "en": "🚀 Launch AgroSense Analysis",
        "hi": "🚀 एनालिसिस शुरू करें!"
    },

    # Loading screen
    "loading_title": {
        "en": "🛰️ Fetching Satellite Data...",
        "hi": "🛰️ सैटेलाइट से डेटा आ रहा है..."
    },
    "loading_subtitle": {
        "en": "Please wait. We are running calculations across space arrays. This takes 15-30 seconds.",
        "hi": "कृपया इंतज़ार करें। हम अंतरिक्ष से डेटा ला रहे हैं। इसमें 15-30 सेकंड लगेंगे।"
    },
    "loading_fact_label": {
        "en": "💡 Did You Know?",
        "hi": "💡 क्या आप जानते हैं?"
    },
    "step_s2": {
        "en": "Fetching Sentinel-2 Optical Data...",
        "hi": "सेंटिनल-2 ऑप्टिकल (कलर) डेटा लाया जा रहा है..."
    },
    "step_s1": {
        "en": "Processing Sentinel-1 SAR Radar Data...",
        "hi": "सेंटिनल-1 सार (राडार) डेटा लाया जा रहा है..."
    },
    "step_era5": {
        "en": "Fetching Climate and Temperature Data...",
        "hi": "मौसम और तापमान का डेटा आ रहा है..."
    },
    "step_indices": {
        "en": "Calculating Crop Health Indices...",
        "hi": "फसल की सेहत का लेवल नापा जा रहा है..."
    },
    "step_stages": {
        "en": "Mapping Phenology & Growth Stages...",
        "hi": "फसल बढ़ने की स्टेज पहचानी जा रही है..."
    },
    "step_ml": {
        "en": "Training AI Stacked Ensemble Model...",
        "hi": "मशीन लर्निंग (AI) द्वारा फसल पहचानी जा रही है..."
    },
    "step_stress": {
        "en": "Detecting Moisture Stress Anomalies...",
        "hi": "मिट्टी और फसल में पानी की कमी जांची जा रही है..."
    },
    "step_water": {
        "en": "Calculating FAO-56 Daily Water Balance...",
        "hi": "दैनिक जल संतुलन (Water Balance) की गणना हो रही है..."
    },
    "step_advisory": {
        "en": "Generating Irrigation Advisories...",
        "hi": "सिचाई की सलाह तैयार की जा रही है..."
    },
    "step_done": {
        "en": "Finalizing Dashboard...",
        "hi": "रिपोर्ट तैयार की जा रही है..."
    },

    # Results Dashboard
    "results_dashboard_title": {
        "en": "📊 Crop Intelligence Report",
        "hi": "📊 फसल रिपोर्ट"
    },
    "source_live_gee": {
        "en": "🛰️ Live GEE Data",
        "hi": "🛰️ लाइव सैटेलाइट डेटा"
    },
    
    # Alert Banners
    "alert_critical": {
        "en": "🚨 URGENT: {area} Hectares Need Water IMMEDIATELY!",
        "hi": "🚨 ज़रूरी सूचना: {area} हेक्टेयर खेत में अभी पानी दीजिए!"
    },
    "alert_warning": {
        "en": "⚠️ ATTENTION: {area} Hectares Require Irrigation within 3 Days.",
        "hi": "⚠️ ध्यान दें: {area} हेक्टेयर खेत में अगले 3 दिनों में पानी देना होगा।"
    },
    "alert_success": {
        "en": "✅ ALL GOOD: Moisture Levels are Adequate. No Irrigation Needed This Week.",
        "hi": "✅ सब ठीक है: खेतों में नमी भरपूर है। इस हफ्ते पानी देने की कोई आवश्यकता नहीं है।"
    },

    # Hero Metrics
    "metric_area_title": {
        "en": "Analysed Area",
        "hi": "विश्लेषण किया गया क्षेत्र"
    },
    "metric_area_unit": {
        "en": "Hectares",
        "hi": "हेक्टेयर"
    },
    "metric_crop_title": {
        "en": "Dominant Crop",
        "hi": "सबसे ज़्यादा उगाई फसल"
    },
    "metric_stress_title": {
        "en": "Moisture Stress",
        "hi": "पानी की कमी (तनाव)"
    },
    "metric_stress_sub": {
        "en": "{severe}% Severe • {mild}% Mild",
        "hi": "{severe}% ज़्यादा • {mild}% हल्की"
    },
    "metric_irr_title": {
        "en": "Needs Irrigation",
        "hi": "पानी की ज़रूरत"
    },
    "metric_irr_sub": {
        "en": "Within Next 3-7 Days",
        "hi": "अगले 3-7 दिनों में"
    },

    # Plain Language Interpretation Cards
    "interpretation_title": {
        "en": "📋 What Does This Mean For You?",
        "hi": "📋 आपके लिए इसका क्या मतलब है?"
    },
    "card_health_title": {
        "en": "🌾 How is your crop doing?",
        "hi": "🌾 आपकी फसल कैसी है?"
    },
    "card_health_status_good": {
        "en": "✅ Very Healthy",
        "hi": "✅ बहुत बढ़िया"
    },
    "card_health_status_warning": {
        "en": "⚠️ Pay Attention",
        "hi": "⚠️ ध्यान दीजिए"
    },
    "card_health_desc": {
        "en": "About {pct}% of your fields are showing signs of moisture stress. If not irrigated soon, crop yield could be reduced.",
        "hi": "आपके लगभग {pct}% खेत में पानी की कमी के लक्षण दिख रहे हैं। समय पर पानी न देने से पैदावार कम हो सकती है।"
    },
    "card_health_btn": {
        "en": "View Crop Map 📊",
        "hi": "फसल मैप देखें 📊"
    },
    
    "card_water_title": {
        "en": "💧 When to water?",
        "hi": "💧 कब पानी दें?"
    },
    "card_water_status_immediate": {
        "en": "🚨 Water Immediately",
        "hi": "🚨 अभी पानी दें!"
    },
    "card_water_status_soon": {
        "en": "📅 Within 3 Days",
        "hi": "📅 3 दिनों में दें"
    },
    "card_water_status_ok": {
        "en": "✅ Sufficient Moisture",
        "hi": "✅ अभी पानी न दें"
    },
    "card_water_desc": {
        "en": "The soil has enough moisture for the next few days. However, {area} hectares must be irrigated by the weekend.",
        "hi": "खेतों की मिट्टी में अगले कुछ दिनों के लिए नमी पर्याप्त है। हालांकि, {area} हेक्टेयर हिस्से में इस हफ्ते पानी देना होगा।"
    },
    "card_water_btn": {
        "en": "View Schedule 📅",
        "hi": "शेड्यूल देखें 📅"
    },

    "card_weather_title": {
        "en": "☀️ 7-Day Weather Outlook",
        "hi": "☀️ अगले 7 दिन का मौसम"
    },
    "card_weather_status_dry": {
        "en": "☁️ No Rain Expected",
        "hi": "☁️ बारिश की कोई उम्मीद नहीं"
    },
    "card_weather_status_rain": {
        "en": "🌧️ Rain Expected",
        "hi": "🌧️ बारिश होने की संभावना"
    },
    "card_weather_desc": {
        "en": "No significant rainfall is expected this week. Rely on manual irrigation as advised by the schedule.",
        "hi": "अगले 7 दिनों में बारिश की संभावना नहीं है। इसलिए हमारे शेड्यूल के अनुसार सिंचाई की तैयारी करें।"
    },
    "card_weather_btn": {
        "en": "View Full Weather 🌤️",
        "hi": "पूरा मौसम देखें 🌤️"
    },

    # Map Controls
    "map_section_title": {
        "en": "🗺️ Field Location & Layers Map",
        "hi": "🗺️ आपके खेत का नक्शा"
    },
    "layer_crop": {
        "en": "Crop Type",
        "hi": "उगाई गई फसल"
    },
    "layer_stress": {
        "en": "Moisture Stress",
        "hi": "पानी की कमी (तनाव)"
    },
    "layer_irr": {
        "en": "Irrigation Advisory",
        "hi": "सिचाई की सलाह"
    },
    "layer_deficit": {
        "en": "Water Deficit (mm)",
        "hi": "पानी की कमी (mm)"
    },
    "layer_ndvi": {
        "en": "NDVI Vigor",
        "hi": "फसल की हरियाली"
    },

    # Legend Labels
    "legend_healthy": {"en": "Healthy Crop", "hi": "स्वस्थ फसल"},
    "legend_mild": {"en": "Mild Stress", "hi": "हल्का तनाव"},
    "legend_mod": {"en": "Moderate Stress", "hi": "मध्यम तनाव"},
    "legend_severe": {"en": "Severe Stress", "hi": "गंभीर तनाव"},
    "legend_no_irr": {"en": "No Water Needed", "hi": "पानी की ज़रूरत नहीं"},
    "legend_soon_irr": {"en": "Irrigate in 3 Days", "hi": "3 दिन में पानी दें"},
    "legend_now_irr": {"en": "IRRIGATE NOW", "hi": "अभी पानी दें"},
    "legend_waterlogged": {"en": "Waterlogged / Wet", "hi": "ज़्यादा गीला (जलभराव)"},

    # Timeline Section
    "timeline_title": {
        "en": "🌱 Soil Moisture & NDVI Timeline",
        "hi": "🌱 मिट्टी की नमी और फसल स्वास्थ्य का ग्राफ"
    },
    "timeline_subtitle": {
        "en": "Measured by satellite every 6 days over the crop cycle",
        "hi": "सैटेलाइट द्वारा हर 6 दिन में मापा गया फसल चक्र"
    },
    "stat_drying": {
        "en": "📉 Soil dried by {val}% in last 6 days",
        "hi": "📉 पिछले 6 दिनों में मिट्टी {val}% सूखी हुई"
    },
    "stat_rain": {
        "en": "☔ {val} mm rain received this week",
        "hi": "☔ इस हफ्ते {val} mm बारिश हुई"
    },
    "stat_need": {
        "en": "💧 Crop needs {val} mm water for optimal growth",
        "hi": "💧 फसल को सही बढ़त के लिए {val} mm पानी चाहिए"
    },

    # Irrigation Schedule Table
    "table_title": {
        "en": "📅 Zone-Wise Irrigation Schedule",
        "hi": "📅 पानी देने का टाइम टेबल"
    },
    "table_sub": {
        "en": "Actionable zones sorted by urgency",
        "hi": "खेत के अलग-अलग हिस्से जहाँ पानी देना ज़रूरी है"
    },
    "col_zone": {"en": "Zone", "hi": "क्षेत्र (ज़ोन)"},
    "col_crop": {"en": "Crop", "hi": "फसल"},
    "col_area": {"en": "Area", "hi": "रकबा (रकबे का नाप)"},
    "col_when": {"en": "When to Irrigate", "hi": "कब पानी दें"},
    "col_status": {"en": "Status", "hi": "हालत"},
    "status_urgent": {"en": "🔴 CRITICAL", "hi": "🔴 बहुत ज़रूरी"},
    "status_soon": {"en": "🟡 SOON", "hi": "🟡 जल्द ही"},
    "status_ok": {"en": "🟢 NORMAL", "hi": "🟢 ठीक है"},
    "status_good": {"en": "✅ WET", "hi": "✅ पर्याप्त नमी"},
    "when_today": {"en": "TODAY", "hi": "आज ही"},
    "when_3days": {"en": "In 3 Days", "hi": "3 दिन में"},
    "when_7days": {"en": "In 7 Days", "hi": "7 दिन में"},
    "when_none": {"en": "Not Needed", "hi": "अभी नहीं चाहिए"},
    "btn_download_csv": {
        "en": "📥 Download Table (Excel/CSV)",
        "hi": "📥 यह टाइम टेबल डाउनलोड करें (Excel/CSV)"
    },

    # Photo Analysis
    "photo_title": {
        "en": "📸 AI Photo Diagnosis Results",
        "hi": "📸 आपके द्वारा भेजी गई फोटो की जांच"
    },
    "photo_condition": {"en": "Condition Identified:", "hi": "जांची गई स्थिति:"},
    "photo_symptoms": {"en": "Visual Symptoms:", "hi": "लक्षण जो दिख रहे हैं:"},
    "photo_remedy": {"en": "Treatment & Remedy:", "hi": "डॉक्टर की सलाह (उपचार):"},
    "photo_confidence": {"en": "🎯 AI Confidence Level: {val}%", "hi": "🎯 AI को {val}% यकीन है"},

    # Voice Query
    "voice_section_title": {
        "en": "🗣️ Ask Your Crop Doctor",
        "hi": "🗣️ कोई सवाल है? डॉक्टर से पूछें"
    },
    "voice_placeholder": {
        "en": "Ask: 'Why are my wheat leaves turning yellow?' or 'When should I irrigate?'",
        "hi": "जैसे: 'गेहूं की पत्तियां पीली क्यों हो रही हैं?' या 'मुझे पानी कब देना चाहिए?'"
    },
    "btn_voice": {"en": "🎤 Ask by Voice", "hi": "🎤 आवाज़ से पूछें (बोले)"},
    "btn_text": {"en": "✉️ Ask by Text", "hi": "✉️ लिखकर पूछें"},
    "answer_header": {"en": "🤖 Crop Doctor's Answer:", "hi": "🤖 डॉक्टर का जवाब:"},
    
    # Detail Tabs
    "tab_overview": {"en": "🌍 Overview Map", "hi": "🌍 मुख्य नक्शा"},
    "tab_crop": {"en": "🌾 Crop Type Map", "hi": "🌾 फसल का नक्शा"},
    "tab_stress": {"en": "🌡️ Stress Analysis", "hi": "🌡️ पानी का तनाव"},
    "tab_water": {"en": "💧 Irrigation Advisories", "hi": "💧 सिचाई की सलाह"},
    "tab_soil": {"en": "🌱 Soil Moisture", "hi": "🌱 मिट्टी की नमी"},
    "tab_accuracy": {"en": "📊 Model Accuracy", "hi": "📊 मॉडल शुद्धता"},
    "tab_download": {"en": "📥 Download Packages", "hi": "📥 डाउनलोड सेंटर"},
    
    # Accuracy Tab
    "toggle_tech": {"en": "👨‍💻 Show Advanced Technical Metrics", "hi": "👨‍💻 एडवांस टेक्निकल जानकारी देखें"},
    "simple_accuracy_title": {
        "en": "🎯 Simple Accuracy Score",
        "hi": "🎯 हमारा स्कोर कार्ड"
    },
    "simple_accuracy_desc": {
        "en": "Our AI model successfully verified crop types in {val}% of the field tests.",
        "hi": "हमारे AI ने 100 खेतों में से {val} खेतों में फसल और पानी की कमी का बिल्कुल सही अंदाज़ा लगाया।"
    },

    # Download Tab
    "dl_card_map_title": {"en": "Crop Map (GeoTIFF)", "hi": "फसल का डिजिटल नक्शा (GeoTIFF)"},
    "dl_card_map_desc": {"en": "30m high-resolution GIS map layer", "hi": "30 मीटर रेजोल्यूशन वाला डिजिटल नक्शा"},
    "dl_card_report_title": {"en": "Complete PDF Report", "hi": "पूरी रिपोर्ट PDF फाइल"},
    "dl_card_report_desc": {"en": "Bilingual maps and advisory summary", "hi": "सभी नक्शे, सलाह और पानी के टाइम टेबल के साथ"},
    "dl_card_table_title": {"en": "Schedule Spreadsheet", "hi": "सिंचाई टाइम टेबल (Excel)"},
    "dl_card_table_desc": {"en": "Zone-wise irrigation coordinates", "hi": "हर क्षेत्र का सिंचाई टाइम टेबल"},
    "dl_card_print_title": {"en": "Printable A4 Sheet", "hi": "प्रिंट करने योग्य पेज"},
    "dl_card_print_desc": {"en": "Black & white print friendly advice", "hi": "प्रिंटर के लिए आसान रिपोर्ट"},
    "btn_download": {"en": "💾 Download File", "hi": "💾 डाउनलोड करें"},
    
    # Sidebar Redesign
    "sb_location_header": {"en": "📍 Selected Location", "hi": "📍 चुनी गई जगह"},
    "sb_season_header": {"en": "📅 Season & Year", "hi": "📅 मौसम और वर्ष"},
    "sb_crops_header": {"en": "🌾 Target Crops", "hi": "🌾 चुनी गई फसलें"},
    "sb_method_header": {"en": "💧 Irrigation Method", "hi": "💧 सिंचाई का तरीका"},
    "method_flood": {"en": "🌊 Flood Irrigation", "hi": "🌊 क्यारी सिंचाई (Flood)"},
    "method_sprinkler": {"en": "💦 Sprinkler", "hi": "💦 फव्वारा सिंचाई (Sprinkler)"},
    "method_drip": {"en": "💧 Drip System", "hi": "💧 ड्रिप सिंचाई (Drip)"},
    "sb_sensitivity": {"en": "🌡️ Stress Alert Level", "hi": "🌡️ पानी की कमी का अलर्ट"},
    "sensitivity_early": {"en": "Alert Early (Mild)", "hi": "जल्दी बताएं (हल्की कमी पर)"},
    "sensitivity_late": {"en": "Alert Late (Severe)", "hi": "देर से बताएं (गंभीर कमी पर)"},
}

# ── Crop Name Translations ────────────────────────────────────────────────────
CROP_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "wheat":        {"en": "Wheat",        "hi": "गेहूं"},
    "rice":         {"en": "Rice",         "hi": "धान (चावल)"},
    "cotton":       {"en": "Cotton",       "hi": "कपास"},
    "maize":        {"en": "Maize",        "hi": "मक्का"},
    "soybean":      {"en": "Soybean",      "hi": "सोयाबीन"},
    "sugarcane":    {"en": "Sugarcane",    "hi": "गन्ना"},
    "groundnut":    {"en": "Groundnut",    "hi": "मूंगफली"},
    "mustard":      {"en": "Mustard",      "hi": "सरसों"},
    "sunflower":    {"en": "Sunflower",    "hi": "सूरजमुखी"},
    "sorghum":      {"en": "Sorghum",      "hi": "ज्वार"},
    "pearl_millet": {"en": "Pearl Millet", "hi": "बाजरा"},
    "chickpea":     {"en": "Chickpea",     "hi": "चना"},
    "lentil":       {"en": "Lentil",       "hi": "मसूर दाल"},
    "barley":       {"en": "Barley",       "hi": "जौ"},
    "watermelon":   {"en": "Watermelon",   "hi": "तरबूज"},
    "cucumber":     {"en": "Cucumber",     "hi": "खीरा"},
    "mungbean":     {"en": "Mung Bean",    "hi": "मूंग"},
    "other":        {"en": "Other",        "hi": "अन्य"},
    "uncertain":    {"en": "Uncertain",    "hi": "अज्ञात"},
}


def translate(key: str, lang: str = "hi") -> str:
    """Return the translated string for a given key. Defaults to Hindi if missing."""
    return TRANSLATIONS.get(key, {}).get(lang, key)


def translate_crop(crop: str, lang: str = "hi") -> str:
    """Return the translated crop name for a given crop key.

    Falls back to the crop key itself if not found in CROP_TRANSLATIONS.
    """
    return CROP_TRANSLATIONS.get(crop.lower(), {}).get(lang, crop)
