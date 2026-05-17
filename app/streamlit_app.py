"""
Streamlit UI for Crop Classification MLOps Pipeline
"""
import base64
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
import plotly.graph_objects as go
import requests
import streamlit as st

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Classification AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8001")

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    :root {
        --primary-color: #22c55e !important;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #050c08 0%,
            #0a1f12 20%,
            #0d2416 40%,
            #091a2a 65%,

            #040a0b 100%
        );
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.98) !important;
        border-right: 1px solid rgba(34, 197, 94, 0.15) !important;
    }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .card-success {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        transition: border-color 0.2s;
    }

    .metric-card:hover { border-color: rgba(34,197,94,0.35); }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #22c55e;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .prediction-box {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.08));
        border: 1.5px solid rgba(34,197,94,0.4);
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-crop {
        font-size: 2.8rem;
        font-weight: 800;
        color: #22c55e;
        margin: 8px 0;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-online {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.3);
    }

    .status-offline {
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.3);
    }

    .divider {
        display: flex;
        justify-content: space-between;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }

    .divider-key { color: rgba(255,255,255,0.5); font-size: 0.875rem; }
    .divider-value { color: #22c55e; font-weight: 600; font-size: 0.875rem; }

    .stButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(34,197,94,0.2) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(34,197,94,0.35) !important;
    }

    /* Slider thumb — green */
    div[data-testid="stSlider"] [role="slider"] {
        background: #22c55e !important;
        border-color: #22c55e !important;
        box-shadow: 0 0 0 2px rgba(34,197,94,0.25) !important;
    }

    /* Slider filled track — green */
    div[data-testid="stSlider"] > div > div > div:not([data-testid]) {
        background: #22c55e !important;
    }

    /* Tick bar min/max — clean */
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {
        background: transparent !important;
        color: rgba(28,239,172,0.35) !important;
        border: none !important;
        font-size: 0.78rem !important;
    }

    /* Target Streamlit's StyledThumbValue class */
    .StyledThumbValue {
        color: white !important;
    }

    /* Alternative selectors */
    div[data-testid="stSlider"] *[style*="color"] {
        color: white !important;
    }

    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] p {
        color: white !important;
    }

    h1, h2, h3, h4 { color: white !important; }
    p, label { color: rgba(255,255,255,0.75) !important; }

    #MainMenu, footer, header { visibility: hidden; }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(34,197,94,0.15) !important;
        color: #22c55e !important;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    button[data-testid="stBaseButton-headerNoPadding"] { color: #22c55e !important; }
    div[data-testid="stTooltipContent"] p { color: #0a0f0d !important; }
    div[data-testid="stTooltipContent"] { color:  !important; }

    .stCode > div > pre > code { background: black !important; }
    .stDownloadButton > button { background: linear-gradient(135deg, #22c55e, #16a34a) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

CROP_EMOJIS = {
    "Rice": "🌾", "Maize": "🌽", "Chickpea": "🫘", "KidneyBeans": "🫘",
    "PigeonPeas": "🫘", "MothBeans": "🫘", "MungBean": "🫘",
    "Blackgram": "🫘", "Lentil": "🌱", "Pomegranate": "🍎",
    "Banana": "🍌", "Mango": "🥭", "Grapes": "🍇", "Watermelon": "🍉",
    "Muskmelon": "🍈", "Apple": "🍎", "Orange": "🍊", "Papaya": "🍈",
    "Coconut": "🥥", "Cotton": "🌸", "Jute": "🌿", "Coffee": "☕",
}

UNITS = {
    "Nitrogen": "kg/ha", "Phosphorus": "kg/ha", "Potassium": "kg/ha",
    "Temperature": "°C", "Humidity": "%", "pH_Value": "pH", "Rainfall": "mm",
}

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.8)", family="Inter"),
    margin=dict(t=40, b=20, l=10, r=10),
)

# ── API helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_api_health(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=60)
def get_performance(api_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api_url}/performance", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def get_summary(api_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api_url}/summary", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def get_model_info(api_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api_url}/model-info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def predict_crop(api_url: str, data: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{api_url}/predict", json=data, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px;'>
        <div style='font-size: 1.4rem; font-weight: 800; color: #22c55e;'>🌾 Crop AI</div>
        <div style='font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 2px;'>
            ML-Powered Crop Advisor
        </div>
    </div>
    """, unsafe_allow_html=True)

    is_healthy = check_api_health(API_URL)
    status_cls = "status-online" if is_healthy else "status-offline"
    status_txt = "● API Online" if is_healthy else "● API Offline"
    st.markdown(
        f'<div class="status-badge {status_cls}">{status_txt}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🌾  Predict", "📂  Batch Predict", "📊  Performance", "📈  Dataset", "🤖  Model Info"],
        label_visibility="collapsed",
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='color: rgba(255,255,255,0.35); font-size: 0.75rem; line-height: 1.9;'>
        FastAPI + Streamlit<br>
        Deployed on AWS EC2<br>
        Random Forest · 22 Crops<br>
        Accuracy: 94.32%<br>
        <br>
        <a href='{API_URL}/docs' target='_blank'
           style='color: #22c55e; text-decoration: none;'>
            📖 API Documentation ↗
        </a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🌾  Predict":

    # ── Hero banner (base64 SVG avoids Streamlit's inline-SVG DOM error) ───────
    _SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 180">'
        '<defs>'
        '<linearGradient id="bgG" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#071510"/>'
        '<stop offset="50%" stop-color="#0d1f10"/>'
        '<stop offset="100%" stop-color="#071510"/>'
        '</linearGradient>'
        '<linearGradient id="gl" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#22c55e" stop-opacity="0.18"/>'
        '<stop offset="100%" stop-color="#22c55e" stop-opacity="0"/>'
        '</linearGradient>'
        '<linearGradient id="gnd" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#16a34a" stop-opacity="0.55"/>'
        '<stop offset="100%" stop-color="#14532d" stop-opacity="0.9"/>'
        '</linearGradient>'
        '</defs>'
        # sky
        '<rect width="1200" height="180" fill="url(#bgG)"/>'
        # stars
        '<circle cx="100" cy="20" r="1.2" fill="white" fill-opacity="0.4"/>'
        '<circle cx="230" cy="12" r="0.8" fill="white" fill-opacity="0.3"/>'
        '<circle cx="370" cy="25" r="1.0" fill="white" fill-opacity="0.35"/>'
        '<circle cx="520" cy="8"  r="1.2" fill="white" fill-opacity="0.3"/>'
        '<circle cx="680" cy="18" r="0.8" fill="white" fill-opacity="0.25"/>'
        '<circle cx="820" cy="10" r="1.0" fill="white" fill-opacity="0.35"/>'
        '<circle cx="960" cy="22" r="1.2" fill="white" fill-opacity="0.3"/>'
        '<circle cx="1090" cy="14" r="0.9" fill="white" fill-opacity="0.25"/>'
        '<circle cx="1170" cy="30" r="1.0" fill="white" fill-opacity="0.3"/>'
        # moon glow
        '<circle cx="1100" cy="35" r="22" fill="#22c55e" fill-opacity="0.06"/>'
        '<circle cx="1100" cy="35" r="14" fill="#22c55e" fill-opacity="0.10"/>'
        '<circle cx="1100" cy="35" r="8"  fill="#86efac" fill-opacity="0.20"/>'
        # horizon glow
        '<ellipse cx="600" cy="130" rx="380" ry="60" fill="url(#gl)"/>'
        # ground
        '<rect y="130" width="1200" height="50" fill="url(#gnd)"/>'
        # field rows
        '<line x1="0" y1="148" x2="1200" y2="148" stroke="#22c55e" stroke-width="0.6" stroke-opacity="0.25"/>'
        '<line x1="0" y1="158" x2="1200" y2="158" stroke="#22c55e" stroke-width="0.6" stroke-opacity="0.20"/>'
        '<line x1="0" y1="168" x2="1200" y2="168" stroke="#22c55e" stroke-width="0.6" stroke-opacity="0.15"/>'
        # left wheat stalks
        '<line x1="60"  y1="130" x2="60"  y2="65"  stroke="#4ade80" stroke-width="2.0" stroke-opacity="0.7"/>'
        '<line x1="60"  y1="65"  x2="48"  y2="48"  stroke="#4ade80" stroke-width="1.2" stroke-opacity="0.6"/>'
        '<line x1="60"  y1="70"  x2="72"  y2="52"  stroke="#4ade80" stroke-width="1.2" stroke-opacity="0.6"/>'
        '<line x1="60"  y1="75"  x2="46"  y2="60"  stroke="#4ade80" stroke-width="1.2" stroke-opacity="0.5"/>'
        '<ellipse cx="60" cy="52" rx="7" ry="16" fill="#22c55e" fill-opacity="0.30"/>'
        '<line x1="100" y1="130" x2="95"  y2="58"  stroke="#4ade80" stroke-width="1.8" stroke-opacity="0.65"/>'
        '<line x1="95"  y1="58"  x2="83"  y2="42"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.55"/>'
        '<line x1="95"  y1="63"  x2="107" y2="46"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.55"/>'
        '<ellipse cx="95" cy="46" rx="6" ry="14" fill="#22c55e" fill-opacity="0.25"/>'
        '<line x1="135" y1="130" x2="140" y2="72"  stroke="#4ade80" stroke-width="1.8" stroke-opacity="0.60"/>'
        '<line x1="140" y1="72"  x2="128" y2="55"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.50"/>'
        '<line x1="140" y1="77"  x2="152" y2="60"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.50"/>'
        '<ellipse cx="140" cy="59" rx="6" ry="13" fill="#22c55e" fill-opacity="0.22"/>'
        '<line x1="170" y1="130" x2="168" y2="80"  stroke="#4ade80" stroke-width="1.5" stroke-opacity="0.50"/>'
        '<ellipse cx="168" cy="68" rx="5" ry="11" fill="#22c55e" fill-opacity="0.18"/>'
        # right wheat stalks
        '<line x1="1140" y1="130" x2="1140" y2="65"  stroke="#4ade80" stroke-width="2.0" stroke-opacity="0.7"/>'
        '<line x1="1140" y1="65"  x2="1128" y2="48"  stroke="#4ade80" stroke-width="1.2" stroke-opacity="0.6"/>'
        '<line x1="1140" y1="70"  x2="1152" y2="52"  stroke="#4ade80" stroke-width="1.2" stroke-opacity="0.6"/>'
        '<ellipse cx="1140" cy="52" rx="7" ry="16" fill="#22c55e" fill-opacity="0.30"/>'
        '<line x1="1100" y1="130" x2="1105" y2="58"  stroke="#4ade80" stroke-width="1.8" stroke-opacity="0.65"/>'
        '<line x1="1105" y1="58"  x2="1093" y2="42"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.55"/>'
        '<line x1="1105" y1="63"  x2="1117" y2="46"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.55"/>'
        '<ellipse cx="1105" cy="46" rx="6" ry="14" fill="#22c55e" fill-opacity="0.25"/>'
        '<line x1="1065" y1="130" x2="1060" y2="72"  stroke="#4ade80" stroke-width="1.8" stroke-opacity="0.60"/>'
        '<line x1="1060" y1="72"  x2="1048" y2="55"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.50"/>'
        '<line x1="1060" y1="77"  x2="1072" y2="60"  stroke="#4ade80" stroke-width="1.1" stroke-opacity="0.50"/>'
        '<ellipse cx="1060" cy="59" rx="6" ry="13" fill="#22c55e" fill-opacity="0.22"/>'
        '<line x1="1030" y1="130" x2="1032" y2="80"  stroke="#4ade80" stroke-width="1.5" stroke-opacity="0.50"/>'
        '<ellipse cx="1032" cy="68" rx="5" ry="11" fill="#22c55e" fill-opacity="0.18"/>'
        # title text
        '<text x="600" y="82" text-anchor="middle"'
        ' font-family="Inter,system-ui,sans-serif" font-size="44" font-weight="800"'
        ' fill="#22c55e">Crop Classification AI</text>'
        '<text x="600" y="112" text-anchor="middle"'
        ' font-family="Inter,system-ui,sans-serif" font-size="15" fill="rgba(255,255,255,0.5)">'
        'ML-Powered Agricultural Intelligence · Random Forest · 22 Crops'
        '</text>'
        # border
        '<rect width="1200" height="180" fill="none"'
        ' stroke="rgba(34,197,94,0.18)" stroke-width="1"/>'
        '</svg>'
    )
    _b64 = base64.b64encode(_SVG.encode()).decode()
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{_b64}"'
        ' style="width:100%;border-radius:16px;margin-bottom:12px;display:block;"/>',
        unsafe_allow_html=True,
    )

    # ── Input sliders ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🌱 Soil Parameters")
        nitrogen    = st.slider("Nitrogen (N) — kg/ha", 0, 140, 50,
                                help="Ratio of Nitrogen content in soil (kg/ha)")
        phosphorus  = st.slider("Phosphorus (P) — kg/ha", 5, 145, 30,
                                help="Ratio of Phosphorus content in soil (kg/ha)")
        potassium   = st.slider("Potassium (K) — kg/ha", 5, 205, 40,
                                help="Ratio of Potassium content in soil (kg/ha)")
        ph          = st.slider("pH Value", 3.5, 10.0, 6.5, step=0.1,
                                help="pH value of the soil (0 = acidic, 14 = alkaline)")

    with col2:
        st.markdown("### 🌤️ Weather Parameters")
        temperature = st.slider("Temperature — °C", 8.0, 45.0, 25.0, step=0.5,
                                help="Average temperature in °C")
        humidity    = st.slider("Humidity — %", 14.0, 100.0, 70.0, step=0.5,
                                help="Relative humidity in %")
        rainfall    = st.slider("Rainfall — mm", 20.0, 300.0, 100.0, step=1.0,
                                help="Annual rainfall in mm")

    # ── Input summary chips ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Current Input")
    chip_cols = st.columns(7)
    chip_data = [
        ("N", nitrogen, "kg/ha"),
        ("P", phosphorus, "kg/ha"),
        ("K", potassium, "kg/ha"),
        ("Temp", temperature, "°C"),
        ("Humidity", humidity, "%"),
        ("pH", ph, ""),
        ("Rain", rainfall, "mm"),
    ]
    for col, (label, value, unit) in zip(chip_cols, chip_data):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size:1.1rem'>{value:.1f}{unit}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ───────────────────────────────────────────────────────
    if st.button("🌾 Predict Best Crop", use_container_width=True):
        if not is_healthy:
            st.error("❌ API is offline. Please start the FastAPI server first.")
        else:
            payload = {
                "Nitrogen": nitrogen, "Phosphorus": phosphorus,
                "Potassium": potassium, "Temperature": temperature,
                "Humidity": humidity, "pH_Value": ph, "Rainfall": rainfall,
            }
            with st.spinner("Analyzing conditions..."):
                result = predict_crop(API_URL, payload)

            if result:
                crop = result["predicted"]
                emoji = CROP_EMOJIS.get(crop, "🌱")
                st.markdown(f"""
                <div class='prediction-box'>
                    <div style='color:rgba(255,255,255,0.5);font-size:0.85rem;
                                text-transform:uppercase;letter-spacing:0.1em;'>
                        Recommended Crop
                    </div>
                    <div class='prediction-crop'>{emoji} {crop}</div>
                    <div style='color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:12px;'>
                        ✅ Prediction successful
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.session_state.history.insert(0, {
                    "Crop": f"{emoji} {crop}",
                    "N": nitrogen, "P": phosphorus, "K": potassium,
                    "Temp (°C)": temperature, "Humidity (%)": humidity,
                    "pH": ph, "Rain (mm)": rainfall,
                })
            else:
                st.error("❌ Prediction failed. Please check the API connection.")

    # ── Prediction history ───────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕒 Prediction History")
        clear_col, _ = st.columns([1, 5])
        with clear_col:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Batch Predict":

    st.markdown("## 📂 Batch Prediction")
    st.markdown(
        "<p>Upload a CSV file with the required columns to predict crops for multiple rows at once.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Required columns")
    st.code("Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH_Value, Rainfall")

    sample_df = pd.DataFrame([
        {"Nitrogen": 80, "Phosphorus": 45, "Potassium": 40,
         "Temperature": 24.0, "Humidity": 82.0, "pH_Value": 6.5, "Rainfall": 200.0},
        {"Nitrogen": 78, "Phosphorus": 48, "Potassium": 20,
         "Temperature": 22.0, "Humidity": 65.0, "pH_Value": 6.0, "Rainfall": 60.0},
        {"Nitrogen": 20, "Phosphorus": 30, "Potassium": 200,
         "Temperature": 38.0, "Humidity": 50.0, "pH_Value": 7.0, "Rainfall": 30.0},
    ])
    st.download_button(
        "⬇ Download Sample CSV",
        data=sample_df.to_csv(index=False).encode(),
        file_name="sample_crop_input.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = ["Nitrogen", "Phosphorus", "Potassium",
                        "Temperature", "Humidity", "pH_Value", "Rainfall"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            elif not is_healthy:
                st.error("❌ API is offline. Please start the FastAPI server first.")
            else:
                st.markdown(f"**{len(df)} rows detected.** Running predictions…")
                results = []
                bar = st.progress(0, text="Starting…")
                for i, row in df.iterrows():
                    res = predict_crop(API_URL, row[required].to_dict())
                    if res:
                        crop = res["predicted"]
                        results.append(f"{CROP_EMOJIS.get(crop, '🌱')} {crop}")
                    else:
                        results.append("❌ Error")
                    bar.progress((i + 1) / len(df),
                                 text=f"Processing row {i + 1} of {len(df)}")

                bar.empty()
                df["Predicted Crop"] = results
                st.success(f"✅ {len(df)} predictions complete!")
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇ Download Results CSV",
                    data=df.to_csv(index=False).encode(),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Performance":

    st.markdown("## 📊 Model Performance")
    perf = get_performance(API_URL)

    if perf:
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            perf.get("accuracy", 0), perf.get("precision", 0),
            perf.get("recall", 0),   perf.get("f1", 0),
        ]
        colors = ["#22c55e", "#16a34a", "#15803d", "#86efac"]

        metric_cols = st.columns(4)
        for label, value, col, color in zip(labels, values, metric_cols, colors):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:{color}'>{value:.1%}</div>
                    <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📊 Bar Chart", "📡 Radar Chart"])

        with tab1:
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                textfont=dict(color="white", size=13),
            ))
            fig.update_layout(
                **CHART_LAYOUT,
                yaxis=dict(range=[0, 1.15], tickformat=".0%",
                           gridcolor="rgba(255,255,255,0.07)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                showlegend=False, height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(34,197,94,0.15)",
                line=dict(color="#22c55e", width=2.5),
                marker=dict(color="#22c55e", size=8),
            ))
            fig2.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                                    gridcolor="rgba(255,255,255,0.15)",
                                    color="rgba(255,255,255,0.6)"),
                    angularaxis=dict(color="rgba(255,255,255,0.8)",
                                     tickfont=dict(size=13)),
                    bgcolor="rgba(0,0,0,0)",
                ),
                **CHART_LAYOUT,
                showlegend=False, height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

        best = labels[values.index(max(values))]
        st.markdown(f"""
        <div class='card'>
            <strong style='color:#22c55e;'>🔍 Insight</strong>
            <p style='margin:6px 0 0;'>
                The model achieves its highest score on
                <strong style='color:white;'>{best}</strong>
                at <strong style='color:#22c55e;'>{max(values):.1%}</strong>.
                Overall performance is strong across all metrics, indicating
                reliable crop recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not fetch performance metrics. Make sure the API is running.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Dataset":

    st.markdown("## 📈 Dataset Summary")
    summary = get_summary(API_URL)

    if summary:
        features = ["Nitrogen", "Phosphorus", "Potassium",
                    "Temperature", "Humidity", "pH_Value", "Rainfall"]

        stats_data = []
        for f in features:
            if f in summary:
                d = summary[f]
                stats_data.append({
                    "Feature": f,
                    "Unit": UNITS.get(f, ""),
                    "Mean":   round(d.get("mean", 0), 2),
                    "Std":    round(d.get("std", 0), 2),
                    "Min":    round(d.get("min", 0), 2),
                    "Median": round(d.get("50%", 0), 2),
                    "Max":    round(d.get("max", 0), 2),
                })

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Feature Value Ranges")

        fig_box = go.Figure()
        for f in features:
            if f not in summary:
                continue
            d = summary[f]
            fig_box.add_trace(go.Box(
                x=[d.get("mean", 0)],
                q1=[d.get("25%", 0)],
                median=[d.get("50%", 0)],
                q3=[d.get("75%", 0)],
                lowerfence=[d.get("min", 0)],
                upperfence=[d.get("max", 0)],
                name=f,
                orientation="h",
                marker_color="#22c55e",
                line_color="#22c55e",
                fillcolor="rgba(34,197,94,0.15)",
            ))
        fig_box.update_layout(
            **CHART_LAYOUT,
            height=420, showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("### 🎯 Feature Averages")
        gauge_cols = st.columns(4)
        # Three-zone colors: low → mid → high green shades
        gauge_steps = [
            {"frac": 0.33, "color": "rgba(34,197,94,0.07)"},
            {"frac": 0.66, "color": "rgba(34,197,94,0.13)"},
            {"frac": 1.00, "color": "rgba(34,197,94,0.20)"},
        ]
        for i, f in enumerate(features):
            if f not in summary:
                continue
            d = summary[f]
            mean  = d.get("mean", 0)
            min_v = d.get("min", 0)
            max_v = d.get("max", 0)
            span  = max_v - min_v
            unit  = UNITS.get(f, "")

            steps = [
                {"range": [min_v, min_v + span * gauge_steps[0]["frac"]],
                 "color": gauge_steps[0]["color"]},
                {"range": [min_v + span * gauge_steps[0]["frac"],
                           min_v + span * gauge_steps[1]["frac"]],
                 "color": gauge_steps[1]["color"]},
                {"range": [min_v + span * gauge_steps[1]["frac"], max_v],
                 "color": gauge_steps[2]["color"]},
            ]

            with gauge_cols[i % 4]:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mean,
                    title={
                        "text": f"{f}<br><span style='font-size:0.7em;"
                                f"color:rgba(255,255,255,0.45)'>{unit}</span>",
                        "font": {"color": "white", "size": 13},
                    },
                    gauge={
                        "axis": {"range": [min_v, max_v],
                                 "tickcolor": "rgba(255,255,255,0.35)"},
                        "bar": {"color": "#22c55e", "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "rgba(255,255,255,0.08)",
                        "steps": steps,
                    },
                    number={"font": {"color": "#22c55e", "size": 18}},
                ))
                fig_g.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    height=220,
                    margin=dict(t=60, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.warning("⚠️ Could not fetch dataset summary. Make sure the API is running.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Info":

    st.markdown("## 🤖 Model Information")
    info = get_model_info(API_URL)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### ⚙️ Hyperparameters")
        hp = {
            "Algorithm":         info.get("model_name", "Random Forest") if info else "Random Forest",
            "N Estimators":      info.get("n_estimators", 100) if info else 100,
            "Max Depth":         info.get("max_depth", 5) if info else 5,
            "Min Samples Split": info.get("min_samples_split", 5) if info else 5,
            "Random State":      info.get("random_state", 42) if info else 42,
        }
        for key, value in hp.items():
            st.markdown(f"""
            <div class='divider'>
                <span class='divider-key'>{key}</span>
                <span class='divider-value'>{value}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 📦 Model Details")
        details = {
            "Model Size":       "799 kB",
            "Training Samples": "1,760",
            "Test Samples":     "440",
            "Total Classes":    "22 crops",
            "Input Features":   "7",
            "Data Split":       "80 / 20",
        }
        for key, value in details.items():
            st.markdown(f"""
            <div class='divider'>
                <span class='divider-key'>{key}</span>
                <span class='divider-value'>{value}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌿 Carbon Footprint")
    c1, c2, c3 = st.columns(3)
    for (label, val, emoji), col in zip([
        ("CO₂ Training", "0.00 g", "🌿"),
        ("CO₂ Inference", "0.00 g", "🌿"),
        ("Energy Rating", "A", "⚡"),
    ], [c1, c2, c3]):
        with col:
            st.markdown(f"""
            <div class='card-success'>
                <div style='font-size:2rem;'>{emoji}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌾 Supported Crops")
    crops = list(CROP_EMOJIS.keys())
    crop_cols = st.columns(6)
    for i, crop in enumerate(crops):
        with crop_cols[i % 6]:
            emoji = CROP_EMOJIS.get(crop, "🌱")
            st.markdown(f"""
            <div style='text-align:center;padding:10px 6px;
                        background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px;margin:4px 0;'>
                <div style='font-size:1.4rem;'>{emoji}</div>
                <div style='font-size:0.72rem;color:rgba(255,255,255,0.6);
                            margin-top:4px;'>{crop}</div>
            </div>""", unsafe_allow_html=True)
