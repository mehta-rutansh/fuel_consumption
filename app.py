import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fuel Consumption Analyzer",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─── CUSTOM CSS (mobile + theme friendly) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background - works with Streamlit's light/dark theme toggle */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-attachment: fixed;
}

#MainMenu, footer { visibility: hidden; }

/* Navbar */
.navbar {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding: 16px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
}
.navbar-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.3px;
}
.navbar-subtitle {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-left: auto;
}
@media (max-width: 640px) {
    .navbar-subtitle { margin-left: 0; width: 100%; }
    .navbar-title { font-size: 1.1rem; }
}

/* Metric cards - responsive */
.metric-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 16px 14px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    min-height: 100px;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,179,237,0.5);
}
.metric-icon { font-size: 1.6rem; margin-bottom: 4px; }
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #63b3ed;
    line-height: 1.15;
    word-break: break-word;
}
.metric-label {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.55);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.section-heading {
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
    margin: 1.3rem 0 0.9rem;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,179,237,0.4);
}

.info-box {
    background: rgba(99,179,237,0.1);
    border-left: 4px solid #63b3ed;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: rgba(255,255,255,0.85);
    font-size: 0.82rem;
    line-height: 1.6;
}

.result-box {
    background: rgba(72,187,120,0.15);
    border: 1px solid rgba(72,187,120,0.4);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-top: 12px;
}
.result-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #68d391;
    line-height: 1;
}
.result-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    margin-top: 6px;
}

.badge-green  { background:#276749; color:#9ae6b4; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; display:inline-block; }
.badge-blue   { background:#2a4365; color:#90cdf4; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; display:inline-block; }
.badge-yellow { background:#744210; color:#fbd38d; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; display:inline-block; }
.badge-orange { background:#7b341e; color:#fbd38d; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; display:inline-block; }
.badge-red    { background:#742a2a; color:#fc8181; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; display:inline-block; }

section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

.stSelectbox label, .stNumberInput label, .stSlider label, .stFileUploader label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #63b3ed, #4299e1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.4px;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,179,237,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,179,237,0.5) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(255,255,255,0.6) !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.2) !important;
    color: #63b3ed !important;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }

div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    display: block;
    border: 1px solid rgba(255,255,255,0.08);
    cursor: pointer;
    transition: all 0.2s;
}
div[role="radiogroup"] label:hover {
    background: rgba(99,179,237,0.15);
    border-color: rgba(99,179,237,0.3);
}

.chart-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
}

hr { border-color: rgba(255,255,255,0.1) !important; }

/* Mobile tweaks */
@media (max-width: 640px) {
    .metric-value { font-size: 1.2rem; }
    .section-heading { font-size: 1rem; }
    .result-value { font-size: 1.9rem; }
    div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL (auto-unzips if only the .zip is present) ─────
@st.cache_resource
def load_model():
    model_path = "fuel_model.pkl"
    zip_path   = "fuel_model.zip"

    if not os.path.exists(model_path) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(".")

    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found. Make sure `fuel_model.pkl` or `fuel_model.zip` "
                  "is in the same folder as app.py and committed to your repo.")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

# ─── LOAD DATA (CSV, with zip fallback for large files too) ───
@st.cache_data
def load_data():
    csv_path = "fuel_engineered.csv"
    zip_path = "fuel_engineered.zip"

    if not os.path.exists(csv_path) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(".")

    if not os.path.exists(csv_path):
        st.error("⚠️ Data file `fuel_engineered.csv` not found. Make sure it is in the "
                  "same folder as app.py and committed to your repo.")
        st.stop()

    return pd.read_csv(csv_path)

model = load_model()
df    = load_data()

# ─── NAVBAR ────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <span style="font-size:1.6rem">⛽</span>
    <span class="navbar-title">Fuel Consumption Analyzer</span>
    <span class="navbar-subtitle">India Automobile Fuel & Emissions Dashboard</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Navigation")
    page = st.radio(
        "", ["🏠  Home", "📊  EDA Charts", "🤖  Predict CO₂", "📁  Bulk Scanner"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.76rem; color:rgba(255,255,255,0.45); line-height:1.8;'>
    <b style='color:rgba(255,255,255,0.7)'>📘 Unit Guide</b><br>
    <b>km/L</b> = Kilometres per Litre<br>
    <b>L/100 km</b> = Litres used per 100 km<br>
    <b>g/km</b> = Grams of CO₂ per km<br>
    <b>mpg</b> = Miles per gallon<br>
    &nbsp;&nbsp;→ Lower L/100km = Better mileage<br><br>
    <b style='color:rgba(255,255,255,0.7)'>🚘 Vehicle Class</b><br>
    Compact/Subcompact = Hatchback<br>
    Mid-size = Family sedan (Dzire, City)<br>
    SUV = Fortuner, Creta type<br>
    Pickup Truck = Bolero Camper type<br>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":

    avg_co2  = df['EMISSIONS'].mean()
    avg_l100 = df['COMB (L/100 km)'].mean()

    cards = [
        ("🚗", f"{len(df):,}", "Total Vehicles"),
        ("📅", "2000-2022", "Years Covered"),
        ("🏭", str(df['MAKE'].nunique()), "Car Brands"),
        ("💨", f"{avg_co2:.0f} g/km", "Avg CO₂"),
        ("⛽", f"{avg_l100:.1f} L/100km", "Avg Fuel Used"),
    ]
    cols = st.columns(5)
    for col, (icon, val, label) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-heading'>📋 Dataset Preview</div>", unsafe_allow_html=True)
    show_cols = ['YEAR','MAKE','MODEL','VEHICLE CLASS','ENGINE SIZE','CYLINDERS','FUEL','EMISSIONS']
    st.dataframe(
        df[show_cols].head(20).rename(columns={
            'ENGINE SIZE':'Engine (L)',
            'VEHICLE CLASS':'Vehicle Type',
            'EMISSIONS':'CO₂ (g/km)'
        }),
        use_container_width=True, height=320
    )

    st.markdown("<div class='section-heading'>📐 Key Statistics</div>", unsafe_allow_html=True)
    stats = df[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','HWY (L/100 km)','COMB (L/100 km)','EMISSIONS']].describe().round(2)
    stats.columns = ['Engine (L)','Cylinders','City (L/100km)','Highway (L/100km)','Combined (L/100km)','CO₂ (g/km)']
    st.dataframe(stats, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>How to read fuel consumption in Indian terms:</b><br>
    If a car shows <b>8 L/100 km</b> → it uses 8 litres to travel 100 km → mileage = <b>12.5 km/L</b><br>
    Indian cars like Maruti Swift ≈ 6.5 L/100 km | SUVs like Creta ≈ 9-11 L/100 km | Trucks ≈ 13+ L/100 km
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA CHARTS
# ══════════════════════════════════════════════════════════════
elif page == "📊  EDA Charts":

    st.markdown("<h2 style='color:white; font-weight:700;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Visual insights from 22,000+ vehicle records (2000-2022)</p>", unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor':   '#1a1a2e',
        'axes.edgecolor':   '#444466',
        'axes.labelcolor':  '#ccccdd',
        'xtick.color':      '#ccccdd',
        'ytick.color':      '#ccccdd',
        'text.color':       '#ccccdd',
        'grid.color':       '#2a2a4a',
        'grid.linestyle':   '--',
        'grid.alpha':       0.5,
    })

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Relationships", "🏷️ Categories", "🗺️ Heatmap"])

    # ── TAB 1: DISTRIBUTIONS ──────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            n, bins, patches = ax.hist(df['EMISSIONS'], bins=45, edgecolor='none')
            cm = plt.colormaps['cool']
            col_scale = (n - n.min()) / (n.max() - n.min() + 1e-9)
            for c_val, patch in zip(col_scale, patches):
                patch.set_facecolor(cm(c_val))
            ax.set_title('CO₂ Emissions Distribution', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('CO₂ Emissions (g/km)')
            ax.set_ylabel('Number of Vehicles')
            ax.axvline(df['EMISSIONS'].mean(), color='#f6e05e', linestyle='--', linewidth=1.5,
                       label=f"Avg: {df['EMISSIONS'].mean():.0f}")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>Most vehicles emit 200-300 g/km. Yellow line = average.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            n, bins, patches = ax.hist(df['COMB (L/100 km)'], bins=45, edgecolor='none')
            cm2 = plt.colormaps['plasma']
            col_scale2 = (n - n.min()) / (n.max() - n.min() + 1e-9)
            for c_val, patch in zip(col_scale2, patches):
                patch.set_facecolor(cm2(c_val))
            ax.set_title('Combined Fuel Consumption', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('L/100 km  (lower = better mileage)')
            ax.set_ylabel('Number of Vehicles')
            ax.axvline(df['COMB (L/100 km)'].mean(), color='#f6e05e', linestyle='--', linewidth=1.5,
                       label=f"Avg: {df['COMB (L/100 km)'].mean():.1f}")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>8 L/100km = 12.5 km/L mileage (like a Maruti Swift)</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            n, bins, patches = ax.hist(df['ENGINE SIZE'], bins=40, edgecolor='none')
            cm3 = plt.colormaps['viridis']
            col_scale3 = (n - n.min()) / (n.max() - n.min() + 1e-9)
            for c_val, patch in zip(col_scale3, patches):
                patch.set_facecolor(cm3(c_val))
            ax.set_title('Engine Size Distribution', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Engine Size (Litres)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>Indian cars mostly have 1.0-2.0L engines</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            cyl_counts = df['CYLINDERS'].value_counts().sort_index()
            bar_colors = ['#63b3ed','#4fd1c5','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7']
            bars = ax.bar(cyl_counts.index, cyl_counts.values,
                          color=(bar_colors * 3)[:len(cyl_counts)],
                          edgecolor='none', width=0.6)
            for bar, val in zip(bars, cyl_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cyl_counts.values)*0.01,
                        f'{val:,}', ha='center', va='bottom', fontsize=8, color='#ccc')
            ax.set_title('Vehicles by Number of Cylinders', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Number of Cylinders')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>4-cylinder engines dominate (like most Indian cars)</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: RELATIONSHIPS ───────────────────────────────────
    with tab2:
        sample = df.sample(min(3000, len(df)), random_state=1)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sc = ax.scatter(sample['ENGINE SIZE'], sample['EMISSIONS'],
                            c=sample['CYLINDERS'], cmap='cool',
                            alpha=0.4, s=12, edgecolors='none')
            fig.colorbar(sc, ax=ax, label='Cylinders')
            ax.set_title('Engine Size vs CO₂ Emissions', fontsize=12, fontweight='bold')
            ax.set_xlabel('Engine Size (L)')
            ax.set_ylabel('CO₂ Emissions (g/km)')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>Bigger engine → more CO₂. Color shows cylinder count.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(sample['FUEL CONSUMPTION'], sample['HWY (L/100 km)'],
                       alpha=0.3, s=10, color='#4fd1c5', edgecolors='none')
            ax.set_title('City vs Highway Fuel Consumption', fontsize=12, fontweight='bold')
            ax.set_xlabel('City Driving (L/100 km)')
            ax.set_ylabel('Highway Driving (L/100 km)')
            ax.grid(True, alpha=0.3)
            lim = [3, 25]
            ax.plot(lim, lim, '--', color='#f6e05e', linewidth=1, alpha=0.7, label='City = Highway')
            ax.legend(fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>Highway driving always uses less fuel than city driving</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 4))
        palette = ['#63b3ed','#4fd1c5','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7']
        cyl_vals = sorted(df['CYLINDERS'].unique())
        bp_data = [df[df['CYLINDERS'] == c]['EMISSIONS'].values for c in cyl_vals]
        bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#666688'),
                        capprops=dict(color='#666688'),
                        flierprops=dict(marker='.', color='#555577', markersize=3))
        for patch, color in zip(bp['boxes'], (palette * 3)[:len(cyl_vals)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(cyl_vals)
        ax.set_title('CO₂ Emissions by Number of Cylinders', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Cylinders')
        ax.set_ylabel('CO₂ Emissions (g/km)')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>More cylinders = higher CO₂. 4-cylinder cars (most Indian cars) are in the middle range.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: CATEGORIES ─────────────────────────────────────
    with tab3:
        fuel_labels = {'X':'Regular Petrol','Z':'Premium Petrol','D':'Diesel','E':'Ethanol','N':'CNG/LPG'}
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fuel_counts = df['FUEL'].value_counts()
            fuel_counts.index = [fuel_labels.get(i, i) for i in fuel_counts.index]
            colors_pie = ['#63b3ed','#f6ad55','#68d391','#b794f4','#fc8181']
            fig, ax = plt.subplots(figsize=(6, 5))
            wedges, texts, autotexts = ax.pie(
                fuel_counts.values,
                labels=fuel_counts.index,
                autopct='%1.1f%%',
                colors=colors_pie[:len(fuel_counts)],
                startangle=140,
                pctdistance=0.82,
                wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2)
            )
            for t in texts:     t.set_fontsize(9);  t.set_color('#ccc')
            for t in autotexts: t.set_fontsize(8);  t.set_color('white'); t.set_fontweight('bold')
            ax.set_title('Fuel Type Distribution', fontsize=12, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>Regular Petrol (X) is most common — same as India</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fuel_co2 = df.groupby('FUEL')['EMISSIONS'].mean()
            fuel_co2.index = [fuel_labels.get(i, i) for i in fuel_co2.index]
            fuel_co2 = fuel_co2.sort_values()
            bar_colors = ['#68d391' if v < fuel_co2.mean() else '#fc8181' for v in fuel_co2.values]
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(fuel_co2.index, fuel_co2.values, color=bar_colors, edgecolor='none', width=0.5)
            for bar, val in zip(bars, fuel_co2.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{val:.0f}', ha='center', fontsize=9, color='#ccc')
            ax.axhline(fuel_co2.mean(), color='#f6e05e', linestyle='--', linewidth=1.5, label='Average')
            ax.legend(fontsize=9)
            ax.set_title('Avg CO₂ by Fuel Type (g/km)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Avg CO₂ (g/km)')
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>Green = below average. CNG has lowest CO₂ emissions.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 4))
        yearly = df.groupby('YEAR')['EMISSIONS'].mean()
        ax.fill_between(yearly.index, yearly.values, alpha=0.2, color='#63b3ed')
        ax.plot(yearly.index, yearly.values, marker='o', color='#63b3ed',
                linewidth=2.5, markersize=5, markerfacecolor='white')
        ax.set_title('Average CO₂ Emissions Trend (2000-2022)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Avg CO₂ (g/km)')
        ax.grid(True, alpha=0.3)
        ax.annotate(f'Peak\n{yearly.max():.0f}g', xy=(yearly.idxmax(), yearly.max()),
                    xytext=(yearly.idxmax()-2, yearly.max()+5),
                    fontsize=8, color='#fc8181', arrowprops=dict(arrowstyle='->', color='#fc8181'))
        ax.annotate(f'Low\n{yearly.min():.0f}g', xy=(yearly.idxmin(), yearly.min()),
                    xytext=(yearly.idxmin()+1, yearly.min()-10),
                    fontsize=8, color='#68d391', arrowprops=dict(arrowstyle='->', color='#68d391'))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>CO₂ emissions have reduced over the years due to better engine technology</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 4: HEATMAP ────────────────────────────────────────
    with tab4:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        cols_for_corr = ['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION',
                         'HWY (L/100 km)','COMB (L/100 km)','COMB (mpg)','EMISSIONS']
        rename_map = {
            'ENGINE SIZE':'Engine (L)',
            'CYLINDERS':'Cylinders',
            'FUEL CONSUMPTION':'City (L/100km)',
            'HWY (L/100 km)':'Hwy (L/100km)',
            'COMB (L/100 km)':'Comb (L/100km)',
            'COMB (mpg)':'Mileage (mpg)',
            'EMISSIONS':'CO₂ (g/km)'
        }
        corr_df = df[cols_for_corr].rename(columns=rename_map).corr()
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    ax=ax, linewidths=0.5, linecolor='#1a1a2e',
                    cbar_kws={'shrink': 0.8}, annot_kws={"size": 10})
        ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=15)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("""
        <p style='color:rgba(255,255,255,0.45); font-size:0.76rem; margin-top:6px;'>
        🔴 Dark red = strong positive link &nbsp;|&nbsp; 🟢 Dark green = strong negative link<br>
        CO₂ is strongly linked to engine size, cylinders, and fuel consumption
        </p>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT CO₂
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Predict CO₂":

    st.markdown("<h2 style='color:white; font-weight:700;'>🤖 Predict CO₂ Emissions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Enter your vehicle details to get an estimated CO₂ emission value</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>How to fill this form (Indian Reference):</b><br>
    • <b>City Fuel Consumption</b> = Litres used per 100 km in city driving (e.g., Swift city ≈ 8 L/100km)<br>
    • <b>Highway Consumption</b> = Litres per 100 km on highway (e.g., Swift highway ≈ 5.5 L/100km)<br>
    • <b>Combined</b> = Average of both → mileage = 100 / Combined<br>
    • <b>mpg</b> = miles per gallon — used in USA. 1 km/L ≈ 2.35 mpg
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-heading'>🔧 Engine Details</div>", unsafe_allow_html=True)
        engine_size = st.number_input(
            "Engine Size (Litres) — e.g. Swift = 1.2, Creta = 1.5",
            min_value=0.8, max_value=8.4, value=1.5, step=0.1
        )
        cylinders = st.selectbox(
            "Number of Cylinders — most Indian cars have 4",
            options=[3, 4, 5, 6, 8, 10, 12, 16], index=1
        )
        year = st.number_input("Vehicle Model Year", min_value=2000, max_value=2022, value=2018, step=1)
        vehicle_age = 2024 - year

    with col2:
        st.markdown("<div class='section-heading'>⛽ Fuel Consumption</div>", unsafe_allow_html=True)
        city_fc = st.slider(
            "City Driving (L/100 km) — lower is better",
            min_value=3.0, max_value=30.0, value=10.0, step=0.5,
            help="Swift city ≈ 8 | Creta city ≈ 12 | Truck city ≈ 16"
        )
        st.markdown(f"<p style='color:#68d391; font-size:0.82rem; margin-top:-10px;'>↳ City mileage: <b>{100/city_fc:.1f} km/L</b></p>", unsafe_allow_html=True)

        hwy_fc = st.slider(
            "Highway Driving (L/100 km)",
            min_value=3.0, max_value=20.0, value=7.0, step=0.5,
            help="Swift highway ≈ 5.5 | Creta highway ≈ 9"
        )
        st.markdown(f"<p style='color:#68d391; font-size:0.82rem; margin-top:-10px;'>↳ Highway mileage: <b>{100/hwy_fc:.1f} km/L</b></p>", unsafe_allow_html=True)

        comb_fc = round((city_fc + hwy_fc) / 2, 1)
        comb_mpg = int(235.21 / comb_fc)

        st.markdown(f"""
        <div style='background:rgba(99,179,237,0.1); border-radius:10px; padding:10px 14px; margin-top:6px;'>
            <span style='color:rgba(255,255,255,0.6); font-size:0.78rem;'>Auto-calculated Combined</span><br>
            <span style='color:#63b3ed; font-weight:700; font-size:1rem;'>{comb_fc} L/100 km</span>
            <span style='color:rgba(255,255,255,0.4); font-size:0.78rem;'> = {100/comb_fc:.1f} km/L = {comb_mpg} mpg</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='section-heading'>🚗 Vehicle Info</div>", unsafe_allow_html=True)
        fuel_options = {
            "Regular Petrol (Maruti, Hyundai)": 3,
            "Premium Petrol (BMW, Audi)": 4,
            "Diesel (Innova, Fortuner, Bolero)": 0,
            "Ethanol / E85 (Flex-fuel)": 1,
            "CNG / Natural Gas (Lowest CO₂)": 2,
        }
        fuel_label = st.selectbox("Fuel Type", list(fuel_options.keys()))
        fuel_enc   = fuel_options[fuel_label]

        vehicle_class_options = {
            "Compact / Hatchback (Swift, i20, Baleno)": 0,
            "Subcompact (Alto, S-Presso, WagonR)": 14,
            "Mid-size Sedan (City, Verna, Ciaz)": 2,
            "Full-size Sedan (Camry, Accord, Octavia)": 1,
            "SUV (Creta, Brezza, Nexon)": 15,
            "SUV Standard (Fortuner, Scorpio)": 17,
            "Minivan / MPV (Innova, Ertiga)": 4,
            "Pickup Truck Standard (Bolero Camper)": 6,
        }
        class_label = st.selectbox("Vehicle Class / Type", list(vehicle_class_options.keys()))
        class_enc   = vehicle_class_options[class_label]

        trans_options = {
            "Automatic (AT)": 0,
            "Automated Manual (AMT)": 1,
            "Auto with Select Shift (AT+Paddle)": 2,
            "CVT (Honda City CVT)": 3,
            "Manual (MT)": 4,
        }
        trans_label = st.selectbox("Transmission Type", list(trans_options.keys()))
        trans_enc   = trans_options[trans_label]

    city_hwy_avg   = (city_fc + hwy_fc) / 2
    engine_per_cyl = engine_size / cylinders
    eff_score      = 100 / comb_fc

    input_row = pd.DataFrame([{
        "ENGINE SIZE": engine_size,
        "CYLINDERS": cylinders,
        "FUEL CONSUMPTION": city_fc,
        "HWY (L/100 km)": hwy_fc,
        "COMB (L/100 km)": comb_fc,
        "COMB (mpg)": comb_mpg,
        "CITY_HWY_AVG": city_hwy_avg,
        "ENGINE_PER_CYL": engine_per_cyl,
        "EFF_SCORE": eff_score,
        "VEHICLE_AGE": vehicle_age,
        "FUEL_ENC": fuel_enc,
        "CLASS_ENC": class_enc,
        "TRANS_ENC": trans_enc,
    }])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict CO₂ Emissions", use_container_width=True)

    if predict_btn:
        result = model.predict(input_row)[0]
        result = max(83.0, result)
        kmpl   = 100 / comb_fc

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-value'>{result:.0f}</div>
                <div class='result-label'>g/km CO₂ Emitted</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='background:rgba(99,179,237,0.12); border:1px solid rgba(99,179,237,0.3);
                        border-radius:16px; padding:20px; text-align:center;'>
                <div style='font-size:2.1rem; font-weight:700; color:#63b3ed;'>{kmpl:.1f}</div>
                <div style='color:rgba(255,255,255,0.6); font-size:0.85rem; margin-top:6px;'>km/L Mileage</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            avg_co2 = df['EMISSIONS'].mean()
            diff    = result - avg_co2
            sign    = "▲" if diff > 0 else "▼"
            color   = "#fc8181" if diff > 0 else "#68d391"
            st.markdown(f"""
            <div style='background:rgba(246,173,85,0.1); border:1px solid rgba(246,173,85,0.3);
                        border-radius:16px; padding:20px; text-align:center;'>
                <div style='font-size:1.7rem; font-weight:700; color:{color};'>{sign}{abs(diff):.0f}</div>
                <div style='color:rgba(255,255,255,0.6); font-size:0.8rem; margin-top:6px;'>vs Dataset Avg<br>({avg_co2:.0f} g/km)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if result < 150:
            badge_class, label, desc = "badge-green",  "🌿 Excellent — Very Low Emissions", "This vehicle is extremely eco-friendly. Better than most CNG vehicles."
        elif result < 200:
            badge_class, label, desc = "badge-blue",   "✅ Good — Low Emissions",           "Below average emissions. Similar to a small hatchback like Maruti Swift."
        elif result < 250:
            badge_class, label, desc = "badge-yellow", "⚠️ Average — Moderate Emissions",   "Around average. Common for mid-size sedans and small SUVs like Creta."
        elif result < 300:
            badge_class, label, desc = "badge-orange", "🟠 High — Above Average Emissions",  "Higher than average. Typical for large SUVs like Fortuner or Scorpio."
        else:
            badge_class, label, desc = "badge-red",    "🔴 Very High — Heavy Emissions",     "Very high CO₂. Common in large trucks, V8/V10 engine vehicles."

        st.markdown(f"<span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:rgba(255,255,255,0.55); font-size:0.82rem; margin-top:8px;'>{desc}</p>", unsafe_allow_html=True)

        pct = min(int((result - 83) / (608 - 83) * 100), 100)
        bar_color = "#68d391" if pct < 30 else "#f6ad55" if pct < 60 else "#fc8181"
        st.markdown(f"""
        <div style='margin-top:16px;'>
            <div style='color:rgba(255,255,255,0.5); font-size:0.76rem; margin-bottom:6px;'>
                CO₂ Scale: 83 g/km (cleanest) → 608 g/km (highest in dataset)
            </div>
            <div style='background:rgba(255,255,255,0.08); border-radius:8px; height:14px; overflow:hidden;'>
                <div style='width:{pct}%; height:100%; background:{bar_color}; border-radius:8px;'></div>
            </div>
            <div style='color:rgba(255,255,255,0.4); font-size:0.74rem; margin-top:4px;'>
                Your vehicle is at the {pct}th percentile of all vehicles in the dataset
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — BULK SCANNER
# ══════════════════════════════════════════════════════════════
elif page == "📁  Bulk Scanner":

    st.markdown("<h2 style='color:white; font-weight:700;'>📁 Bulk CSV Scanner</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Upload any CSV or Excel file with car data — map your columns once, predict CO₂ for every row</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>How this works:</b><br>
    1. Upload any file — column names don't need to match this dataset<br>
    2. The app guesses which of your columns mean "Engine Size", "Cylinders", etc.<br>
    3. Fix any wrong guesses using the dropdowns<br>
    4. Click Predict — get a CO₂ + mileage column added to every row, downloadable as CSV
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.markdown("<div class='section-heading'>📋 Your Uploaded Data</div>", unsafe_allow_html=True)
        st.dataframe(user_df.head(10), use_container_width=True)
        st.caption(f"{len(user_df):,} rows × {len(user_df.columns)} columns detected")

        user_cols = user_df.columns.tolist()

        required_fields = {
            "engine_size": ("Engine Size (L)",            ["engine", "size", "litre", "liter", "cc", "displacement"]),
            "cylinders":   ("Cylinders",                  ["cylinder", "cyl"]),
            "city_fc":     ("City Fuel Consumption (L/100km)", ["city", "urban"]),
            "hwy_fc":      ("Highway Fuel Consumption (L/100km)", ["hwy", "highway", "hw"]),
            "comb_fc":     ("Combined Fuel Consumption (L/100km)", ["comb", "combined", "average", "avg"]),
            "year":        ("Model Year",                 ["year", "yr", "model year"]),
            "fuel":        ("Fuel Type",                  ["fuel", "petrol", "diesel"]),
            "vclass":      ("Vehicle Class",               ["class", "category", "type", "segment"]),
            "trans":       ("Transmission",                ["trans", "gearbox", "gear"]),
        }

        def guess_column(keywords, columns):
            best_col, best_score = None, 0
            for col in columns:
                col_lower = col.lower()
                score = sum(1 for kw in keywords if kw in col_lower)
                if score > best_score:
                    best_score, best_col = score, col
            return best_col if best_score > 0 else None

        st.markdown("<div class='section-heading'>🔗 Column Mapping</div>", unsafe_allow_html=True)
        st.caption("Auto-matched where possible — please review and correct if needed. Select 'None' to skip optional fields.")

        col_options = ["— None / Not Available —"] + user_cols
        mapping = {}

        map_col1, map_col2 = st.columns(2)
        field_items = list(required_fields.items())
        half = len(field_items) // 2 + 1

        for i, (key, (label, keywords)) in enumerate(field_items):
            guess = guess_column(keywords, user_cols)
            default_index = col_options.index(guess) if guess in col_options else 0
            target_col = map_col1 if i < half else map_col2
            with target_col:
                mapping[key] = st.selectbox(label, options=col_options, index=default_index, key=f"map_{key}")

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Bulk Prediction", use_container_width=True)

        if run_btn:
            work = pd.DataFrame()
            missing_required = []

            if mapping["engine_size"] != "— None / Not Available —":
                work["ENGINE SIZE"] = pd.to_numeric(user_df[mapping["engine_size"]], errors="coerce")
            else:
                missing_required.append("Engine Size")

            if mapping["cylinders"] != "— None / Not Available —":
                work["CYLINDERS"] = pd.to_numeric(user_df[mapping["cylinders"]], errors="coerce")
            else:
                missing_required.append("Cylinders")

            if missing_required:
                st.error(f"⚠️ These required fields are missing: {', '.join(missing_required)}. "
                          "Predictions need at least Engine Size and Cylinders to be meaningful.")
                st.stop()

            has_city = mapping["city_fc"] != "— None / Not Available —"
            has_hwy  = mapping["hwy_fc"]  != "— None / Not Available —"
            has_comb = mapping["comb_fc"] != "— None / Not Available —"

            if has_city:
                work["CITY_FC"] = pd.to_numeric(user_df[mapping["city_fc"]], errors="coerce")
            if has_hwy:
                work["HWY_FC"] = pd.to_numeric(user_df[mapping["hwy_fc"]], errors="coerce")
            if has_comb:
                work["COMB_FC"] = pd.to_numeric(user_df[mapping["comb_fc"]], errors="coerce")

            if not has_comb and has_city and has_hwy:
                work["COMB_FC"] = (work["CITY_FC"] + work["HWY_FC"]) / 2
            elif not has_comb and has_city:
                work["COMB_FC"] = work["CITY_FC"]
            elif not has_comb and has_hwy:
                work["COMB_FC"] = work["HWY_FC"]
            elif not has_comb:
                work["COMB_FC"] = 4.0 + work["ENGINE SIZE"] * 1.8

            if not has_city:
                work["CITY_FC"] = work["COMB_FC"] * 1.15
            if not has_hwy:
                work["HWY_FC"] = work["COMB_FC"] * 0.85

            work["COMB_MPG"] = 235.21 / work["COMB_FC"]

            if mapping["year"] != "— None / Not Available —":
                work["YEAR"] = pd.to_numeric(user_df[mapping["year"]], errors="coerce").fillna(2015)
            else:
                work["YEAR"] = 2015
            work["VEHICLE_AGE"] = 2024 - work["YEAR"]

            fuel_map = {
                "d": 0, "diesel": 0,
                "e": 1, "ethanol": 1, "e85": 1,
                "n": 2, "cng": 2, "natural gas": 2, "lpg": 2,
                "x": 3, "petrol": 3, "regular": 3, "gasoline": 3,
                "z": 4, "premium": 4,
            }
            if mapping["fuel"] != "— None / Not Available —":
                raw_fuel = user_df[mapping["fuel"]].astype(str).str.lower().str.strip()
                work["FUEL_ENC"] = raw_fuel.map(fuel_map).fillna(3).astype(int)
            else:
                work["FUEL_ENC"] = 3

            class_map = {
                'compact': 0, 'full-size': 1, 'full size': 1, 'mid-size': 2, 'mid size': 2, 'sedan': 2,
                'minicompact': 3, 'minivan': 4, 'mpv': 4,
                'pickup truck - small': 5, 'pickup truck - standard': 6, 'pickup': 6, 'truck': 6,
                'special purpose vehicle': 9,
                'station wagon - mid-size': 10, 'station wagon - small': 11, 'wagon': 11,
                'subcompact': 14, 'hatchback': 14,
                'suv': 15, 'suv - small': 16, 'suv - standard': 17,
                'two-seater': 20, 'van - cargo': 21, 'van - passenger': 22, 'van': 22,
            }
            if mapping["vclass"] != "— None / Not Available —":
                raw_class = user_df[mapping["vclass"]].astype(str).str.lower().str.strip()
                work["CLASS_ENC"] = raw_class.map(class_map).fillna(2).astype(int)
            else:
                work["CLASS_ENC"] = 2

            trans_map = {'a': 0, 'automatic': 0, 'am': 1, 'amt': 1, 'as': 2, 'av': 3, 'cvt': 3, 'm': 4, 'manual': 4}
            if mapping["trans"] != "— None / Not Available —":
                raw_trans = user_df[mapping["trans"]].astype(str).str.lower().str.strip()
                def map_trans(val):
                    if val in trans_map:
                        return trans_map[val]
                    for key, code in trans_map.items():
                        if val.startswith(key):
                            return code
                    return 0
                work["TRANS_ENC"] = raw_trans.apply(map_trans)
            else:
                work["TRANS_ENC"] = 0

            work["CITY_HWY_AVG"]   = (work["CITY_FC"] + work["HWY_FC"]) / 2
            work["ENGINE_PER_CYL"] = work["ENGINE SIZE"] / work["CYLINDERS"].replace(0, 1)
            work["EFF_SCORE"]      = 100 / work["COMB_FC"].replace(0, np.nan)

            before_rows = len(work)
            work = work.dropna(subset=["ENGINE SIZE", "CYLINDERS", "COMB_FC"])
            dropped = before_rows - len(work)

            if len(work) == 0:
                st.error("⚠️ No valid rows left after cleaning. Please check that Engine Size and Cylinders columns contain numbers.")
                st.stop()

            X_bulk = work[[
                "ENGINE SIZE", "CYLINDERS", "CITY_FC", "HWY_FC", "COMB_FC", "COMB_MPG",
                "CITY_HWY_AVG", "ENGINE_PER_CYL", "EFF_SCORE", "VEHICLE_AGE",
                "FUEL_ENC", "CLASS_ENC", "TRANS_ENC"
            ]].fillna(0)

            X_bulk = X_bulk.rename(columns={
                "CITY_FC":  "FUEL CONSUMPTION",
                "HWY_FC":   "HWY (L/100 km)",
                "COMB_FC":  "COMB (L/100 km)",
                "COMB_MPG": "COMB (mpg)",
            })

            X_bulk = X_bulk[[
                "ENGINE SIZE", "CYLINDERS", "FUEL CONSUMPTION", "HWY (L/100 km)",
                "COMB (L/100 km)", "COMB (mpg)", "CITY_HWY_AVG",
                "ENGINE_PER_CYL", "EFF_SCORE", "VEHICLE_AGE",
                "FUEL_ENC", "CLASS_ENC", "TRANS_ENC"
            ]]

            predictions = model.predict(X_bulk)
            predictions = np.clip(predictions, 83, 650)

            result_df = user_df.loc[work.index].copy()
            result_df["Predicted CO2 (g/km)"] = predictions.round(1)
            result_df["Mileage (km/L)"] = (100 / work["COMB_FC"]).round(2)

            def rate(co2):
                if co2 < 150: return "🌿 Excellent"
                elif co2 < 200: return "✅ Good"
                elif co2 < 250: return "⚠️ Average"
                elif co2 < 300: return "🟠 High"
                else: return "🔴 Very High"

            result_df["Emission Rating"] = result_df["Predicted CO2 (g/km)"].apply(rate)

            if dropped > 0:
                st.warning(f"⚠️ Skipped {dropped} row(s) with missing/invalid Engine Size, Cylinders, or Fuel Consumption data.")

            st.success(f"✅ Predicted CO₂ emissions for {len(result_df):,} vehicles")

            cards = [
                ("📊", f"{len(result_df):,}", "Vehicles Scanned"),
                ("💨", f"{result_df['Predicted CO2 (g/km)'].mean():.0f} g/km", "Avg Predicted CO₂"),
                ("⛽", f"{result_df['Mileage (km/L)'].mean():.1f} km/L", "Avg Mileage"),
                ("🌿", f"{(result_df['Predicted CO2 (g/km)'] < 200).sum()}", "Eco-Friendly Vehicles"),
            ]
            scols = st.columns(4)
            for c, (icon, val, label) in zip(scols, cards):
                c.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-heading'>📥 Results</div>", unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, height=420)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_out,
                file_name="co2_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("<div class='section-heading'>📈 Prediction Distribution</div>", unsafe_allow_html=True)
            plt.rcParams.update({
                'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e',
                'axes.edgecolor': '#444466', 'axes.labelcolor': '#ccccdd',
                'xtick.color': '#ccccdd', 'ytick.color': '#ccccdd',
                'text.color': '#ccccdd', 'grid.color': '#2a2a4a',
            })
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(result_df["Predicted CO2 (g/km)"], bins=30, color="#63b3ed", edgecolor="none")
            ax.set_xlabel("Predicted CO₂ (g/km)")
            ax.set_ylabel("Number of Vehicles")
            ax.set_title("Distribution of Predicted CO₂ for Uploaded Vehicles", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    else:
        st.markdown("""
        <div style='text-align:center; padding:50px 20px; color:rgba(255,255,255,0.3);'>
            <div style='font-size:2.6rem;'>📁</div>
            <div style='font-size:0.95rem; margin-top:10px;'>Upload a CSV or Excel file above to get started</div>
        </div>
        """, unsafe_allow_html=True)
