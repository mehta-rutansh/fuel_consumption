import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fuel Consumption Analysis",
    page_icon="⛽",
    layout="wide"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); background-attachment: fixed; }
#MainMenu, footer { visibility: hidden; }

.navbar {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding: 16px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
}
.navbar-title { font-size: 1.3rem; font-weight: 700; color: #fff; }
.navbar-subtitle { font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-left: auto; }
@media (max-width: 640px) { .navbar-subtitle { margin-left: 0; width: 100%; } }

.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 16px 14px; text-align: center; min-height: 100px;
}
.metric-icon { font-size: 1.6rem; margin-bottom: 4px; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #63b3ed; }
.metric-label { font-size: 0.68rem; color: rgba(255,255,255,0.55); margin-top: 4px; text-transform: uppercase; }

.section-heading {
    font-size: 1.1rem; font-weight: 600; color: #fff;
    margin: 1.3rem 0 0.9rem; padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,179,237,0.4);
}
.info-box {
    background: rgba(99,179,237,0.1); border-left: 4px solid #63b3ed;
    border-radius: 0 10px 10px 0; padding: 12px 16px; margin: 10px 0;
    color: rgba(255,255,255,0.85); font-size: 0.82rem; line-height: 1.6;
}
.result-box {
    background: rgba(72,187,120,0.15); border: 1px solid rgba(72,187,120,0.4);
    border-radius: 16px; padding: 20px; text-align: center;
}
.result-value { font-size: 2.4rem; font-weight: 700; color: #68d391; }
.result-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 6px; }

.badge-green  { background:#276749; color:#9ae6b4; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-blue   { background:#2a4365; color:#90cdf4; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-yellow { background:#744210; color:#fbd38d; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-orange { background:#7b341e; color:#fbd38d; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-red    { background:#742a2a; color:#fc8181; padding:6px 16px; border-radius:20px; font-weight:600; font-size:0.82rem; }

section[data-testid="stSidebar"] { background: rgba(15,12,41,0.92) !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
.stSelectbox label, .stNumberInput label, .stSlider label { color: rgba(255,255,255,0.7) !important; font-size: 0.85rem !important; }
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important; color: white !important;
}
.stButton > button {
    background: linear-gradient(135deg, #63b3ed, #4299e1) !important; color: white !important;
    border: none !important; border-radius: 12px !important; padding: 12px 24px !important;
    font-weight: 600 !important; width: 100% !important;
    box-shadow: 0 4px 15px rgba(99,179,237,0.3) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: rgba(255,255,255,0.6) !important; font-size: 0.85rem !important; }
.stTabs [aria-selected="true"] { background: rgba(99,179,237,0.2) !important; color: #63b3ed !important; }

.stDataFrame { border-radius: 12px; overflow: hidden; }
div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05); border-radius: 10px; padding: 9px 12px;
    margin-bottom: 6px; display: block; border: 1px solid rgba(255,255,255,0.08);
}
.chart-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 16px; margin-bottom: 16px; }
hr { border-color: rgba(255,255,255,0.1) !important; }

@media (max-width: 640px) {
    .metric-value { font-size: 1.2rem; }
    .result-value { font-size: 1.9rem; }
    div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL & DATA (with zip auto-extract) ────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("fuel_model.pkl") and os.path.exists("fuel_model.zip"):
        with zipfile.ZipFile("fuel_model.zip", "r") as zf:
            zf.extractall(".")
    if not os.path.exists("fuel_model.pkl"):
        st.error("⚠️ Model file not found. Add fuel_model.pkl or fuel_model.zip to the repo.")
        st.stop()
    return pickle.load(open("fuel_model.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("fuel_engineered.xls")

model = load_model()
df    = load_data()

# ─── NAVBAR ────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <span style="font-size:1.6rem">⛽</span>
    <span class="navbar-title">Fuel Consumption Analysis for Automobiles</span>
    <span class="navbar-subtitle">Examining vehicle attributes & fuel efficiency</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Navigation")
    page = st.radio(
        "",
        ["🏠  Overview", "📊  EDA & Visualizations", "🏆  Efficiency Rankings",
         "🔗  Correlation Analysis", "🤖  Fuel Consumption Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.76rem; color:rgba(255,255,255,0.45); line-height:1.8;'>
    <b style='color:rgba(255,255,255,0.7)'>📘 Unit Guide</b><br>
    <b>L/100 km</b> = Litres used per 100 km<br>
    <b>km/L</b> = Kilometres per Litre<br>
    <b>g/km</b> = Grams of CO₂ per km<br>
    Lower L/100km = Better fuel efficiency
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    st.markdown("<h2 style='color:white; font-weight:700;'>⛽ Fuel Consumption Analysis for Automobiles</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    This project examines automobile attributes — engine size, cylinders, transmission type,
    vehicle class, fuel type, and CO₂ emissions — to understand their impact on
    <b>fuel efficiency</b>. The analysis covers data preprocessing, exploratory data analysis (EDA),
    correlation analysis, and a fuel consumption prediction model, helping manufacturers,
    policymakers, and consumers make informed decisions about fuel-efficient vehicles.
    </div>
    """, unsafe_allow_html=True)

    avg_fc  = df['COMB (L/100 km)'].mean()
    avg_co2 = df['EMISSIONS'].mean()

    cards = [
        ("🚗", f"{len(df):,}", "Total Vehicles"),
        ("📅", "2000-2022", "Years Covered"),
        ("🏭", str(df['MAKE'].nunique()), "Car Brands"),
        ("⛽", f"{avg_fc:.1f} L/100km", "Avg Fuel Consumption"),
        ("💨", f"{avg_co2:.0f} g/km", "Avg CO₂ Emission"),
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
    show_cols = ['YEAR','MAKE','MODEL','VEHICLE CLASS','ENGINE SIZE','CYLINDERS','FUEL','COMB (L/100 km)','EMISSIONS']
    st.dataframe(
        df[show_cols].head(20).rename(columns={
            'ENGINE SIZE':'Engine (L)', 'VEHICLE CLASS':'Vehicle Type',
            'COMB (L/100 km)':'Fuel Consumption (L/100km)', 'EMISSIONS':'CO₂ (g/km)'
        }),
        use_container_width=True, height=320
    )

    st.markdown("<div class='section-heading'>📐 Key Statistics</div>", unsafe_allow_html=True)
    stats = df[['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','HWY (L/100 km)','COMB (L/100 km)','EMISSIONS']].describe().round(2)
    stats.columns = ['Engine (L)','Cylinders','City (L/100km)','Highway (L/100km)','Combined (L/100km)','CO₂ (g/km)']
    st.dataframe(stats, use_container_width=True)

    st.markdown("<div class='section-heading'>🧪 Data Preprocessing Applied</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    ✅ Checked for missing values — none found in this dataset<br>
    ✅ Removed duplicate vehicle records<br>
    ✅ Standardized inconsistent text formatting in categorical columns<br>
    ✅ Verified data types for all numeric and categorical fields
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA & VISUALIZATIONS  (the core of the project)
# ══════════════════════════════════════════════════════════════
elif page == "📊  EDA & Visualizations":

    st.markdown("<h2 style='color:white; font-weight:700;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Bar charts, histograms, scatter plots, pie charts — visualizing fuel consumption trends across vehicle attributes</p>", unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e', 'axes.edgecolor': '#444466',
        'axes.labelcolor': '#ccccdd', 'xtick.color': '#ccccdd', 'ytick.color': '#ccccdd',
        'text.color': '#ccccdd', 'grid.color': '#2a2a4a', 'grid.linestyle': '--', 'grid.alpha': 0.5,
    })

    tab1, tab2, tab3 = st.tabs(["📈 Distributions (Histogram)", "🔵 Relationships (Scatter)", "📊 Categories (Bar / Pie)"])

    # ── TAB 1: Histograms ──────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.hist(df['COMB (L/100 km)'], bins=45, color='#63b3ed', edgecolor='none')
            ax.axvline(df['COMB (L/100 km)'].mean(), color='#f6e05e', linestyle='--', linewidth=1.5,
                       label=f"Avg: {df['COMB (L/100 km)'].mean():.1f}")
            ax.set_title('Fuel Consumption Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Combined Fuel Consumption (L/100 km)')
            ax.set_ylabel('Number of Vehicles')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem;'>Most vehicles consume 8-12 L/100km. Lower values = more efficient.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.hist(df['ENGINE SIZE'], bins=40, color='#68d391', edgecolor='none')
            ax.set_title('Engine Size Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Engine Size (Litres)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem;'>Most vehicles in the dataset have 1.5-3.5L engines.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.hist(df['EMISSIONS'], bins=50, color='#f6ad55', edgecolor='none')
        ax.set_title('CO₂ Emissions Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('CO₂ Emissions (g/km)')
        ax.set_ylabel('Number of Vehicles')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Scatter plots ───────────────────────────────────
    with tab2:
        sample = df.sample(min(3000, len(df)), random_state=1)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(sample['ENGINE SIZE'], sample['COMB (L/100 km)'], alpha=0.3, s=10, color='#63b3ed')
            ax.set_title('Engine Size vs Fuel Consumption', fontsize=12, fontweight='bold')
            ax.set_xlabel('Engine Size (L)')
            ax.set_ylabel('Fuel Consumption (L/100 km)')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem;'>Larger engines consistently consume more fuel per 100km.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(sample['FUEL CONSUMPTION'], sample['HWY (L/100 km)'], alpha=0.3, s=10, color='#4fd1c5')
            ax.set_title('City vs Highway Fuel Consumption', fontsize=12, fontweight='bold')
            ax.set_xlabel('City (L/100 km)')
            ax.set_ylabel('Highway (L/100 km)')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem;'>Highway driving uses less fuel than city driving.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.scatter(sample['COMB (L/100 km)'], sample['EMISSIONS'], alpha=0.3, s=10, color='#fc8181')
        ax.set_title('Fuel Consumption vs CO₂ Emissions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fuel Consumption (L/100 km)')
        ax.set_ylabel('CO₂ Emissions (g/km)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<p style='color:rgba(255,255,255,0.45); font-size:0.76rem;'>Vehicles that consume more fuel also emit more CO₂ — a near-linear relationship.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Bar / Pie ────────────────────────────────────────
    with tab3:
        fuel_labels = {'X':'Regular Petrol','Z':'Premium Petrol','D':'Diesel','E':'Ethanol','N':'CNG/LPG'}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fc = df['FUEL'].value_counts()
            fc.index = [fuel_labels.get(i, i) for i in fc.index]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.pie(fc.values, labels=fc.index, autopct='%1.1f%%', startangle=140,
                   colors=['#63b3ed','#f6ad55','#68d391','#b794f4','#fc8181'][:len(fc)],
                   wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2))
            ax.set_title('Fuel Type Distribution', fontsize=12, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            avg_by_fuel = df.groupby('FUEL')['COMB (L/100 km)'].mean()
            avg_by_fuel.index = [fuel_labels.get(i, i) for i in avg_by_fuel.index]
            avg_by_fuel = avg_by_fuel.sort_values()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(avg_by_fuel.index, avg_by_fuel.values, color='#63b3ed', edgecolor='none', width=0.5)
            ax.set_title('Avg Fuel Consumption by Fuel Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('L/100 km')
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        top = df.groupby('VEHICLE CLASS')['COMB (L/100 km)'].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.barh(top.index, top.values, color='#b794f4')
        ax.set_title('Avg Fuel Consumption by Vehicle Class (Top 15)', fontsize=12, fontweight='bold')
        ax.set_xlabel('L/100 km')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — EFFICIENCY RANKINGS
# (directly satisfies: "identifies vehicles with higher or lower
#  fuel efficiency")
# ══════════════════════════════════════════════════════════════
elif page == "🏆  Efficiency Rankings":

    st.markdown("<h2 style='color:white; font-weight:700;'>🏆 Fuel Efficiency Rankings</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Identifying vehicles with the highest and lowest fuel efficiency</p>", unsafe_allow_html=True)

    rank_by = st.selectbox("Rank by", ["MAKE", "VEHICLE CLASS", "FUEL", "TRANSMISSION"])

    avg_table = df.groupby(rank_by)['COMB (L/100 km)'].agg(['mean', 'count']).reset_index()
    avg_table.columns = [rank_by, 'Avg Fuel Consumption (L/100km)', 'Vehicle Count']
    avg_table = avg_table[avg_table['Vehicle Count'] >= 5]  # avoid tiny samples
    avg_table['Avg Fuel Consumption (L/100km)'] = avg_table['Avg Fuel Consumption (L/100km)'].round(2)

    most_efficient  = avg_table.sort_values('Avg Fuel Consumption (L/100km)').head(10)
    least_efficient = avg_table.sort_values('Avg Fuel Consumption (L/100km)', ascending=False).head(10)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-heading'>🌿 Most Fuel Efficient</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(most_efficient[rank_by], most_efficient['Avg Fuel Consumption (L/100km)'], color='#68d391')
        ax.invert_yaxis()
        ax.set_xlabel('L/100 km (lower = better)')
        ax.set_title(f'Top 10 Most Efficient ({rank_by.title()})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(most_efficient, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("<div class='section-heading'>🔴 Least Fuel Efficient</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(least_efficient[rank_by], least_efficient['Avg Fuel Consumption (L/100km)'], color='#fc8181')
        ax.invert_yaxis()
        ax.set_xlabel('L/100 km (higher = worse)')
        ax.set_title(f'Top 10 Least Efficient ({rank_by.title()})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(least_efficient, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='info-box'>
    💡 This ranking helps <b>consumers</b> pick efficient vehicles, <b>manufacturers</b> benchmark their
    models against competitors, and <b>policymakers</b> identify categories that may need stricter
    efficiency standards.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATION ANALYSIS
# (directly satisfies: "Correlation analysis... between variables
#  such as engine size and fuel consumption")
# ══════════════════════════════════════════════════════════════
elif page == "🔗  Correlation Analysis":

    st.markdown("<h2 style='color:white; font-weight:700;'>🔗 Correlation Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Strength of relationships between vehicle attributes and fuel consumption</p>", unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e', 'axes.edgecolor': '#444466',
        'axes.labelcolor': '#ccccdd', 'xtick.color': '#ccccdd', 'ytick.color': '#ccccdd', 'text.color': '#ccccdd',
    })

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    cols_for_corr = ['ENGINE SIZE','CYLINDERS','FUEL CONSUMPTION','HWY (L/100 km)','COMB (L/100 km)','EMISSIONS']
    rename_map = {
        'ENGINE SIZE':'Engine (L)', 'CYLINDERS':'Cylinders', 'FUEL CONSUMPTION':'City (L/100km)',
        'HWY (L/100 km)':'Hwy (L/100km)', 'COMB (L/100 km)':'Fuel Consumption (L/100km)', 'EMISSIONS':'CO₂ (g/km)'
    }
    corr_df = df[cols_for_corr].rename(columns=rename_map).corr()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                linewidths=0.5, linecolor='#1a1a2e', cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap — Vehicle Attributes vs Fuel Consumption', fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Numeric correlation table specifically vs fuel consumption
    st.markdown("<div class='section-heading'>📊 Correlation with Fuel Consumption</div>", unsafe_allow_html=True)
    corr_with_target = corr_df['Fuel Consumption (L/100km)'].drop('Fuel Consumption (L/100km)').sort_values(ascending=False)
    corr_table = corr_with_target.reset_index()
    corr_table.columns = ['Attribute', 'Correlation Coefficient']
    corr_table['Correlation Coefficient'] = corr_table['Correlation Coefficient'].round(3)
    corr_table['Strength'] = corr_table['Correlation Coefficient'].abs().apply(
        lambda v: "Very Strong" if v > 0.8 else "Strong" if v > 0.6 else "Moderate" if v > 0.4 else "Weak"
    )
    st.dataframe(corr_table, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>Engine size</b> and <b>cylinders</b> show a strong positive correlation with fuel consumption —
    bigger engines with more cylinders consume more fuel per 100 km. CO₂ emissions are very strongly
    correlated with fuel consumption since they are a direct byproduct of fuel burned.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — FUEL CONSUMPTION PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Fuel Consumption Predictor":

    st.markdown("<h2 style='color:white; font-weight:700;'>🤖 Fuel Consumption Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-top:-10px;'>Predicts Combined Fuel Consumption (L/100 km) from vehicle attributes</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    💡 This model takes vehicle attributes (engine size, cylinders, CO₂ emissions, fuel type, vehicle
    class, transmission, age) and predicts the expected <b>fuel consumption</b> in L/100 km —
    the central goal of this project.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-heading'>🔧 Engine Details</div>", unsafe_allow_html=True)
        engine_size = st.number_input("Engine Size (Litres)", min_value=0.8, max_value=8.4, value=2.0, step=0.1)
        cylinders   = st.selectbox("Number of Cylinders", [3, 4, 5, 6, 8, 10, 12, 16], index=1)
        year        = st.number_input("Vehicle Model Year", min_value=2000, max_value=2022, value=2018, step=1)
        vehicle_age = 2024 - year

    with col2:
        st.markdown("<div class='section-heading'>💨 Emissions</div>", unsafe_allow_html=True)
        emissions = st.slider("CO₂ Emissions (g/km)", min_value=83, max_value=608, value=220, step=1,
                               help="Used as an input attribute, since it correlates strongly with fuel consumption")
        st.markdown(f"<p style='color:rgba(255,255,255,0.5); font-size:0.8rem;'>Dataset average: {df['EMISSIONS'].mean():.0f} g/km</p>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='section-heading'>🚗 Vehicle Info</div>", unsafe_allow_html=True)
        fuel_options = {
            "Regular Petrol": 3, "Premium Petrol": 4, "Diesel": 0,
            "Ethanol / E85": 1, "CNG / Natural Gas": 2,
        }
        fuel_label = st.selectbox("Fuel Type", list(fuel_options.keys()))
        fuel_enc   = fuel_options[fuel_label]

        vehicle_class_options = {
            "Compact": 0, "Subcompact": 14, "Mid-size": 2, "Full-size": 1,
            "SUV": 15, "SUV Standard": 17, "Minivan": 4, "Pickup Truck Standard": 6,
        }
        class_label = st.selectbox("Vehicle Class", list(vehicle_class_options.keys()))
        class_enc   = vehicle_class_options[class_label]

        trans_options = {
            "Automatic (AT)": 0, "Automated Manual (AMT)": 1,
            "Auto with Select Shift": 2, "CVT": 3, "Manual (MT)": 4,
        }
        trans_label = st.selectbox("Transmission Type", list(trans_options.keys()))
        trans_enc   = trans_options[trans_label]

    input_row = pd.DataFrame([{
        "ENGINE SIZE": engine_size,
        "CYLINDERS": cylinders,
        "EMISSIONS": emissions,
        "VEHICLE_AGE": vehicle_age,
        "FUEL_ENC": fuel_enc,
        "CLASS_ENC": class_enc,
        "TRANS_ENC": trans_enc,
    }])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Predict Fuel Consumption", use_container_width=True):
        result = model.predict(input_row)[0]
        result = max(2.0, result)
        kmpl   = 100 / result

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-value'>{result:.1f}</div>
                <div class='result-label'>L/100 km Predicted</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='background:rgba(99,179,237,0.12); border:1px solid rgba(99,179,237,0.3);
                        border-radius:16px; padding:20px; text-align:center;'>
                <div style='font-size:2.1rem; font-weight:700; color:#63b3ed;'>{kmpl:.1f}</div>
                <div style='color:rgba(255,255,255,0.6); font-size:0.85rem; margin-top:6px;'>km/L Mileage</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            avg_fc = df['COMB (L/100 km)'].mean()
            diff   = result - avg_fc
            sign   = "▲" if diff > 0 else "▼"
            color  = "#fc8181" if diff > 0 else "#68d391"
            st.markdown(f"""
            <div style='background:rgba(246,173,85,0.1); border:1px solid rgba(246,173,85,0.3);
                        border-radius:16px; padding:20px; text-align:center;'>
                <div style='font-size:1.7rem; font-weight:700; color:{color};'>{sign}{abs(diff):.1f}</div>
                <div style='color:rgba(255,255,255,0.6); font-size:0.8rem; margin-top:6px;'>vs Dataset Avg<br>({avg_fc:.1f} L/100km)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if result < 7:
            badge_class, label = "badge-green",  "🌿 Excellent — Very Fuel Efficient"
        elif result < 9:
            badge_class, label = "badge-blue",   "✅ Good — Above Average Efficiency"
        elif result < 12:
            badge_class, label = "badge-yellow", "⚠️ Average Efficiency"
        elif result < 15:
            badge_class, label = "badge-orange", "🟠 Below Average Efficiency"
        else:
            badge_class, label = "badge-red",    "🔴 Poor Fuel Efficiency"
        st.markdown(f"<span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)
