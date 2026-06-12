import os
import pickle
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="Automatidata — NYC Taxi Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for a polished look
st.markdown("""
<style>
    /* Enforce modern typography structure globally */
    .main {
        background-color: #fcfaf2;
        color: #334e68;
        font-family: 'Inter', sans-serif;
    }
    
    /* Global heading overrides */
    h1, .title-text {
        font-family: 'Inter', sans-serif;
        color: #d99b26 !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #627d98 !important;
        font-size: 1.05rem !important;
        margin-top: 0px;
        margin-bottom: 30px;
        line-height: 1.5;
    }
    
    h2 {
        font-family: 'Inter', sans-serif;
        color: #d99b26 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        color: #334e68 !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    h4 {
        font-family: 'Inter', sans-serif;
        color: #334e68 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    /* Content text body */
    p, span, label, div {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    
    /* Custom metric card */
    .metric-card {
        background-color: #f5f2e9;
        border-top: 4px solid #d99b26;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(51,78,104,0.06);
        transition: transform 0.2s ease;
        border: 1px solid #e2ddcf;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #334e68;
        margin-top: 5px;
    }
    
    .metric-label {
        font-size: 0.78rem;
        color: #627d98;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
    }
    
    /* Prediction Verdict Card */
    .verdict-card {
        padding: 25px;
        border-radius: 12px;
        margin-top: 15px;
        border: 1px solid #e2ddcf;
        box-shadow: 0 6px 15px rgba(51,78,104,0.08);
    }
    
    .verdict-card.generous {
        background: linear-gradient(135deg, rgba(217,155,38,0.1), rgba(217,155,38,0.02));
        border-left: 6px solid #d99b26;
    }
    
    .verdict-card.standard {
        background: linear-gradient(135deg, rgba(98,125,152,0.1), rgba(98,125,152,0.02));
        border-left: 6px solid #627d98;
    }
    
    /* Domain Insight Box */
    .insight-box {
        background-color: #f5f2e9;
        border-left: 4px solid #d99b26;
        border-right: 1px solid #e2ddcf;
        border-top: 1px solid #e2ddcf;
        border-bottom: 1px solid #e2ddcf;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #486581;
    }
    
    /* Highlight text styling */
    .highlight-yellow {
        color: #d99b26;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib style for light theme matching the cream container background
plt.style.use('default')
sns.set_theme(style="whitegrid", rc={
    "grid.color": "#e2ddcf",
    "axes.facecolor": "#f5f2e9",
    "figure.facecolor": "#fcfaf2",
    "text.color": "#334e68",
    "axes.labelcolor": "#627d98",
    "xtick.color": "#627d98",
    "ytick.color": "#627d98",
    "font.sans-serif": ["Inter", "DejaVu Sans", "sans-serif"]
})

# ==========================================
# PATH RESOLUTIONS
# ==========================================
project_root = os.path.dirname(os.path.abspath(__file__))
regressor_path = os.path.join(project_root, "models", "fare_regressor.pkl")
classifier_path = os.path.join(project_root, "models", "tip_classifier.pkl")
route_stats_path = os.path.join(project_root, "data", "processed", "route_stats.json")
processed_data_path = os.path.join(project_root, "data", "processed", "processed_taxi_data.csv")

# ==========================================
# CACHED DATA LOADING
# ==========================================
@st.cache_data
def load_processed_data():
    if os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        return df
    return None

@st.cache_resource
def load_models():
    regressor = None
    classifier = None
    if os.path.exists(regressor_path):
        with open(regressor_path, "rb") as f:
            regressor = pickle.load(f)
    if os.path.exists(classifier_path):
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
    return regressor, classifier

@st.cache_data
def load_route_statistics():
    if os.path.exists(route_stats_path):
        with open(route_stats_path, "r") as f:
            return json.load(f)
    return None

# Load dataset and models
df_taxi = load_processed_data()
reg_model, clf_model = load_models()
route_stats = load_route_statistics()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown(
    '<div style="text-align: center;"><span style="font-size: 3.5rem;">🚕</span></div>', 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<h2 style="text-align: center; color: #F7C300; margin-top: 0px; font-weight: 800; letter-spacing: -0.5px;">Automatidata</h2>', 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<p style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: -15px;">NYC Taxi Churn & Tip Analytics</p>', 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Dashboard Overview (EDA)",
        "🔮 Trip Predictor (Dual-Stage AI)",
        "📈 Model Report & Performance"
    ]
)

st.sidebar.markdown("---")

# Sidebar system status info
if df_taxi is not None and reg_model is not None and clf_model is not None:
    st.sidebar.success("🟢 System Status: Active")
    st.sidebar.markdown(f"**Dataset size:** {df_taxi.shape[0]:,} credit card trips")
    st.sidebar.markdown(f"**Regressor:** XGBoost")
    st.sidebar.markdown(f"**Classifier:** Random Forest")
else:
    st.sidebar.error("🔴 System Status: Missing Components")

# ==========================================
# 1. TAB: DASHBOARD OVERVIEW (EDA)
# ==========================================
if menu == "📊 Dashboard Overview (EDA)":
    st.markdown('<h1 class="title-text">NYC Yellow Taxi Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Interactive statistics and exploratory analysis on trip distances, fares, and tipping behaviors.</p>', unsafe_allow_html=True)
    
    if df_taxi is None:
        st.warning("⚠️ Processed dataset not found. Please make sure data/processed/processed_taxi_data.csv exists.")
    else:
        # Filter Layout
        st.markdown("### 🔍 Filter Dataset")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            dist_filter = st.slider("Filter by Trip Distance (miles)", 0.0, float(df_taxi['trip_distance'].max()), (0.0, 15.0))
        with col_f2:
            hour_filter = st.slider("Filter by Hour of Day", 0, 23, (0, 23))
            
        # Apply filters
        filtered_df = df_taxi[
            (df_taxi['trip_distance'] >= dist_filter[0]) & 
            (df_taxi['trip_distance'] <= dist_filter[1]) &
            (df_taxi['tpep_pickup_datetime'].dt.hour >= hour_filter[0]) &
            (df_taxi['tpep_pickup_datetime'].dt.hour <= hour_filter[1])
        ]
        
        # KPI Section
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_trips = len(filtered_df)
        avg_fare = filtered_df['fare_amount'].mean() if total_trips > 0 else 0
        avg_tip = filtered_df['tip_amount'].mean() if total_trips > 0 else 0
        generous_rate = (filtered_df['generous'].mean() * 100) if total_trips > 0 else 0
        
        with kpi1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Total Filtered Trips</div>'
                f'<div class="metric-value">{total_trips:,}</div></div>', 
                unsafe_allow_html=True
            )
        with kpi2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Average Fare Amount</div>'
                f'<div class="metric-value">${avg_fare:.2f}</div></div>', 
                unsafe_allow_html=True
            )
        with kpi3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Average Tip Amount</div>'
                f'<div class="metric-value">${avg_tip:.2f}</div></div>', 
                unsafe_allow_html=True
            )
        with kpi4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Generous Tipper Rate</div>'
                f'<div class="metric-value">{generous_rate:.1f}%</div></div>', 
                unsafe_allow_html=True
            )
            
        st.markdown("---")
        st.markdown("### 📊 8 Visualisasi Utama Eksplorasi Data (EDA)")
        st.markdown("Grafik disajikan dengan resolusi tinggi (High-DPI) untuk ketajaman visual maksimal. Sumbu dan satuan telah disesuaikan dengan standar industri.")

        # Section 1: Karakteristik Perjalanan & Tarif
        with st.expander("📍 1. Karakteristik Perjalanan & Korelasi Tarif (Jarak vs. Biaya)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 1. Distribusi Jarak Perjalanan")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    sns.histplot(data=filtered_df, x='trip_distance', bins=30, kde=True, color='#d99b26', ax=ax, edgecolor='#f5f2e9', alpha=0.85)
                ax.set_xlabel("Trip Distance (miles)")
                ax.set_ylabel("Frequency (Count)")
                ax.set_title("Distribution of NYC Taxi Trip Distances", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Menunjukkan sebaran frekuensi jarak perjalanan taksi. Sebagian besar perjalanan terkonsentrasi pada jarak pendek di bawah 4 mil.")
            
            with col2:
                st.markdown("#### 2. Jarak Perjalanan vs. Biaya Tarif")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                sample_df = filtered_df.sample(min(len(filtered_df), 1500), random_state=42) if len(filtered_df) > 0 else filtered_df
                if len(sample_df) > 0:
                    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount', hue='generous', palette={0: '#8a92a6', 1: '#d99b26'}, alpha=0.6, ax=ax)
                    sns.regplot(data=sample_df, x='trip_distance', y='fare_amount', scatter=False, color='#334e68', ax=ax)
                ax.set_xlabel("Trip Distance (miles)")
                ax.set_ylabel("Fare Amount ($)")
                ax.set_title("Relationship: Trip Distance vs. Fare Amount", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Membuktikan korelasi positif linear yang sangat kuat antara jarak dan tarif dasar perjalanan taksi.")

        # Section 2: Tren Waktu & Tarif Bandara
        with st.expander("⏰ 2. Tren Berdasarkan Jam & Analisis Rate Code (Jadwal vs. Tipping)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 3. Rata-rata Tarif berdasarkan Jam")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    filtered_df['hour_extracted'] = filtered_df['tpep_pickup_datetime'].dt.hour
                    hourly_fare = filtered_df.groupby('hour_extracted')['fare_amount'].mean()
                    ax.plot(hourly_fare.index, hourly_fare.values, marker='o', color='#d99b26', linewidth=2, markersize=5)
                    ax.fill_between(hourly_fare.index, hourly_fare.values, alpha=0.15, color='#d99b26')
                    ax.set_xticks(range(0, 24, 2))
                    ax.set_xlim(0, 23)
                ax.set_xlabel("Hour of Day (24-Hour Format)")
                ax.set_ylabel("Average Fare Amount ($)")
                ax.set_title("Hourly Trend of Average Fare Amount", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Puncak tarif tertinggi terjadi di jam sibuk sore hari (jam 16.00-19.00) dan jam-jam larut malam.")
                
            with col2:
                st.markdown("#### 4. Rata-rata Tip berdasarkan Rate Code")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    ratecode_map = {1: 'Standard', 2: 'JFK', 3: 'Newark', 4: 'Nassau', 5: 'Negotiated', 6: 'Group'}
                    filtered_df['Ratecode_Name'] = filtered_df['RatecodeID'].map(ratecode_map)
                    rate_stats = filtered_df.groupby('Ratecode_Name')['tip_amount'].mean().sort_values(ascending=False)
                    sns.barplot(x=rate_stats.values, y=rate_stats.index, palette='YlOrBr_r', ax=ax, edgecolor='#f5f2e9')
                ax.set_xlabel("Average Tip Amount ($)")
                ax.set_ylabel("Rate Code Category")
                ax.set_title("Average Tip Amount by Rate Code", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Pemberian tip tertinggi terdapat pada rute ke bandara (JFK & Newark) karena tarif dasarnya yang tinggi.")

        # Section 3: Analisis Tipping & Kapasitas Penumpang
        with st.expander("👥 3. Perilaku Tipping Harian & Pengaruh Jumlah Penumpang", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 5. Persentase Tip Dermawan berdasarkan Hari")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    day_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    day_stats = filtered_df.groupby('day')['generous'].mean().reindex(day_order) * 100
                    bars = ax.bar(day_stats.index.str.capitalize(), day_stats.values, color='#d99b26', alpha=0.85, edgecolor='#d99b26', width=0.55)
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, color='#334e68', fontweight='bold')
                    ax.set_ylim(0, 100)
                ax.set_ylabel("Generous Tipper Rate (%)")
                ax.set_xlabel("Day of Week")
                ax.set_title("Generous Tipper Rate (%) by Day of Week", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Menunjukkan kestabilan persentase penumpang pemberi tip murah hati di kisaran 45% - 55% setiap harinya.")
                
            with col2:
                st.markdown("#### 6. Distribusi Tarif berdasarkan Jumlah Penumpang")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    sns.boxplot(data=filtered_df, x='passenger_count', y='fare_amount', color='#d99b26', ax=ax, fliersize=1)
                ax.set_xlabel("Passenger Count (People)")
                ax.set_ylabel("Fare Amount ($)")
                ax.set_title("Fare Amount Range by Passenger Count", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Menunjukkan rentang dan outlier tarif taksi untuk tiap kapasitas jumlah penumpang.")

        # Section 4: Pangsa Vendor & Korelasi Fitur
        with st.expander("🔌 4. Pangsa Pasar Vendor & Matriks Korelasi Fitur", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 7. Pangsa Perjalanan berdasarkan Vendor")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    vendor_counts = filtered_df['VendorID'].value_counts()
                    labels = ['VeriFone (Vendor 2)', 'Creative Mobile (Vendor 1)'][:len(vendor_counts)]
                    ax.pie(vendor_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#d99b26', '#8a92a6'], 
                           wedgeprops=dict(width=0.4, edgecolor='#f5f2e9'))
                ax.set_title("Trip Share Distribution by Taxi Vendor", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Menampilkan persentase pembagian pasar armada taksi antara dua vendor utama di New York City.")
                
            with col2:
                st.markdown("#### 8. Matriks Korelasi Fitur Utama")
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                if len(filtered_df) > 0:
                    corr_cols = ['trip_distance', 'duration', 'fare_amount', 'tip_amount', 'tip_percent']
                    corr_matrix = filtered_df[corr_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='YlOrBr', fmt='.2f', ax=ax, cbar=False, annot_kws={"size": 10, "weight": "bold"})
                ax.set_title("Correlation Matrix of Key Numerical Variables", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Korelasi statistik antar fitur numerik. Tarif memiliki korelasi yang sangat kuat dengan durasi dan jarak.")

# ==========================================
# 2. TAB: TRIP PREDICTOR (DUAL-STAGE AI)
# ==========================================
elif menu == "🔮 Trip Predictor (Dual-Stage AI)":
    st.markdown('<h1 class="title-text">Taxi Fare & Tip Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Enter trip parameters below. The AI will estimate the base fare and analyze the tipping probability.</p>', unsafe_allow_html=True)
    
    if reg_model is None or clf_model is None or route_stats is None:
        st.error("⚠️ AI Models or Route Statistics not loaded. Please verify the model files in models/ directory.")
    else:
        # Define Presets
        presets = {
            "casual": {
                "vendor": 2, "passengers": 1, "distance": 1.5, "duration": 8.0,
                "pu_id": 140, "do_id": 263, "hour": 15, "day": "Tuesday", "month": "June", "rate_code": 1
            },
            "commuter": {
                "vendor": 2, "passengers": 1, "distance": 4.5, "duration": 22.0,
                "pu_id": 236, "do_id": 231, "hour": 8, "day": "Monday", "month": "October", "rate_code": 1
            },
            "airport": {
                "vendor": 1, "passengers": 2, "distance": 19.5, "duration": 48.0,
                "pu_id": 132, "do_id": 230, "hour": 17, "day": "Friday", "month": "November", "rate_code": 2
            }
        }
        
        # Callback to update distance and duration based on selected PU/DO location
        def update_route_defaults():
            pu = st.session_state.pu_input
            do = st.session_state.do_input
            key = f"{pu} {do}"
            d_dist = route_stats["grouped_dist"].get(key, route_stats["global_avg_dist"])
            d_dur = route_stats["grouped_dur"].get(key, route_stats["global_avg_dur"])
            st.session_state.distance_input = float(d_dist)
            st.session_state.duration_input = float(d_dur)
            
        # Callback to load preset values into session state
        def apply_demo_preset(preset_key):
            p = presets[preset_key]
            st.session_state.vendor_input = p["vendor"]
            st.session_state.passenger_input = p["passengers"]
            st.session_state.distance_input = p["distance"]
            st.session_state.duration_input = p["duration"]
            st.session_state.pu_input = p["pu_id"]
            st.session_state.do_input = p["do_id"]
            st.session_state.hour_input = p["hour"]
            st.session_state.day_input = p["day"]
            st.session_state.month_input = p["month"]
            st.session_state.rate_code_input = p["rate_code"]

        # Initialize session state defaults if they are missing
        if "pu_input" not in st.session_state:
            st.session_state.pu_input = 100
        if "do_input" not in st.session_state:
            st.session_state.do_input = 231
            
        # Get defaults based on current state values
        curr_key = f"{st.session_state.pu_input} {st.session_state.do_input}"
        init_dist = route_stats["grouped_dist"].get(curr_key, route_stats["global_avg_dist"])
        init_dur = route_stats["grouped_dur"].get(curr_key, route_stats["global_avg_dur"])
        
        if "distance_input" not in st.session_state:
            st.session_state.distance_input = float(init_dist)
        if "duration_input" not in st.session_state:
            st.session_state.duration_input = float(init_dur)
        if "passenger_input" not in st.session_state:
            st.session_state.passenger_input = 1
        if "vendor_input" not in st.session_state:
            st.session_state.vendor_input = 2
        if "rate_code_input" not in st.session_state:
            st.session_state.rate_code_input = 1
        if "hour_input" not in st.session_state:
            st.session_state.hour_input = 12
        if "day_input" not in st.session_state:
            st.session_state.day_input = "Monday"
        if "month_input" not in st.session_state:
            st.session_state.month_input = "March"

        # Quick Presets Area
        st.markdown("### 💡 Quick Load Demo Profiles")
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            if st.button("🚕 Short Local Trip", on_click=apply_demo_preset, args=("casual",)):
                st.toast("Loaded Short Local Trip Preset!")
        with col_p2:
            if st.button("🏢 Daily Commuter Rush", on_click=apply_demo_preset, args=("commuter",)):
                st.toast("Loaded Daily Commuter Preset!")
        with col_p3:
            if st.button("✈️ Airport JFK Trip", on_click=apply_demo_preset, args=("airport",)):
                st.toast("Loaded JFK Airport Preset!")
                
        st.markdown("---")
        st.markdown("### 📝 Input Trip Parameters")
        
        # Layout columns
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            pu_id = st.number_input("Pickup Location ID (PULocationID)", min_value=1, max_value=265, key="pu_input", on_change=update_route_defaults)
            do_id = st.number_input("Dropoff Location ID (DOLocationID)", min_value=1, max_value=265, key="do_input", on_change=update_route_defaults)
            
            distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, step=0.1, key="distance_input",
                                       help="Masukkan estimasi jarak. Default diisi otomatis berdasarkan rata-rata historis rute terpilih.")
            duration = st.number_input("Trip Duration (minutes)", min_value=0.5, max_value=300.0, step=0.5, key="duration_input",
                                       help="Masukkan estimasi durasi. Default diisi otomatis berdasarkan rata-rata historis rute terpilih.")
            
            passengers = st.number_input("Passenger Count", min_value=1, max_value=6, key="passenger_input")

        with form_col2:
            vendor = st.selectbox("Taxi Vendor", [1, 2], key="vendor_input", format_func=lambda x: f"Vendor {x} (Creative Mobile)" if x == 1 else f"Vendor {x} (VeriFone)")
            rate_code = st.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 6], key="rate_code_input", format_func=lambda x: {
                1: "1 — Standard Rate",
                2: "2 — JFK Airport Rate",
                3: "3 — Newark Airport Rate",
                4: "4 — Nassau/Westchester Rate",
                5: "5 — Negotiated Fare",
                6: "6 — Group Ride Rate"
            }.get(x, str(x)))
            
            hour = st.slider("Pickup Hour (0-23)", 0, 23, key="hour_input")
            day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="day_input")
            month = st.selectbox("Month of Year", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], key="month_input")

        # Dynamic Route statistics retrieval from processed dataset
        num_samples = 0
        actual_generous_rate = None
        avg_tip_pct = None
        
        if df_taxi is not None:
            route_df = df_taxi[(df_taxi["PULocationID"] == pu_id) & (df_taxi["DOLocationID"] == do_id)]
            num_samples = len(route_df)
            if num_samples > 0:
                actual_generous_rate = route_df["generous"].mean() * 100
                avg_tip_pct = route_df["tip_percent"].mean() * 100

        # Check velocity validation
        if duration > 0 and distance > 0:
            avg_speed = distance / (duration / 60)
            if avg_speed > 60:
                st.warning(f"⚠️ Warning: Rata-rata kecepatan berkendara sangat tinggi ({avg_speed:.1f} mph). Harap pastikan keselarasan nilai Distance dan Duration.")
                
        predict_clicked = st.button("🚀 Run AI Dual-Stage Inference", type="primary")
        
        if predict_clicked:
            with st.spinner("AI is executing regression and classification pipelines..."):
                time.sleep(0.4)  # simulate sub-second API latency
                
                # Time features processing
                day_lower = day.lower()
                month_abbr = month.lower()[:3]
                
                is_weekday = day_lower not in ["saturday", "sunday"]
                is_rush_time = (6 <= hour < 10) or (16 <= hour < 20)
                rush_hour = 1 if (is_weekday and is_rush_time) else 0
                
                am_rush = 1 if (6 <= hour < 10) else 0
                daytime = 1 if (10 <= hour < 16) else 0
                pm_rush = 1 if (16 <= hour < 20) else 0
                nighttime = 1 if (hour >= 20 or hour < 6) else 0
                
                # Retrieve default historical averages based on selected PU/DO for display comparison
                route_key = f"{pu_id} {do_id}"
                hist_dist = route_stats["grouped_dist"].get(route_key, route_stats["global_avg_dist"])
                hist_dur = route_stats["grouped_dur"].get(route_key, route_stats["global_avg_dur"])
                
                # Stage 1 Inference: Fare prediction using actual user input distance & duration
                df_reg = pd.DataFrame([{
                    "VendorID": str(vendor),
                    "passenger_count": passengers,
                    "mean_distance": distance,
                    "mean_duration": duration,
                    "rush_hour": rush_hour
                }])
                
                predicted_fare = reg_model.predict(df_reg)[0]
                predicted_fare = max(2.50, predicted_fare) # baseline initial starting flag rate
                
                # Stage 2 Inference: Tip generous prediction using actual user input distance & duration
                df_clf = pd.DataFrame([{
                    "passenger_count": passengers,
                    "mean_distance": distance,
                    "mean_duration": duration,
                    "predicted_fare": predicted_fare,
                    "am_rush": am_rush,
                    "daytime": daytime,
                    "pm_rush": pm_rush,
                    "nighttime": nighttime,
                    "VendorID": str(vendor),
                    "RatecodeID": str(rate_code),
                    "PULocationID": str(pu_id),
                    "DOLocationID": str(do_id),
                    "day": day_lower,
                    "month": month_abbr
                }])
                
                is_generous = clf_model.predict(df_clf)[0]
                prob_generous = clf_model.predict_proba(df_clf)[0][1]
                
                st.markdown("---")
                st.markdown("### 🏁 AI Inference Results")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("#### Stage 1: Estimasi Tarif (Fare Regressor)")
                    st.metric(label="Predicted Fare Amount ($)", value=f"${predicted_fare:.2f}", delta=f"${(predicted_fare - 2.50):.2f} above NYC base flag fare")
                    
                    st.info(f"📍 **Route History:** Rata-rata statistik rute historis PULocation {pu_id} → DOLocation {do_id} adalah **{hist_dist:.2f} miles** dan **{hist_dur:.2f} menit**.")
                    
                    # Display route-level stats dynamically
                    if num_samples > 0:
                        st.markdown(f"""
                        <div style="background-color: #f5f2e9; border-left: 4px solid #d99b26; padding: 12px 15px; border-radius: 6px; margin-top: 10px; font-size: 0.9rem; border: 1px solid #e2ddcf; color: #334e68;">
                            📊 <strong>Statistik Rute Terpilih di Dataset:</strong><br>
                            • Jumlah sampel historis: <strong>{num_samples} perjalanan</strong><br>
                            • Persentase Generous Tipper asli: <strong>{actual_generous_rate:.1f}%</strong><br>
                            • Rata-rata persentase tip: <strong>{avg_tip_pct:.1f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #f5f2e9; border-left: 4px solid #ef4444; padding: 12px 15px; border-radius: 6px; margin-top: 10px; font-size: 0.9rem; border: 1px solid #e2ddcf; color: #334e68;">
                            📊 <strong>Statistik Rute Terpilih di Dataset:</strong><br>
                            • Rute ini <strong>belum pernah terlihat</strong> dalam dataset training.<br>
                            • Model menggunakan nilai rata-rata global (<strong>52.6%</strong>) sebagai estimasi awal.
                        </div>
                        """, unsafe_allow_html=True)
                        
                with res_col2:
                    st.markdown("#### Stage 2: Analisis Tip (Tip Classifier)")
                    
                    verdict_class = "generous" if is_generous == 1 else "standard"
                    verdict_icon = "🟢" if is_generous == 1 else "🟡"
                    verdict_text = "GENEROUS TIPPER (Likely ≥ 20%)" if is_generous == 1 else "STANDARD TIPPER (Likely < 20%)"
                    
                    st.markdown(
                        f'<div class="verdict-card {verdict_class}">'
                        f'<h3>{verdict_icon} Verdict: {verdict_text}</h3>'
                        f'<p>Model mendeteksi probabilitas penumpang memberikan tip murah hati sebesar <span class="highlight-yellow">{prob_generous*100:.1f}%</span>.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(float(prob_generous))
                    
                # Action plan
                st.markdown("#### 📋 Driver Action Plan & Strategy")
                if is_generous == 1:
                    st.success("💰 **High-Value Target:** Penumpang di rute ini cenderung memberi tip tinggi. Berikan layanan ramah, kabin bersih, AC nyaman, dan mengemudi dengan halus untuk mengamankan tip $\ge$ 20%.")
                else:
                    st.warning("⚠️ **Retention Challenge:** Penumpang di rute ini cenderung memberikan tip standar. Jaga komunikasi yang baik dan tawarkan kenyamanan ekstra untuk mendongkrak kemungkinan mendapatkan tip lebih besar.")
                    
        # Domain Insight & Explainable AI Card explaining tipping base rate and overfitting
        st.markdown(f"""
        <div class="insight-box">
            <strong>💡 Analisis Domain & Penjelasan Hasil Prediksi (Explainable AI):</strong><br>
            • <strong>Mengapa hasil prediksi tip cenderung tinggi?</strong> Di New York City, pembayaran taksi lewat kartu kredit secara otomatis menampilkan tombol pintas tip default sebesar 
            <span class="highlight-yellow">20%, 25%, dan 30%</span>. Secara statistik, <strong>52.6%</strong> perjalanan dengan kartu kredit masuk kategori "Generous" (Tip &ge; 20%). Oleh karena itu, model AI dilatih pada data yang didominasi oleh perilaku default ini.<br>
            • <strong>Mengapa perubahan jarak/durasi tidak drastis mengubah hasil tip?</strong> Dataset pelatihan ini memiliki 12.183 baris. Karena NYC memiliki 265 zona, terdapat lebih dari 70.000 kemungkinan rute. 
            Saat ini rute PULocation {pu_id} → DOLocation {do_id} hanya memiliki <strong>{num_samples} sampel</strong> di dataset. Ketika sampel historis rute sangat minim (misalnya 1 sampel) dan sampel tersebut kebetulan memberikan tip tinggi, model <i>Random Forest</i> cenderung mengalami <strong>overfitting lokal</strong> terhadap ID rute tersebut, sehingga prediksi tip akan terus-menerus bernilai tinggi untuk rute tersebut tidak peduli seberapa kecil durasi atau jarak yang diinput.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 3. TAB: MODEL REPORT & PERFORMANCE
# ==========================================
elif menu == "📈 Model Report & Performance":
    st.markdown('<h1 class="title-text">Model Performance & Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Performance details of the non-linear fare regressor (XGBoost) and the generous tip classifier (Random Forest).</p>', unsafe_allow_html=True)
    
    # Validation metrics KPIs
    rep1, rep2, rep3, rep4 = st.columns(4)
    with rep1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Stage 1 R² Score</div>'
            '<div class="metric-value">0.6682</div><span style="color:#22c55e; font-size:0.8rem;">▲ High variance explained</span></div>', 
            unsafe_allow_html=True
        )
    with rep2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Stage 1 MAE</div>'
            '<div class="metric-value">$3.07</div><span style="color:#8a92a6; font-size:0.8rem;">Mean Absolute Error</span></div>', 
            unsafe_allow_html=True
        )
    with rep3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Stage 2 Recall</div>'
            '<div class="metric-value">78.6%</div><span style="color:#22c55e; font-size:0.8rem;">Minimizes false negatives</span></div>', 
            unsafe_allow_html=True
        )
    with rep4:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Stage 2 F1-Score</div>'
            '<div class="metric-value">73.1%</div><span style="color:#22c55e; font-size:0.8rem;">Overall balanced metric</span></div>', 
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    st.markdown("### 📈 Validation Plots from Model Training")
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        st.markdown("#### Actual vs. Predicted Fares")
        reg_plot = os.path.join(project_root, "models", "regression_actual_vs_predicted.png")
        if os.path.exists(reg_plot):
            st.image(reg_plot, width=420)
        else:
            st.warning("⚠️ Regression plot not found in models/ directory.")
        st.caption("Bagan ini membandingkan tarif taksi aktual dengan tarif taksi hasil prediksi model regressor XGBoost. Pola linear yang solid menunjukkan tingkat kecocokan model yang baik (R² = 0.6682).")
            
    with plot_col2:
        st.markdown("#### Confusion Matrix")
        cm_plot = os.path.join(project_root, "models", "confusion_matrix.png")
        if os.path.exists(cm_plot):
            st.image(cm_plot, width=420)
        else:
            st.warning("⚠️ Confusion matrix plot not found in models/ directory.")
        st.caption("Confusion matrix menunjukkan performa deteksi tip murah hati. Model memiliki tingkat True Positive (Recall) yang tinggi (78.6%), meminimalkan hilangnya peluang taksi memperoleh tip murah hati.")
            
    st.markdown("---")
    
    plot_col3, plot_col4 = st.columns(2)
    
    with plot_col3:
        st.markdown("#### ROC Curve (AUC)")
        roc_plot = os.path.join(project_root, "models", "roc_curve.png")
        if os.path.exists(roc_plot):
            st.image(roc_plot, width=420)
        else:
            st.warning("⚠️ ROC curve plot not found in models/ directory.")
        st.caption("Kurva ROC menunjukkan trade-off antara True Positive Rate dan False Positive Rate. Luas area di bawah kurva (AUC) mengukur keandalan klasifikasi tip model Random Forest.")
        
    with plot_col4:
        st.markdown("#### Top 10 Feature Importance")
        if clf_model is not None:
            try:
                preprocessor = clf_model.named_steps['preprocessor']
                classifier = clf_model.named_steps['classifier']
                numeric_features = ['passenger_count', 'mean_distance', 'mean_duration', 'predicted_fare', 'am_rush', 'daytime', 'pm_rush', 'nighttime']
                cat_encoder = preprocessor.named_transformers_['cat']
                cat_features = list(cat_encoder.get_feature_names_out(['VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'day', 'month']))
                all_features = numeric_features + cat_features
                importances = classifier.feature_importances_
                feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
                sns.barplot(x=feat_imp.values, y=feat_imp.index, color='#d99b26', ax=ax, edgecolor='#f5f2e9')
                ax.set_xlabel("Relative Importance Score")
                ax.set_ylabel("Model Features")
                ax.set_title("Top 10 Most Important Features", fontsize=11, fontweight='bold', pad=10, color='#d99b26')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Gagal memuat fitur penting: {e}")
        else:
            st.warning("⚠️ Model klasifikasi tidak dimuat.")
        st.caption("Grafik ini menunjukkan kontribusi relatif dari 10 fitur teratas dalam memprediksi penumpang yang dermawan. Vendor dan tarif perjalanan merupakan prediktor dominan.")
