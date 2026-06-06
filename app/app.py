"""
Portfolio Streamlit App
=======================
Author  : Aditya Ulil Albab
NIM     : A11.2023.15093
Kampus  : Universitas Dian Nuswantoro
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
IMG_DIR    = os.path.join(BASE_DIR, "image")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def img(name: str) -> str:
    """Return full path for an image in the image/ folder."""
    return os.path.join(IMG_DIR, name)


def model_file(name: str) -> str:
    """Return full path for a file in the models/ folder."""
    return os.path.join(MODELS_DIR, name)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Aditya Ulil Albab | Data Science Portfolio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Gradient title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Section headings */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 1.2rem 0 0.8rem 0;
    }
    /* Skill badges */
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 13px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 3px;
        font-weight: 500;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea18, #764ba218);
        border: 1px solid #667eea40;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_home, tab_project, tab_predict, tab_viz = st.tabs([
    "🏠 Home",
    "📂 Proyek Saya",
    "🏡 House Price Predictor",
    "📊 Visualisasi Model & Data",
])


# ══════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════
with tab_home:

    # ── Header ────────────────────────────────
    st.markdown('<div class="main-title">Hi, I\'m Aditya Ulil Albab 👋</div>',
                unsafe_allow_html=True)
    st.subheader("Mahasiswa Teknik Informatika · Universitas Dian Nuswantoro")

    col_photo, col_bio = st.columns([1, 3])

    with col_photo:
        profile_path = img("profile.jpeg")
        if os.path.exists(profile_path):
            st.image(profile_path, width=200)
        else:
            st.markdown("### 👨‍💻")

    with col_bio:
        st.write("""
        Mahasiswa semester 6 Teknik Informatika di Universitas Dian Nuswantoro, Semarang,
        dengan ketertarikan kuat di bidang **Data Science** dan **Machine Learning**.
        Aktif mengerjakan proyek berbasis data mulai dari analisis sentimen wisata,
        prediksi time series, hingga pipeline ML end-to-end siap produksi.
        """)
        st.write("**Core Skills:** Machine Learning · NLP · Time Series · Data Visualization · Streamlit")

    st.divider()

    # ── Metrics ───────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semester", "6")
    c2.metric("Proyek Portfolio", "3+")
    c3.metric("Best Model R²", "90%+")
    c4.metric("Reviews Dianalisis", "708")

    st.divider()

    # ── About section ─────────────────────────
    st.markdown('<div class="section-title">👤 Tentang Saya</div>', unsafe_allow_html=True)

    col_info, col_skills = st.columns(2)

    with col_info:
        st.markdown("""
        | Info | Detail |
        |------|--------|
        | **Nama** | Aditya Ulil Albab |
        | **NIM** | A11.2023.15093 |
        | **Universitas** | Universitas Dian Nuswantoro |
        | **Jurusan** | Teknik Informatika |
        | **Semester** | 6 |
        | **Domisili** | Semarang, Jawa Tengah |
        | **GitHub** | [github.com/justlyznn](https://github.com/justlyznn) |
        """)

    with col_skills:
        st.markdown("**🛠️ Keahlian Teknis**")
        skills = [
            "Python", "Machine Learning", "Deep Learning", "NLP",
            "Scikit-learn", "TensorFlow", "Pandas", "NumPy",
            "Matplotlib", "Seaborn", "Plotly", "Streamlit",
            "SQL", "Git", "Data Visualization",
        ]
        badges = " ".join([f"<span class='skill-badge'>{s}</span>" for s in skills])
        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Quick project preview ─────────────────
    st.markdown('<div class="section-title">🚀 Featured Projects</div>', unsafe_allow_html=True)

    fp1, fp2, fp3 = st.columns(3)

    with fp1:
        with st.container(border=True):
            st.markdown("**🏖️ Search Engine Sentimen Karimunjawa**")
            st.caption("NLP · Sentiment Analysis · Streamlit")
            st.write("Analisis 708 ulasan pantai dengan klasifikasi sentimen positif/negatif.")
            st.info("👉 Detail di tab **Proyek Saya**")

    with fp2:
        with st.container(border=True):
            st.markdown("**🥇 Prediksi Harga Gold (LSTM)**")
            st.caption("Deep Learning · Time Series · LSTM")
            st.write("Prediksi harga emas dengan LSTM — data historis 2021–2026.")
            st.info("👉 Detail di tab **Proyek Saya**")

    with fp3:
        with st.container(border=True):
            st.markdown("**🏡 House Price Prediction**")
            st.caption("Regression · Gradient Boosting · R²=90%")
            st.write("Pipeline ML end-to-end prediksi harga rumah dari dataset Kaggle.")
            st.info("👉 Coba di tab **House Price Predictor**")


# ══════════════════════════════════════════════
# TAB 2 — PROYEK SAYA
# ══════════════════════════════════════════════
with tab_project:

    st.markdown('<div class="section-title">📂 Proyek Saya</div>', unsafe_allow_html=True)
    st.write("Berikut proyek-proyek Data Science & ML yang telah saya kerjakan:")

    # ── PROJECT 1 ─────────────────────────────
    st.markdown("---")
    st.markdown("### 🏖️ Project 1 — Search Engine Sentimen Pantai Karimunjawa")

    col_img1, col_desc1 = st.columns([1, 2])

    with col_img1:
        p1_img = img("porto1_search_engine.jpeg")
        if os.path.exists(p1_img):
            st.image(p1_img, use_container_width=True,
                     caption="Tampilan Search Engine Karimunjawa")
        else:
            st.info("📷 Image: porto1_search_engine.jpeg")

    with col_desc1:
        st.markdown("""
        **Deskripsi:**
        Aplikasi web berbasis Streamlit untuk mencari dan menganalisis ulasan
        pantai-pantai di Kepulauan Karimunjawa. Menggabungkan **analisis sentimen**
        (positif/negatif) dengan pencarian berbasis kata kunci dan filter rating,
        sehingga wisatawan dapat membuat keputusan perjalanan yang lebih informatif.

        **Teknologi:**
        - 🐍 Python · Streamlit
        - 🤖 NLP & Sentiment Analysis
        - 📊 Data Mining dari review Google Maps
        - 📈 Plotly untuk visualisasi interaktif

        **Highlight:**
        - 708 ulasan dianalisis dari berbagai pantai
        - Rating rata-rata keseluruhan: **4.53 ⭐**
        - TOP 3 pantai: Pantai Bobby · Tanjung Gelam · Sunset Beach
        - Filter by rating, sentimen, dan kata kunci
        """)

        col_btn1a, col_btn1b = st.columns(2)
        with col_btn1a:
            st.markdown(
                "[![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?logo=github)]"
                "(https://github.com/justlyznn/stki-uas-A11.2023.15093-Aditya-Ulil-Albab.git)"
            )
        with col_btn1b:
            st.metric("Review Positif", "573 / 708")

    # ── PROJECT 2 ─────────────────────────────
    st.markdown("---")
    st.markdown("### 🥇 Project 2 — Prediksi Harga Gold (LSTM Time Series)")

    col_img2, col_desc2 = st.columns([1, 2])

    with col_img2:
        p2_img = img("porto2_prediksi_gold.jpeg")
        if os.path.exists(p2_img):
            st.image(p2_img, use_container_width=True,
                     caption="Actual vs Predicted Gold Price")
        else:
            st.info("📷 Image: porto2_prediksi_gold.jpeg")

    with col_desc2:
        st.markdown("""
        **Deskripsi:**
        Model **LSTM (Long Short-Term Memory)** untuk memprediksi harga emas (Gold/XAU)
        berdasarkan data historis 2021–2026. Model menangkap pola tren jangka panjang
        dan memberikan prediksi harga **30 hari ke depan**.

        **Teknologi:**
        - 🧠 Deep Learning · LSTM / RNN
        - 📈 Time Series Analysis
        - 🐍 Python · TensorFlow / Keras
        - 📊 Matplotlib · Plotly

        **Highlight:**
        - Data historis 5+ tahun (Jan 2021 – 2026)
        - Tren harga gold: **$1,700 → $4,600**
        - Prediksi 30 hari ke depan (future forecast)
        - Visualisasi Actual vs Predicted yang akurat
        """)

        col_btn2a, col_btn2b = st.columns(2)
        with col_btn2a:
            st.markdown(
                "[![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?logo=github)]"
                "(https://github.com/justlyznn/FINAL_PROJECT_DATA_MINING.git)"
            )
        with col_btn2b:
            st.metric("Forecast Horizon", "30 Days")

    # ── PROJECT 3 ─────────────────────────────
    st.markdown("---")
    st.markdown("### 🏡 Project 3 — House Price Prediction (Gradient Boosting)")

    col_desc3, col_stats3 = st.columns([2, 1])

    with col_desc3:
        st.markdown("""
        **Deskripsi:**
        Pipeline machine learning end-to-end untuk memprediksi harga rumah
        menggunakan dataset **House Prices: Advanced Regression Techniques** dari Kaggle.
        Membandingkan 3 algoritma regresi (Gradient Boosting, Random Forest, Ridge)
        dan memilih model terbaik berdasarkan metrik evaluasi komprehensif.

        **Teknologi:**
        - 🌳 Gradient Boosting · Random Forest · Ridge Regression
        - ⚙️ Scikit-learn Pipeline (end-to-end preprocessing + training)
        - 🔧 Feature engineering · imputation · encoding
        - 🚀 Streamlit untuk deployment & prediksi interaktif

        **Highlight:**
        - Dataset: **1,460 rumah × 81 fitur**
        - Best Model: **Gradient Boosting** — R² = **0.9032**
        - Upload CSV untuk prediksi batch
        - Input manual dengan estimasi kategori harga
        """)
        st.info("💡 Coba langsung prediksi di tab **🏡 House Price Predictor**!")

    with col_stats3:
        st.markdown("**📊 Model Performance:**")
        st.metric("Best Model", "Gradient Boosting")
        st.metric("R² Score", "0.9032")
        st.metric("RMSE", "~$27,246")
        st.metric("Training Data", "1,460 rows")


# ══════════════════════════════════════════════
# TAB 3 — HOUSE PRICE PREDICTOR
# ══════════════════════════════════════════════
with tab_predict:

    st.markdown('<div class="section-title">🏡 House Price Prediction</div>',
                unsafe_allow_html=True)
    st.write("Upload CSV atau isi parameter manual untuk mendapatkan prediksi harga rumah.")

    # ── Load model ────────────────────────────
    @st.cache_resource(show_spinner="Memuat model…")
    def load_model():
        model_path = model_file("house_price_model.pkl")
        feat_path  = model_file("feature_info.json")
        try:
            model     = joblib.load(model_path)
            with open(feat_path) as f:
                feat_info = json.load(f)
            return model, feat_info
        except FileNotFoundError:
            return None, None

    model, feat_info = load_model()

    if model is None:
        st.error(
            "⚠️ Model belum ditemukan di folder `models/`.  \n"
            "Jalankan dulu: `python src/train_house_price.py`"
        )
    else:
        st.success("✅ Model Gradient Boosting berhasil dimuat  (R² = 0.9032)")

        tab_csv, tab_manual = st.tabs(["📁 Upload CSV", "🔢 Input Manual"])

        # ── Sub-tab: Upload CSV ────────────────
        with tab_csv:
            st.markdown("#### 📁 Prediksi Batch via CSV")
            st.caption(
                "Format CSV: kolom fitur sesuai dataset Kaggle House Prices  "
                "(tanpa kolom `Id` dan `SalePrice`)."
            )

            uploaded = st.file_uploader("Upload file CSV", type=["csv"])

            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    st.markdown(f"✅ **{df_up.shape[0]:,} baris × {df_up.shape[1]} kolom** dibaca.")

                    with st.expander("Preview data (10 baris pertama)"):
                        st.dataframe(df_up.head(10), use_container_width=True)

                    if st.button("🚀 Jalankan Prediksi Batch", key="batch"):
                        with st.spinner("Memproses…"):
                            DROP = ['PoolQC', 'MiscFeature', 'Alley', 'Fence',
                                    'MasVnrType', 'FireplaceQu', 'Id', 'SalePrice']
                            df_clean = df_up.drop(columns=DROP, errors='ignore')

                            try:
                                preds = model.predict(df_clean)
                                df_result = df_up.copy()
                                df_result['Predicted_SalePrice'] = preds.round(0).astype(int)

                                st.markdown("#### 🎯 Hasil Prediksi")
                                show_cols = ['Predicted_SalePrice'] + [
                                    c for c in df_up.columns
                                    if c != 'Predicted_SalePrice'
                                ]
                                st.dataframe(df_result[show_cols], use_container_width=True)

                                # Summary
                                s1, s2, s3 = st.columns(3)
                                s1.metric("Rata-rata", f"${preds.mean():,.0f}")
                                s2.metric("Tertinggi", f"${preds.max():,.0f}")
                                s3.metric("Terendah",  f"${preds.min():,.0f}")

                                # Distribution
                                fig_dist = px.histogram(
                                    x=preds, nbins=30,
                                    title="Distribusi Prediksi Harga",
                                    labels={'x': 'Predicted SalePrice (USD)', 'y': 'Count'},
                                    color_discrete_sequence=['#667eea'],
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)

                                # Download
                                st.download_button(
                                    "⬇️ Download Hasil (CSV)",
                                    data=df_result.to_csv(index=False),
                                    file_name="hasil_prediksi_harga_rumah.csv",
                                    mime="text/csv",
                                )

                            except Exception as e:
                                st.error(f"Error prediksi: {e}")
                                st.info("Pastikan format kolom CSV sesuai dataset Kaggle House Prices.")

                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

        # ── Sub-tab: Input Manual ──────────────
        with tab_manual:
            st.markdown("#### 🔢 Estimasi Harga via Parameter Manual")

            col1, col2, col3 = st.columns(3)

            with col1:
                overall_qual  = st.slider("Overall Quality (1–10)", 1, 10, 7,
                                          help="Kualitas bahan & finishing rumah")
                gr_liv_area   = st.number_input("Living Area (sqft)", 300, 6000, 1500, 50)
                year_built    = st.number_input("Tahun Dibangun", 1872, 2024, 2000)

            with col2:
                total_bsmt    = st.number_input("Total Basement SF", 0, 6000, 800, 50)
                garage_area   = st.number_input("Garage Area (sqft)", 0, 1500, 400, 50)
                full_bath     = st.selectbox("Full Bathrooms", [0, 1, 2, 3], index=2)

            with col3:
                bedroom       = st.selectbox("Bedrooms Above Grade", list(range(7)), index=3)
                garage_cars   = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)
                neighborhood  = st.selectbox("Neighborhood", [
                    'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst',
                    'Gilbert', 'NridgHt', 'Sawyer',
                ])

            if st.button("🔮 Prediksi Harga", key="manual"):
                with st.spinner("Menghitung…"):
                    input_df = pd.DataFrame([{
                        'MSSubClass': 60, 'LotFrontage': 70.0, 'LotArea': 9000,
                        'OverallQual': overall_qual, 'OverallCond': 5,
                        'YearBuilt': year_built, 'YearRemodAdd': year_built,
                        'MasVnrArea': 0.0, 'BsmtFinSF1': total_bsmt // 2,
                        'BsmtFinSF2': 0, 'BsmtUnfSF': total_bsmt // 2,
                        'TotalBsmtSF': total_bsmt, '1stFlrSF': gr_liv_area // 2,
                        '2ndFlrSF': gr_liv_area // 2, 'LowQualFinSF': 0,
                        'GrLivArea': gr_liv_area, 'BsmtFullBath': 1,
                        'BsmtHalfBath': 0, 'FullBath': full_bath, 'HalfBath': 0,
                        'BedroomAbvGr': bedroom, 'KitchenAbvGr': 1,
                        'TotRmsAbvGrd': bedroom + 3, 'Fireplaces': 1,
                        'GarageYrBlt': float(year_built), 'GarageCars': garage_cars,
                        'GarageArea': garage_area, 'WoodDeckSF': 0,
                        'OpenPorchSF': 0, 'EnclosedPorch': 0, '3SsnPorch': 0,
                        'ScreenPorch': 0, 'PoolArea': 0, 'MiscVal': 0,
                        'MoSold': 6, 'YrSold': 2023,
                        'MSZoning': 'RL', 'Street': 'Pave', 'LotShape': 'Reg',
                        'LandContour': 'Lvl', 'Utilities': 'AllPub',
                        'LotConfig': 'Inside', 'LandSlope': 'Gtl',
                        'Neighborhood': neighborhood, 'Condition1': 'Norm',
                        'Condition2': 'Norm', 'BldgType': '1Fam',
                        'HouseStyle': '2Story', 'RoofStyle': 'Gable',
                        'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd',
                        'Exterior2nd': 'VinylSd', 'ExterQual': 'Gd',
                        'ExterCond': 'TA', 'Foundation': 'PConc',
                        'BsmtQual': 'Gd', 'BsmtCond': 'TA',
                        'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ',
                        'BsmtFinType2': 'Unf', 'Heating': 'GasA',
                        'HeatingQC': 'Ex', 'CentralAir': 'Y',
                        'Electrical': 'SBrkr', 'KitchenQual': 'Gd',
                        'Functional': 'Typ', 'GarageType': 'Attchd',
                        'GarageFinish': 'RFn', 'GarageQual': 'TA',
                        'GarageCond': 'TA', 'PavedDrive': 'Y',
                        'SaleType': 'WD', 'SaleCondition': 'Normal',
                    }])

                    try:
                        price = model.predict(input_df)[0]

                        # Category label
                        if price < 130_000:
                            label, color = "💰 Rumah Terjangkau",    "normal"
                        elif price < 200_000:
                            label, color = "🏠 Rumah Kelas Menengah", "normal"
                        elif price < 300_000:
                            label, color = "🏡 Rumah Premium",        "normal"
                        else:
                            label, color = "🏰 Rumah Mewah",          "normal"

                        st.success(f"## Estimasi Harga: **${price:,.0f}**")
                        st.info(f"Kategori: **{label}**")
                        st.caption("*Estimasi berdasarkan model Gradient Boosting — R² = 0.9032")

                        # Radar chart of key inputs
                        radar_feats = ['OverallQual', 'GrLivArea_k',
                                       'TotalBsmtSF_k', 'GarageArea_k', 'Bedrooms']
                        radar_vals  = [
                            overall_qual / 10,
                            min(gr_liv_area / 5000, 1),
                            min(total_bsmt / 4000, 1),
                            min(garage_area / 1200, 1),
                            bedroom / 6,
                        ]
                        fig_radar = go.Figure(go.Scatterpolar(
                            r=radar_vals + [radar_vals[0]],
                            theta=radar_feats + [radar_feats[0]],
                            fill='toself',
                            fillcolor='rgba(102,126,234,0.2)',
                            line=dict(color='#667eea'),
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title="Profil Fitur Rumah (Normalized)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")


# ══════════════════════════════════════════════
# TAB 4 — VISUALISASI MODEL & DATA
# ══════════════════════════════════════════════
with tab_viz:

    st.markdown('<div class="section-title">📊 Visualisasi Model & Data</div>',
                unsafe_allow_html=True)

    # ── Model selector ────────────────────────
    selected = st.selectbox(
        "🔍 Pilih Model:",
        ["Gradient Boosting", "Random Forest", "Ridge Regression"],
    )

    # Static metrics (from training)
    ALL_METRICS = {
        "Gradient Boosting": {"RMSE": 27_246.19, "MAE": 16_359.43, "R2": 0.9032},
        "Random Forest":     {"RMSE": 30_412.50, "MAE": 18_742.10, "R2": 0.8811},
        "Ridge Regression":  {"RMSE": 40_823.90, "MAE": 27_614.20, "R2": 0.7342},
    }

    m = ALL_METRICS[selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{m['R2']:.4f}")
    c2.metric("RMSE",     f"${m['RMSE']:,.2f}")
    c3.metric("MAE",      f"${m['MAE']:,.2f}")

    st.divider()

    # ── Model comparison charts ───────────────
    st.markdown("#### 🏆 Perbandingan Semua Model")

    df_metrics = pd.DataFrame(ALL_METRICS).T.reset_index()
    df_metrics.columns = ['Model', 'RMSE', 'MAE', 'R2']

    fig_r2 = px.bar(
        df_metrics, x='Model', y='R2',
        title="R² Score per Model (Higher = Better)",
        color='R2', color_continuous_scale='Viridis',
        text_auto='.4f',
    )
    fig_r2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_r2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_rmse = px.bar(
            df_metrics, x='Model', y='RMSE',
            title="RMSE per Model (Lower = Better)",
            color='RMSE', color_continuous_scale='Reds_r',
            text_auto='.0f',
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col_b:
        fig_mae = px.bar(
            df_metrics, x='Model', y='MAE',
            title="MAE per Model (Lower = Better)",
            color='MAE', color_continuous_scale='Oranges_r',
            text_auto='.0f',
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    st.divider()

    # ── Feature distribution (from train_clean.csv) ───
    st.markdown("#### 📊 Distribusi Fitur Dataset")

    clean_path = model_file("train_clean.csv")
    if os.path.exists(clean_path):
        @st.cache_data(show_spinner=False)
        def load_clean():
            return pd.read_csv(clean_path)

        df_train = load_clean()
        num_feats = [c for c in df_train.select_dtypes(include='number').columns
                     if c != 'SalePrice']

        sel_feat = st.selectbox("Pilih fitur:", num_feats)

        col_dist, col_box = st.columns(2)
        with col_dist:
            fig_h = px.histogram(
                df_train, x=sel_feat, nbins=40,
                title=f"Distribusi: {sel_feat}",
                color_discrete_sequence=['#667eea'],
            )
            st.plotly_chart(fig_h, use_container_width=True)
        with col_box:
            fig_b = px.box(
                df_train, y=sel_feat,
                title=f"Box Plot: {sel_feat}",
                color_discrete_sequence=['#764ba2'],
            )
            st.plotly_chart(fig_b, use_container_width=True)

        st.divider()

        # ── Correlation bar ───────────────────
        st.markdown("#### 🔥 Korelasi Fitur vs SalePrice")

        num_df   = df_train.select_dtypes(include='number')
        if 'SalePrice' in num_df.columns:
            corr = num_df.corr()['SalePrice'].drop('SalePrice').abs().nlargest(15)
            fig_corr = px.bar(
                x=corr.values, y=corr.index,
                orientation='h',
                title="Top 15 Fitur Berkorelasi dengan SalePrice",
                labels={'x': 'Absolute Correlation', 'y': 'Feature'},
                color=corr.values, color_continuous_scale='Viridis',
            )
            fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.warning(
            "⚠️ `models/train_clean.csv` tidak ditemukan.  \n"
            "Jalankan `python src/train_house_price.py` terlebih dahulu."
        )

    st.divider()

    # ── Footer ────────────────────────────────
    st.markdown("""
    <div style='text-align:center; color:#888; padding:20px;'>
        Made with ❤️ by <b>Aditya Ulil Albab</b> · A11.2023.15093<br>
        Universitas Dian Nuswantoro · Data Science & Machine Learning Bootcamp
    </div>
    """, unsafe_allow_html=True)
