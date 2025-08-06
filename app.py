import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import io
import joblib
import tempfile
from plotly.subplots import make_subplots
from keras.models import load_model
from datetime import datetime, timedelta
# Tema Seaborn
sns.set_theme(style="darkgrid")

# Sembunyikan sidebar
st.markdown("""
            <style>
            [data-testid="stSidebar"] { display: none; }
            </style>
""", unsafe_allow_html=True)

# ========== Styling Tambahan ==========
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 10px 20px;
        margin-right: 8px;
        background-color: #e0e7ff;
        border-radius: 8px;
        color: black;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
        padding: 16px;
        background-color: #e6f2ff;
        border-left: 5px solid #3399ff;
    }
    .stMarkdown p {
        font-size: 15px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Load Model dan Scaler ==========
model = load_model("model_lstm_baru.h5", compile=False)
scaler_y = joblib.load("scaler_y.pkl")
input_seq = np.load("last_input.npy") # (1, time_steps, features)

# ========== Fungsi Prediksi recursive ==========
def recursive_forecast(model, input_seq, n_steps, scaler):
    predictions = []
    current_input = input_seq.copy()  # (1, time_steps, features)

    for _ in range(n_steps):
        pred = model.predict(current_input, verbose=0)   # pred: (1, n_features)
        predictions.append(pred[0])  # ambil isi array

        next_input = np.append(current_input[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
        current_input = next_input  # bentuk: (1, time_steps, features)

    predictions = np.array(predictions)  # (n_steps, n_features)
    return scaler.inverse_transform(predictions)

# ========== Fungsi Narasi Dinamis ==========
def generate_narrative(df, selected_params):
    narratives = {}
    for param in selected_params:
        if len(df) < 2:
            narratives[param] = f"Tidak cukup data untuk membuat narasi {param}."
            continue

        latest_val = df[param].iloc[-1]
        prev_val = df[param].iloc[-2]
        change = latest_val - prev_val
        percent = (change / prev_val) * 100 if prev_val != 0 else 0
 
        if percent > 20:
            status = "meningkat tajam"
        elif percent > 5:
            status = "meningkat"
        elif percent < -20:
            status = "menurun tajam"
        elif percent < -5:
            status = "menurun"
        else:
            status = "stabil"

        narratives[param] = (
            f"<b>{param}</b> diprediksi <b>{status}</b> sebesar <b>{percent:.1f}%</b> "
            f"pada minggu terakhir dibanding minggu sebelumnya."
        )
    return narratives

 # ================= UI & Input Setup =================
# Judul Besar di Tengah
st.markdown("""
<h1 style='text-align:center; font-size:42px; margin-bottom:0;'>🌤️ Prediksi Kualitas Udara <span style='color:#6228C22;'>Kota Surabaya</span></h1>
""", unsafe_allow_html=True)

# Subjudul dengan gaya lembut
st.markdown("<p style='text-align:center; font-size:18px; color:#aaa;'>Menggunakan model LSTM untuk memprediksi parameter ISPU secara mingguan</p>", unsafe_allow_html=True)

# Tombol mulai di tengah
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Mulai", use_container_width=True):
            st.session_state.started = True
            st.rerun()
        else:
            # Notifikasi info jika tombol belum ditekan
            st.markdown("""
            <div style='background-color:#e3f2fd;
                        padding:10px 16px;
                        border-left:4px solid #2196f3;
                        border-radius:6px;
                        color:#0d47a1;
                        max-width:600px;
                        margin:auto;
                        margin-top:16px;
                        text-align:center;'>
            ℹ️ Klik tombol <b>Mulai</b> untuk memilih parameter dan memulai prediksi.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

# ========== UI Utama ==========
if st.session_state.started:
    st.header("🛠️ Pengaturan Prediksi")
    n_weeks = st.slider("Berapa minggu ke depan ingin diprediksi?", 1, 52, 12)
    param_names = ['PM10', 'SO2', 'CO', 'O3', 'NO2']
    selected_params = st.multiselect(
        "Pilih parameter ISPU yang ingin diprediksi:",
        param_names,
        default=[],
        help="Pilih satu atau lebih parameter untuk ditampilkan dalam grafik dan narasi."
    )
    predict_button = st.button("🔮Prediksi")

if predict_button:
    if not selected_params:
        st.markdown(
            """
            <div style='background-color:#fff3cd; padding:10px; border-left:6px solid #ffecb5; 
                border-radius:5px; font-weight:bold; color:#333;'>⚠️ Silakan pilih minimal satu parameter ISPU untuk memulai prediksi.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        with st.spinner("⏳ Sedang memproses prediksi..."):
            forecast = recursive_forecast(model, input_seq, n_weeks, scaler_y)
            df_pred = pd.DataFrame(forecast, columns=param_names)
            dates = [datetime.today() + timedelta(weeks=i) for i in range(n_weeks)]
            df_pred['Tanggal'] = [d.strftime('%Y-%m-%d') for d in dates]
            df_pred = df_pred[['Tanggal'] + param_names]

            tab1, tab2, tab3 = st.tabs(["📈 Grafik Prediksi", "📝 Interpretasi", "📤 Unduhan"])

            with tab1:
                st.subheader("📊 Grafik Prediksi per Parameter")
                colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                for i, name in enumerate(selected_params):
                    idx = param_names.index(name)
                    values = forecast[:, idx]
                    mean_val = values.mean()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name=f"{name}",
                        marker=dict(color=colors[i % len(colors)]),
                        line=dict(width=2),
                        hovertemplate=(
                            f"<b>{name}</b><br>"
                            "Tanggal: %{x|%d-%b-%Y}<br>"
                            "Nilai: %{y:.2f}<extra></extra>"
                        )
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=[mean_val]*len(dates),
                        mode='lines',
                        name='Rata-rata',
                        line=dict(color=colors[i % len(colors)], width=1.5, dash='dash'),
                        hoverinfo='skip'
                    ))
                    fig.add_hline(
                        y=100,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text="Batas ISPU",
                        annotation_position="top left"
                    )
                    fig.update_layout(
                        title=f"Prediksi {name}",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=60, b=40),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if len(values) >= 2:
                        latest_val = values[-1]
                        prev_val = values[-2]
                        change = latest_val - prev_val
                        percent = (change / prev_val) * 100 if prev_val != 0 else 0
                        if percent > 20:
                            status = "meningkat tajam"
                        elif percent > 5:
                            status = "meningkat"
                        elif percent < -20:
                            status = "menurun tajam"
                        elif percent < -5:
                            status = "menurun"
                        else:
                            status = "stabil"
                        st.markdown(
                            f"📝 <b>{name}</b> diprediksi <b>{status}</b> sebesar <b>{percent:.1f}%</b> pada minggu terakhir dibanding minggu sebelumnya.",
                            unsafe_allow_html=True
                        )

            with tab2:
                st.subheader("📝 Interpretasi Dinamis")
                narratives = generate_narrative(df_pred, selected_params)
                for param in selected_params:
                    st.markdown(f"🔹 {narratives[param]}", unsafe_allow_html=True)

            with tab3:
                st.subheader("📤 Unduh Hasil Prediksi")
                csv_buffer = io.StringIO()
                df_pred.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Unduh CSV Hasil Prediksi",
                    data=csv_buffer.getvalue(),
                    file_name="prediksi_udara.csv",
                    mime="text/csv",
                    help="Unduh hasil prediksi dalam format Excel/CSV"
                )
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="🖼️ Unduh Grafik (PNG)",
                    data=img_bytes,
                    file_name="grafik_prediksi.png",
                    mime="image/png"
                )


