import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# =====================================
# Page config (WAJIB di atas)
# =====================================
st.set_page_config(
    page_title="Right Whale Call Detection",
    page_icon="🐋",
    layout="centered"
)

# =====================================
# Load model (sekali saja)
# =====================================
@st.cache_resource
def load_cnn_model():
    return load_model("best_cnn1d_model.keras")

model = load_cnn_model()

# =====================================
# Preprocessing (SAMA seperti training)
# =====================================
def preprocess_timeseries(ts):
    ts = np.array(ts, dtype=np.float32)
    ts = (ts - ts.mean()) / (ts.std() + 1e-8)
    ts = ts.reshape(1, -1, 1)
    return ts


# =====================================
# HEADER
# =====================================
st.markdown(
    """
    <h1 style='text-align: center;'>🐋 Right Whale Call Detection</h1>
    <p style='text-align: center; font-size: 16px;'>
        Sistem klasifikasi time series untuk mendeteksi suara Paus Kanan Atlantik
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================
# Upload Section
# =====================================
st.subheader("📂 Upload File Time Series")
st.write("Unggah **1 file CSV/TXT** berisi **1 data time series suara** (tanpa label).")

uploaded_file = st.file_uploader(
    "Pilih file",
    type=["txt", "csv"]
)

# =====================================
# Prediction
# =====================================
if uploaded_file is not None:
    try:
        # Load data
        ts = np.loadtxt(uploaded_file, delimiter=",")

        st.success("File berhasil diupload ✅")

        # Info file
        st.info(f"📊 Panjang time series: **{len(ts)} data point**")

        # Preprocess
        X = preprocess_timeseries(ts)

        # Predict
        with st.spinner("🔍 Menganalisis suara paus..."):
            prediction = model.predict(X)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Label mapping
        label_map = {
            0: "No Call (Bukan Suara Paus)",
            1: "Right Whale Call 🐋"
        }

        # =====================================
        # Result Section
        # =====================================
        st.subheader("📌 Hasil Prediksi")

        if predicted_class == 1:
            st.success(f"🐋 **TERDETEKSI SUARA PAUS**")
        else:
            st.warning("❌ **TIDAK TERDETEKSI SUARA PAUS**")

        st.write(f"**Prediksi:** {label_map[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # =====================================
        # Probability Visualization
        # =====================================
        st.subheader("📈 Confidence per Kelas")

        prob_df = pd.DataFrame({
            "Kelas": ["No Call", "Right Whale Call"],
            "Probabilitas": prediction[0]
        })

        st.bar_chart(prob_df.set_index("Kelas"))

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")

# =====================================
# Footer
# =====================================
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>"
    "CNN 1D Time Series Classification • RightWhaleCalls Dataset"
    "</p>",
    unsafe_allow_html=True
)
