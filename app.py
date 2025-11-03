import streamlit as st
import pandas as pd
import joblib

# Memuat model dan scaler untuk setiap hari
@st.cache_resource
def load_models_and_scalers():
    # Memuat model dan scaler untuk hari 1 sampai hari 5
    model_day1 = joblib.load('knn_day1_model.pkl')
    model_day2 = joblib.load('knn_day2_model.pkl')
    model_day3 = joblib.load('knn_day3_model.pkl')
    model_day4 = joblib.load('knn_day4_model.pkl')
    model_day5 = joblib.load('knn_day5_model.pkl')
    
    scaler_day1 = joblib.load('scaler_day1.pkl')
    scaler_day2 = joblib.load('scaler_day2.pkl')
    scaler_day3 = joblib.load('scaler_day3.pkl')
    scaler_day4 = joblib.load('scaler_day4.pkl')
    scaler_day5 = joblib.load('scaler_day5.pkl')
    
    return model_day1, model_day2, model_day3, model_day4, model_day5, \
           scaler_day1, scaler_day2, scaler_day3, scaler_day4, scaler_day5

# Fungsi untuk kategorisasi berdasarkan nilai kadar NO₂
def get_kategori(nilai):
    """Kategorisasi berdasarkan threshold"""
    if nilai <= 0.000025:  # threshold rendah (misalnya)
        return "🟢 **RENDAH**", "Kadar NO₂ rendah, kualitas udara baik."
    elif nilai <= 0.000030:  # threshold sedang (misalnya)
        return "🟡 **SEDANG**", "Kadar NO₂ sedang, masih dalam batas aman."
    else:
        return "🔴 **TINGGI**", "Kadar NO₂ tinggi, waspadai kualitas udara!"

# Judul aplikasi
st.title("🌫️ **Prediksi Kadar NO₂ - Besok dan Lusa**")
st.caption("**Prediksi kadar NO₂ troposfer (mol/m²) menggunakan model KNN berdasarkan data Sentinel-5P.**")

# Memuat model dan scaler
model_day1, model_day2, model_day3, model_day4, model_day5, \
scaler_day1, scaler_day2, scaler_day3, scaler_day4, scaler_day5 = load_models_and_scalers()

# Meminta input kadar NO₂ untuk t-5, t-4, t-3, t-2, t-1
t5 = st.number_input("Masukkan kadar NO₂ untuk t-5 (5 hari yang lalu)", value=0.000025, format="%.6f")
t4 = st.number_input("Masukkan kadar NO₂ untuk t-4 (4 hari yang lalu)", value=0.000027, format="%.6f")
t3 = st.number_input("Masukkan kadar NO₂ untuk t-3 (3 hari yang lalu)", value=0.000029, format="%.6f")
t2 = st.number_input("Masukkan kadar NO₂ untuk t-2 (2 hari yang lalu)", value=0.000031, format="%.6f")
t1 = st.number_input("Masukkan kadar NO₂ untuk t-1 (Hari ini)", value=0.000030, format="%.6f")

# Pilihan model prediksi untuk hari 1 (besok) dan hari 2 (lusa)
prediksi_type = st.selectbox(
    "Pilih model untuk prediksi besok dan lusa:",
    ("1_hari", "3_hari"),
    format_func=lambda x: "Prediksi 1 Hari (Besok)" if x == "1_hari" else "Prediksi 3 Hari (Lusa)"
)

# Fungsi untuk prediksi
if st.button("🔮 **Prediksi Besok dan Lusa**", type="primary"):
    if prediksi_type == "1_hari":
        # Menambahkan fitur yang hilang (t-5, t-4, t-3, t-2) untuk konsistensi dengan model
        X = pd.DataFrame({'t-5': [t5], 't-4': [t4], 't-3': [t3], 't-2': [t2], 't-1': [t1]})
        # Menjaga urutan kolom konsisten dengan pelatihan
        X = X[['t-5', 't-4', 't-3', 't-2', 't-1']]  # Urutan kolom yang konsisten
        X_scaled = scaler_day1.transform(X)  # Transformasi dengan scaler
        y_pred_besok = model_day1.predict(X_scaled)[0]  # Prediksi
        kategori_besok, keterangan_besok = get_kategori(y_pred_besok)

        st.subheader("Hasil Prediksi Besok (t+1)")
        st.metric("NO₂ (Besok)", f"{y_pred_besok:.6f} mol/m²")
        st.markdown(kategori_besok)
        st.caption(keterangan_besok)
    
    if prediksi_type == "3_hari":
        # Menambahkan fitur yang hilang (t-5, t-4) untuk konsistensi dengan model
        X = pd.DataFrame({'t-5': [t5], 't-4': [t4], 't-3': [t3], 't-2': [t2], 't-1': [t1]})
        # Menjaga urutan kolom konsisten dengan pelatihan
        X = X[['t-5', 't-4', 't-3', 't-2', 't-1']]  # Urutan kolom yang konsisten
        X_scaled = scaler_day2.transform(X)  # Transformasi dengan scaler
        y_pred_besok, y_pred_lusa = model_day2.predict(X_scaled)  # Prediksi

        kategori_besok, keterangan_besok = get_kategori(y_pred_besok)
        kategori_lusa, keterangan_lusa = get_kategori(y_pred_lusa)
        
        st.subheader("Hasil Prediksi Besok (t+1) dan Lusa (t+2)")
        st.metric("NO₂ (Besok)", f"{y_pred_besok:.6f} mol/m²")
        st.markdown(kategori_besok)
        st.caption(keterangan_besok)
        
        st.metric("NO₂ (Lusa)", f"{y_pred_lusa:.6f} mol/m²")
        st.markdown(kategori_lusa)
        st.caption(keterangan_lusa)

# Menambahkan caption untuk menyelesaikan aplikasi
st.divider()
st.caption("Model: KNN Regression (1 dan 2 lag) | Data: 268 observasi | Metode Prediksi: KNN")
