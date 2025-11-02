import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import tempfile

st.set_page_config(
    page_title="Audio Classification: Buka vs Tutup",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Arial', sans-serif;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        transform: scale(1.05);
    }
    .prediction-buka {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .prediction-tutup {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .feature-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
    }
    .st-file-uploader {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .st-file-uploader:hover {
        background-color: #e1efff;
    }
    .st-spinner {
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Audio Classification: Suara Buka vs Tutup</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        model = joblib.load('model_results/rf_model_buka_tutup.pkl')
        return model, True
    except FileNotFoundError:
        st.error("Model tidak ditemukan! Pastikan file 'saved_models/rf_model_buka_tutup.pkl' tersedia.")
        return None, False

def extract_features(y, sr=22050):
    """
    Extract audio features exactly like in your notebook
    """
    feat = {
        "mean": np.mean(y),
        "var": np.var(y),
        "skew": stats.skew(y),
        "kurt": stats.kurtosis(y),
        "rms": np.sqrt(np.mean(y**2)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
        "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    }
    
    hist, _ = np.histogram(y, bins=50, density=True)
    hist = hist[hist > 0]  
    feat["entropy"] = -np.sum(hist * np.log2(hist + 1e-10))
    
    return feat

def preprocess_audio(y, sr, target_sr=22050):
    """
    Preprocess audio exactly like in your notebook
    """
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # durasi tidak lebih dari 1 detik
    max_len = int(sr * 1.0)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]
    
    # Normalisasi
    y = y / np.max(np.abs(y) + 1e-6)
    
    return y, sr

def predict_audio(audio_file, model):
    """
    Predict audio classification - same logic as your notebook
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Preprocess
        y_processed, sr_processed = preprocess_audio(y, sr)
        
        # Extract features
        features = extract_features(y_processed, sr_processed)
        
        # Convert to DataFrame
        X_new = pd.DataFrame([features])
        
        # Predict
        pred_label = model.predict(X_new)[0]
        pred_proba = model.predict_proba(X_new)[0]
        
        return pred_label, pred_proba, features, y_processed, sr_processed
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None, None, None

def create_waveform_plot(y, sr, title="Audio Waveform"):
    """
    Create waveform plot using plotly
    """
    time = np.linspace(0, len(y)/sr, len(y))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=y,
        mode='lines',
        name='Amplitude',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def main():
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    st.sidebar.header("📋 Model Information")
    st.sidebar.info(f"""
    **Model:** Random Forest
    **Classes:** {', '.join(model.classes_)}
    **Trees:** {model.n_estimators}
    **Features:** {model.n_features_in_}
    **Preprocessing:** Audio normalization & 1-second duration
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Pilih file audio (format WAV). Suara dapat diambil dari link berikut: https://www.kaggle.com/datasets/muhammadridhoisdi/audio-recognition-buka-and-tutup",
            type=['wav'],
            help="Upload audio file yang ingin diklasifikasi (akan diproses menjadi 22050 Hz, 1 detik)"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name
            
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Classify Audio", type="primary"):
                
                with st.spinner("🔄 Processing audio..."):
                    prediction, probabilities, features, y_processed, sr_processed = predict_audio(
                        temp_file_path, model
                    )
                
                if prediction is not None:
                    with col2:
                        st.header("Classification Results")
                        
                        confidence = max(probabilities) * 100
                        
                        if prediction.lower() == 'buka':
                            st.markdown(f"""
                            <div class="prediction-box prediction-buka">
                                <h2>BUKA</h2>
                                <p><strong>Confidence: {confidence:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box prediction-tutup">
                                <h2>TUTUP</h2>
                                <p><strong>Confidence: {confidence:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.subheader("Prediction Probabilities")
                        
                        prob_text = f"""
                        **Probabilitas:**
                        - Buka: {probabilities[0]*100:.1f}%
                        - Tutup: {probabilities[1]*100:.1f}%
                        """
                        st.markdown(prob_text)
                        
                        prob_df = pd.DataFrame({
                            'Class': model.classes_,
                            'Probability': probabilities * 100
                        })
                        
                        fig = px.bar(
                            prob_df, 
                            x='Class', 
                            y='Probability',
                            title='Class Probabilities (%)',
                            color='Probability',
                            color_continuous_scale='RdYlBu_r',
                            text='Probability'
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.header("Audio Analysis")

                if y_processed is not None:
                    waveform_fig = create_waveform_plot(
                        y_processed, sr_processed, "Processed Audio Waveform (1 second, 22050 Hz)"
                    )
                    st.plotly_chart(waveform_fig, use_container_width=True)
                
                os.unlink(temp_file_path)
    
    st.header("Cara Menggunakan Aplikasi Ini")
    st.markdown("""
    1. **Upload** file audio (format: hanya bisa format WAV)
    2. **Click** tombol "Classify Audio" untuk memproses
    3. **Lihat** hasil prediksi dan confidence score
    4. **Analisis** waveform dan fitur audio yang diekstrak
    
    **Processing Details:**
    - Audio akan di-resample ke 22050 Hz
    - Durasi akan dipotong/diperpanjang menjadi tepat 1 detik
    - Normalisasi amplitudo akan diterapkan
    - 10 fitur akustik akan diekstrak untuk klasifikasi
    """)

if __name__ == "__main__":
    main()
