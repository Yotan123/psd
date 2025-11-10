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
# Import dari streamlit-audiorec
from streamlit_audiorec import st_audiorec

st.set_page_config(
    page_title="Audio Classifier - Aksi & Pembicara", # Updated title
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Updated CSS styling with modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 15px;
        color: transparent;
    }

    .prediction-container {
        background: linear-gradient(145deg, #f0f0f0, #ffffff);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    .prediction-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }

    .prediction-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }

    .prediction-error {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }

    .feature-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }

    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        transform: scale(1.02);
    }

    .metric-card {
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    .audio-player {
        background: linear-gradient(145deg, #f0f0f0, #ffffff);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .info-section {
        background: linear-gradient(145deg, #e3f2fd, #ffffff);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
    }

    .button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .button-record {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border: none;
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }

    .button-record:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
    }

    .button-stop {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border: none;
        color: #333;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }

    .button-stop:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 154, 158, 0.4);
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .pulse-animation {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .glow-effect {
        position: relative;
    }

    .glow-effect::after {
        content: '';
        position: absolute;
        top: -5px;
        left: -5px;
        right: -5px;
        bottom: -5px;
        background: linear-gradient(45deg, #667eea, #764ba2, #4facfe, #00f2fe);
        border-radius: 20px;
        z-index: -1;
        opacity: 0.3;
        filter: blur(10px);
    }

    .recording-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #ff6b6b;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1s infinite;
    }

    .tab-content {
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .microphone-section {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }

    .microphone-section h3 {
        color: white;
        margin: 0 0 1rem 0;
    }

    .microphone-section p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient effect
st.markdown('<h1 class="main-title">🎧 Klasifikasi Audio: Aksi Buka/Tutup & Pengenalan Pembicara</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_models(): # Renamed function
    """Load the trained Random Forest models (action and speaker)"""
    action_model = None
    speaker_model = None
    action_model_path = os.path.join('model_results', 'rf_model_buka_tutup_tuned.pkl')
    speaker_model_path = os.path.join('model_results', 'rf_model_speaker_recognition_updated.pkl')
    models_loaded = True
    loaded_paths = {}

    try:
        if os.path.exists(action_model_path):
            action_model = joblib.load(action_model_path)
            loaded_paths['action_model'] = action_model_path
            st.success(f"✅ Model Aksi dimuat dari: {action_model_path}")
        else:
            st.error(f"❌ Model Aksi tidak ditemukan di: {action_model_path}")
            models_loaded = False

        if os.path.exists(speaker_model_path):
            speaker_model = joblib.load(speaker_model_path)
            loaded_paths['speaker_model'] = speaker_model_path
            st.success(f"✅ Model Pembicara dimuat dari: {speaker_model_path}")
        else:
            st.error(f"❌ Model Pembicara tidak ditemakan di: {speaker_model_path}")
            models_loaded = False

    except Exception as e:
        st.error(f"❌ Error saat memuat satu atau lebih model: {str(e)}")
        st.info("Pastikan file model ada dan dalam format yang benar.")
        models_loaded = False

    return action_model, speaker_model, models_loaded, loaded_paths

def extract_features(y, sr=22050):
    """
    Extract audio features exactly like in your notebook
    """
    try:
        feat = {
            "mean": np.mean(y),
            "var": np.var(y),
            "skew": stats.skew(y),
            "kurt": stats.kurtosis(y),
            "rms": np.sqrt(np.mean(y**2)),
            "zcr": np.mean(librosa.feature.zero_crossing_rate(y)[0]),
            "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
            "bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
            "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        }

        hist, _ = np.histogram(y, bins=50, density=True)
        hist = hist[hist > 0]
        feat["entropy"] = -np.sum(hist * np.log2(hist + 1e-10))

        return feat
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None

def preprocess_audio(y, sr, target_sr=22050):
    """
    Preprocess audio exactly like in your notebook
    """
    try:
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
        y = y / (np.max(np.abs(y)) + 1e-6)

        return y, sr
    except Exception as e:
        st.error(f"Audio preprocessing error: {str(e)}")
        return None, None

def predict_audio(audio_file, action_model, speaker_model): # Updated signature
    """
    Predict audio classification and speaker recognition
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)  # Load with original sample rate

        # Preprocess
        y_processed, sr_processed = preprocess_audio(y, sr)

        if y_processed is None:
            return None, None, None, None, None, None, None # Updated return for None

        # Extract features
        features = extract_features(y_processed, sr_processed)

        if features is None:
            return None, None, None, None, None, None, None # Updated return for None

        # Convert to DataFrame
        X_new = pd.DataFrame([features])

        # Predict action
        pred_action_label = action_model.predict(X_new)[0]
        pred_action_proba = action_model.predict_proba(X_new)[0]

        # Predict speaker
        pred_speaker_label = speaker_model.predict(X_new)[0]
        pred_speaker_proba = speaker_model.predict_proba(X_new)[0]

        # --- START: Modified logic for speaker prediction --- #
        if pred_speaker_label == 'Unknown':
            # Find indices for 'Asep' and 'Yotan'
            asep_idx = np.where(speaker_model.classes_ == 'Asep')[0]
            yotan_idx = np.where(speaker_model.classes_ == 'Yotan')[0]

            if len(asep_idx) > 0 and len(yotan_idx) > 0:
                prob_asep = pred_speaker_proba[asep_idx[0]]
                prob_yotan = pred_speaker_proba[yotan_idx[0]]

                if prob_asep > prob_yotan:
                    pred_speaker_label = 'Asep'
                    speaker_confidence = prob_asep
                else:
                    pred_speaker_label = 'Yotan'
                    speaker_confidence = prob_yotan
            else:
                # Fallback if Asep or Yotan not in classes (shouldn't happen with updated model)
                speaker_confidence = max(pred_speaker_proba)
        else:
            speaker_confidence = max(pred_speaker_proba)
        # --- END: Modified logic for speaker prediction --- #

        return pred_action_label, pred_action_proba, pred_speaker_label, pred_speaker_proba, features, y_processed, sr_processed

    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        return None, None, None, None, None, None, None # Updated return for None

def create_waveform_plot(y, sr, title="Audio Waveform"):
    """
    Create waveform plot using plotly
    """
    try:
        time = np.linspace(0, len(y)/sr, len(y))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=y,
            mode='lines',
            name='Amplitude',
            line=dict(color='#667eea', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time (seconds)',
            yaxis_title='Amplitudo',
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        return fig
    except Exception as e:
        st.error(f"Waveform plot error: {str(e)}")
        return None

def main():
    # Load models
    action_model, speaker_model, models_loaded, loaded_paths = load_models() # Updated call

    if not models_loaded:
        st.error("🚨 **CRITICAL ERROR**: Satu atau lebih model tidak dapat dimuat. Aplikasi tidak dapat berjalan tanpa model.") # Updated message
        st.stop()

    # Sidebar with model info
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🤖 Informasi Model</div>', unsafe_allow_html=True) # Updated title

        if action_model is not None:
            st.info(f"""
            **🎯 Model Aksi (Buka/Tutup):**
            **Tipe:** Random Forest
            **Kelas:** {', '.join(action_model.classes_)}
            **Path:** {loaded_paths.get('action_model', 'N/A')}
            """)
        if speaker_model is not None:
            st.info(f"""
            **🎤 Model Pembicara (Asep/Yotan):**
            **Tipe:** Random Forest
            **Kelas:** {', '.join(speaker_model.classes_)}
            **Path:** {loaded_paths.get('speaker_model', 'N/A')}
            """)

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Rekam Suara"])

    with tab1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.header("🎤 Unggah & Analisis Audio")

        uploaded_file = st.file_uploader(
            "Unggah file audio WAV untuk klasifikasi",
            type=['wav'],
            help="Unggah file audio WAV (akan diproses menjadi 1 detik @ 22050 Hz)"
        )

        if uploaded_file is not None:
            # File details
            file_details = {
                "📁 Nama File": uploaded_file.name,
                "📊 Ukuran File": f"{uploaded_file.size / 1024:.2f} KB",
                "🎵 Status": "✅ Siap untuk diproses"
            }

            st.json(file_details)

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name

            # Audio player
            st.markdown('<div class="audio-player">', unsafe_allow_html=True)
            st.audio(temp_file_path, format='audio/wav')
            st.markdown('</div>', unsafe_allow_html=True)

            # Classification button
            if st.button("🔍 Analisis Audio", type="primary"):
                with st.spinner("🔄 Memproses audio... Ini mungkin memakan waktu beberapa detik..."):
                    pred_action_label, pred_action_proba, pred_speaker_label, pred_speaker_proba, features, y_processed, sr_processed = predict_audio(
                        temp_file_path, action_model, speaker_model # Updated call
                    )

                if pred_action_label is not None: # Updated check
                    # Results in col2
                    col1, col2 = st.columns([1, 1])
                    with col2:
                        st.markdown('<div class="prediction-container glow-effect">', unsafe_allow_html=True)
                        st.header("📊 Hasil Klasifikasi")

                        # Display Action Prediction
                        action_confidence = max(pred_action_proba) * 100
                        predicted_action_class = pred_action_label.lower()
                        if predicted_action_class == 'buka':
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                                <h2 style="color: white; margin: 0;">🎯 PREDIKSI AKSI: BUKA</h2>
                                <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Confidence: <strong>{action_confidence:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
                                <h2 style="color: white; margin: 0;">🎯 PREDIKSI AKSI: TUTUP</h2>
                                <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Confidence: <strong>{action_confidence:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Display Speaker Prediction
                        speaker_confidence = max(pred_speaker_proba) * 100
                        st.subheader("🎤 Prediksi Pembicara")
                        # The predict_audio function now guarantees pred_speaker_label is Asep or Yotan
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #ffd700 0%, #ffa500 100%);">
                            <h2 style="color: white; margin: 0;">👤 PEMBICARA: {pred_speaker_label.upper()}</h2>
                            <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Confidence: <strong>{speaker_confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability details for Action
                        st.subheader("📈 Detail Probabilitas Aksi")
                        prob_action_df = pd.DataFrame({
                            'Kelas': action_model.classes_,
                            'Probabilitas (%)': pred_action_proba * 100
                        })
                        fig_action = px.bar(
                            prob_action_df,
                            x='Kelas',
                            y='Probabilitas (%)',
                            title='Distribusi Probabilitas Aksi',
                            color='Probabilitas (%)',
                            color_continuous_scale='RdYlGn',
                            text='Probabilitas (%)'
                        )
                        fig_action.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_action.update_layout(
                            showlegend=False,
                            height=300, # Smaller height
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_action, use_container_width=True)

                        # Probability details for Speaker
                        st.subheader("📈 Detail Probabilitas Pembicara")
                        # Re-calculate prob_speaker_df to exclude 'Unknown' if it was originally predicted and overridden
                        speaker_classes_filtered = [cls for cls in speaker_model.classes_ if cls != 'Unknown']
                        speaker_proba_filtered = [proba for cls, proba in zip(speaker_model.classes_, pred_speaker_proba) if cls != 'Unknown']

                        prob_speaker_df = pd.DataFrame({
                            'Kelas': speaker_classes_filtered,
                            'Probabilitas (%)': np.array(speaker_proba_filtered) * 100
                        })
                        fig_speaker = px.bar(
                            prob_speaker_df,
                            x='Kelas',
                            y='Probabilitas (%)',
                            title='Distribusi Probabilitas Pembicara',
                            color='Probabilitas (%)',
                            color_continuous_scale='RdYlGn',
                            text='Probabilitas (%)'
                        )
                        fig_speaker.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_speaker.update_layout(
                            showlegend=False,
                            height=300, # Smaller height
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_speaker, use_container_width=True)

                        # Feature analysis
                        st.subheader("🔍 Analisis Fitur")
                        if features:
                            feature_cols = st.columns(3)
                            # feature_names = list(features.keys()) # This line is not used
                            for i, (name, value) in enumerate(features.items()):
                                col_idx = i % 3
                                with feature_cols[col_idx]:
                                    st.markdown(f"""
                                    <div class="feature-card">
                                        <strong>{name.upper()}</strong><br>
                                        <small>{value:.4f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.markdown('</div>', unsafe_allow_html=True)

                # Waveform analysis
                if y_processed is not None:
                    st.markdown('<div class="info-section">', unsafe_allow_html=True)
                    st.subheader("📊 Visualisasi Audio")
                    waveform_fig = create_waveform_plot(
                        y_processed, sr_processed, "Waveform Audio (1 detik, 22050 Hz)"
                    )
                    if waveform_fig:
                        st.plotly_chart(waveform_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Clean up temp file
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.header("🎤 Rekam Suara")

        # Gunakan st_audiorec
        wav_audio_data = st_audiorec()

        if wav_audio_data:
            st.audio(wav_audio_data, format='audio/wav')

            # Simpan data audio ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(wav_audio_data)
                recorded_file_path = tmp_file.name

            # Analyze button
            if st.button("🔍 Analisis Audio Rekaman", type="primary"):
                with st.spinner("🔄 Memproses audio rekaman..."):
                    pred_action_label, pred_action_proba, pred_speaker_label, pred_speaker_proba, features, y_processed, sr_processed = predict_audio(
                        recorded_file_path, action_model, speaker_model # Updated call
                    )

                if pred_action_label is not None: # Updated check
                    col1, col2 = st.columns([1, 1])
                    with col2:
                        st.markdown('<div class="prediction-container glow-effect">', unsafe_allow_html=True)
                        st.header("📊 Hasil Klasifikasi")

                        # Display Action Prediction
                        action_confidence = max(pred_action_proba) * 100
                        predicted_action_class = pred_action_label.lower()
                        if predicted_action_class == 'buka':
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                                <h2 style="color: white; margin: 0;">🎯 PREDIKSI AKSI: BUKA</h2>
                                <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Confidence: <strong>{action_confidence:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
                                <h2 style="color: white; margin: 0;">🎯 PREDIKSI AKSI: TUTUP</h2>
                                <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Confidence: <strong>{action_confidence:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Display Speaker Prediction
                        speaker_confidence = max(pred_speaker_proba) * 100
                        st.subheader("🎤 Prediksi Pembicara")
                        # The predict_audio function now guarantees pred_speaker_label is Asep or Yotan
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #ffd700 0%, #ffa500 100%);">
                            <h2 style="color: white; margin: 0;">👤 PEMBICARA: {pred_speaker_label.upper()}</h2>
                            <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Confidence: <strong>{speaker_confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability details for Action
                        st.subheader("📈 Detail Probabilitas Aksi")
                        prob_action_df = pd.DataFrame({
                            'Kelas': action_model.classes_,
                            'Probabilitas (%)': pred_action_proba * 100
                        })
                        fig_action = px.bar(
                            prob_action_df,
                            x='Kelas',
                            y='Probabilitas (%)',
                            title='Distribusi Probabilitas Aksi',
                            color='Probabilitas (%)',
                            color_continuous_scale='RdYlGn',
                            text='Probabilitas (%)'
                        )
                        fig_action.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_action.update_layout(
                            showlegend=False,
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_action, use_container_width=True)

                        # Probability details for Speaker
                        st.subheader("📈 Detail Probabilitas Pembicara")
                        # Re-calculate prob_speaker_df to exclude 'Unknown' if it was originally predicted and overridden
                        speaker_classes_filtered = [cls for cls in speaker_model.classes_ if cls != 'Unknown']
                        speaker_proba_filtered = [proba for cls, proba in zip(speaker_model.classes_, pred_speaker_proba) if cls != 'Unknown']

                        prob_speaker_df = pd.DataFrame({
                            'Kelas': speaker_classes_filtered,
                            'Probabilitas (%)': np.array(speaker_proba_filtered) * 100
                        })
                        fig_speaker = px.bar(
                            prob_speaker_df,
                            x='Kelas',
                            y='Probabilitas (%)',
                            title='Distribusi Probabilitas Pembicara',
                            color='Probabilitas (%)',
                            color_continuous_scale='RdYlGn',
                            text='Probabilitas (%)'
                        )
                        fig_speaker.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_speaker.update_layout(
                            showlegend=False,
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_speaker, use_container_width=True)

                        # Feature analysis
                        st.subheader("🔍 Analisis Fitur")
                        if features:
                            feature_cols = st.columns(3)
                            for i, (name, value) in enumerate(features.items()):
                                col_idx = i % 3
                                with feature_cols[col_idx]:
                                    st.markdown(f"""
                                    <div class="feature-card">
                                        <strong>{name.upper()}</strong><br>
                                        <small>{value:.4f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.markdown('</div>', unsafe_allow_html=True)

                # Waveform analysis for recorded audio
                if y_processed is not None:
                    st.markdown('<div class="info-section">', unsafe_allow_html=True)
                    st.subheader("📊 Visualisasi Audio")
                    waveform_fig = create_waveform_plot(
                        y_processed, sr_processed, "Waveform Audio (1 detik, 22050 Hz)"
                    )
                    if waveform_fig:
                        st.plotly_chart(waveform_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Clean up temp file
                os.unlink(recorded_file_path)

        st.markdown('</div>', unsafe_allow_html=True)

    # Info section
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.header("ℹ️ Panduan Penggunaan")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
        ### 📋 Langkah-langkah:
        1. **Unggah** file audio WAV atau **Rekam** langsung (jika didukung)
        2. **Klik** tombol "Analisis Audio"
        3. **Lihat** hasil prediksi aksi (Buka/Tutup) dan pembicara (Asep/Yotan/Tidak Dikenal)
        4. **Analisis** fitur dan waveform

        ### 🎯 Fitur Utama:
        - Klasifikasi suara "Buka" vs "Tutup"
        - Pengenalan Pembicara "Asep" vs "Yotan"
        - Visualisasi waveform
        - Probabilitas klasifikasi
        - Ekstraksi fitur audio
        - Rekam suara langsung (jika didukung)
        """)

    with col_info2:
        st.markdown("""
        ### ⚙️ Prosesing:
        - Resampling: 22050 Hz
        - Durasi: 1 detik (potong/isi)
        - Normalisasi amplitudo
        - 10 fitur akustik diekstrak

        ### 📊 Model:
        - Random Forest Classifier (untuk Aksi)
        - Random Forest Classifier (untuk Pembicara)
        - Multi-feature extraction
        - Real-time prediction
        - Cloud deployment friendly
        """)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()