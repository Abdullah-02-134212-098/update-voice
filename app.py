import streamlit as st
import numpy as np
import pandas as pd
import librosa
import os
import pickle
from pydub import AudioSegment
from pydub.utils import which
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# ✅ Set FFmpeg Path for Streamlit Cloud
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

st.set_page_config(page_title="Voice Recognition System", page_icon=":microphone:", layout="wide")

# Initialize session state variables
if 'correct_label' not in st.session_state:
    st.session_state.correct_label = "Choose Gender"
if 'features' not in st.session_state:
    st.session_state.features = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'last_app_mode' not in st.session_state:
    st.session_state.last_app_mode = None

# Function to reset session state variables
def reset_session_state():
    st.session_state.correct_label = "Choose Gender"
    st.session_state.features = None
    st.session_state.prediction = None

# Sidebar for navigation
st.sidebar.title("Navigation")
current_app_mode = st.sidebar.selectbox("Choose the app mode", ["Load Audio File", "Visualize Data"], key="app_mode_selection")

# Reset session state when app mode changes
if st.session_state.last_app_mode != current_app_mode:
    reset_session_state()
    st.session_state.last_app_mode = current_app_mode

# Global variables
classifiers = {}
le = None

def extract_features(audio, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).flatten()
    meanfreq = np.mean(spectral_centroid)
    sd = np.std(spectral_centroid)
    median = np.median(spectral_centroid)
    mode = float(stats.mode(spectral_centroid)[0])
    Q25 = np.percentile(spectral_centroid, 25)
    Q75 = np.percentile(spectral_centroid, 75)
    IQR = Q75 - Q25
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    return {
        "meanfreq": round(meanfreq / 1000, 2),
        "sd": round(sd / 1000, 2),
        "median": round(median / 1000, 2),
        "mode": round(mode / 1000, 2),
        "Q25": round(Q25 / 1000, 2),
        "Q75": round(Q75 / 1000, 2),
        "IQR": round(IQR / 1000, 2),
        "skew": round(spectral_contrast / 1000, 2),
        "kurt": round(spectral_rolloff / 1000, 2),
    }

# ✅ Fixed: Load Pretrained Models Instead of Training
def load_classifiers():
    global classifiers, le
    classifiers = {}

    model_files = ['SVM_model.pkl', 'Random Forest_model.pkl', 'XGBoost_model.pkl', 'stacking_model.pkl']
    missing_files = [f for f in model_files if not os.path.exists(f)]

    if missing_files:
        st.error(f"Missing model files: {', '.join(missing_files)}. Please check GitHub deployment.")
        st.stop()

    for model_name in ['SVM', 'Random Forest', 'XGBoost']:
        with open(f'{model_name}_model.pkl', 'rb') as f:
            classifiers[model_name] = pickle.load(f)

    with open('stacking_model.pkl', 'rb') as f:
        classifiers['Stacking'] = pickle.load(f)

    if os.path.exists('label_encoder.pkl'):
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    else:
        st.error("Label encoder is missing. Please upload `label_encoder.pkl` to GitHub.")
        st.stop()

# ✅ Fixed: Ensure Audio Loading Works
def load_and_scale_audio(file):
    try:
        audio = AudioSegment.from_file(file)
        audio = audio.set_frame_rate(22050).set_channels(1)  # Convert to mono, 22050Hz
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Normalize
        return samples, 22050
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None

# Main panel based on sidebar selection
if current_app_mode == "Load Audio File":
    st.title("Load Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"], key="upload_audio")
    load_classifiers()
    
    if uploaded_file is not None:
        audio, sr = load_and_scale_audio(uploaded_file)
        if audio is not None:
            features = extract_features(audio, sr)
            st.info("Extracted Features:")
            st.write(features)

            stacking_model = classifiers.get('Stacking')
            if stacking_model is None:
                st.error("Stacking model is not loaded. Please upload `stacking_model.pkl` to GitHub.")
                st.stop()

            base_models = [classifiers.get(name) for name in ['SVM', 'Random Forest', 'XGBoost']]
            base_predictions = np.zeros((1, len(base_models)))
            for i, model in enumerate(base_models):
                base_predictions[:, i] = model.predict_proba([list(features.values())])[:, 1]

            prediction = stacking_model.predict(base_predictions)
            predicted_label = le.inverse_transform(prediction)[0]
            st.success(f'Predicted Label: {predicted_label}')

elif current_app_mode == "Visualize Data":
    st.title("Visualize Data")
    df = pd.read_csv('gendervoice.csv')
    df = df.drop_duplicates()
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    load_classifiers()
    st.write(df.head())
