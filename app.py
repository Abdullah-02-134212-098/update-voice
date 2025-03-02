import streamlit as st
import numpy as np
import pandas as pd
import librosa
import os
import pickle
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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
current_app_mode = st.sidebar.selectbox("Choose the app mode", ["Train Classifiers", "Load Audio File", "Visualize Data"], key="app_mode_selection")

# Check if the app mode has changed and reset session state if necessary
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

def load_classifiers():
    global classifiers, le
    classifiers = {}
    for model_name in ['SVM', 'Random Forest', 'XGBoost']:
        if os.path.exists(f'{model_name}_model.pkl'):
            with open(f'{model_name}_model.pkl', 'rb') as f:
                classifiers[model_name] = pickle.load(f)
    if os.path.exists('stacking_model.pkl'):
        with open('stacking_model.pkl', 'rb') as f:
            classifiers['Stacking'] = pickle.load(f)
    if os.path.exists('label_encoder.pkl'):
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

# 📌 **NO MICROPHONE RECORDING - USE FILE UPLOAD INSTEAD**
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
if current_app_mode == "Train Classifiers":
    st.title("Train Classifiers")
    df = pd.read_csv('gendervoice.csv')
    df = df.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label']) 
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    
    X = df[['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt']]
    y = df['Label']
    classifiers = {
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier()
    }

    for name, clf in classifiers.items():
        clf.fit(X, y)
        with open(f'{name}_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        st.success(f'{name} trained successfully')

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

elif current_app_mode == "Load Audio File":
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
    df = df.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label'])
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    load_classifiers()
    st.write(df.head())

