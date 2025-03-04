# 🎤 Voice-to-Gender-Mini Classification System

[![Streamlit App](https://img.shields.io/badge/Streamlit-Online-green)](https://voicetogender.streamlit.app/)

This is a **machine learning-based voice classification system** that predicts gender from audio files.  
It uses **pretrained models** (SVM, Random Forest, XGBoost, and Stacking) to analyze and classify voices.

🚀 **Try it now:** [voicetogender.streamlit.app](https://voicetogender.streamlit.app/)

---

## 📌 Features
✔ **Upload an audio file** (WAV, MP3, FLAC, OGG)  
✔ **Extracts acoustic features** using `librosa`  
✔ **Predicts gender** using a **Stacking Classifier** (SVM, Random Forest, XGBoost)  
✔ **Pretrained models are used** (no need to retrain)  
✔ **Visualize dataset** using various graphs  

---

## 🛠 Installation (Run Locally)
You can run this app on your local machine using **Python & Streamlit**.

### **Clone the Repository**
```sh
https://github.com/Abdullah-02-134212-098/voice-to-gender-mini.git

streamlit run app.py
