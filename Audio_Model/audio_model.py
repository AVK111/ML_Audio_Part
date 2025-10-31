import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# ✅ Load pre-trained model & scaler
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "audio_emotion_svc.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"⚠️ Error loading model/scaler: {e}")
    model, scaler = None, None

# Emotion label encoder (same as used during training)
emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
encoder = LabelEncoder()
encoder.fit(emotion_labels)


# -------------------------------
# 🎵 Feature Extraction
# -------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            raise ValueError("Empty audio file or no speech detected.")

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        features = np.hstack([mfccs, chroma, contrast])
        return features

    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None


# -------------------------------
# 🎧 Predict Emotion from Audio File
# -------------------------------
def predict_emotion(file_path):
    if model is None or scaler is None:
        print("❌ Model or scaler not loaded properly.")
        return "Error"

    features = extract_features(file_path)
    if features is None:
        return "Error"

    try:
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        emotion_label = encoder.inverse_transform([prediction])[0]
        return emotion_label
    except Exception as e:
        print(f"❌ Emotion prediction failed: {e}")
        return "Error"


# -------------------------------
# 🧠 Predict Stress Level from Audio
# -------------------------------
def predict_audio_stress(file_path):
    """
    Always returns a 4-tuple:
    (score, predicted_emotion, stress_label, depression_indicator)
    """
    safe_return = (0.0, "Error", "Normal stress", "No major depression sign")

    try:
        # Handle invalid path
        if not os.path.exists(file_path):
            if isinstance(file_path, str) and file_path.isalpha():
                predicted_emotion = file_path.lower()
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        else:
            predicted_emotion = predict_emotion(file_path)

        predicted_emotion = str(predicted_emotion).lower()

        # Emotion → Stress mapping
        if predicted_emotion in ["sad", "fearful", "angry", "sadness", "fear"]:
            stress_label = "High stress"
            score = 0.9
        elif predicted_emotion in ["neutral", "calm"]:
            stress_label = "Low stress"
            score = 0.2
        elif predicted_emotion in ["happy", "joy", "happiness", "excited"]:
            stress_label = "Low stress"
            score = 0.15
        else:
            stress_label = "Normal stress"
            score = 0.45

        depression_indicator = (
            "Possible depression indicator"
            if predicted_emotion in ["sad", "sadness"]
            else "No major depression sign"
        )

        return score, predicted_emotion, stress_label, depression_indicator

    except Exception as e:
        print(f"❌ predict_audio_stress exception: {e}")
        return safe_return
