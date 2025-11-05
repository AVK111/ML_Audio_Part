import joblib
import numpy as np
import librosa

# Load your trained model
model = joblib.load('improved_emotion_model.pkl')
scaler = joblib.load('improved_scaler.pkl')
encoder = joblib.load('emotion_encoder.pkl')

def extract_features(file_path):
    """Extract features (same as training)"""
    y, sr = librosa.load(file_path, sr=22050, duration=3.0)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    mfccs_max = np.max(mfccs.T, axis=0)
    mfccs_min = np.min(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    chroma_std = np.std(chroma.T, axis=0)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    contrast_std = np.std(contrast.T, axis=0)
    
    # Other features
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)
    
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean, sc_std = np.mean(sc), np.std(sc)
    
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)
    sr_mean, sr_std = np.mean(sr_feat), np.std(sr_feat)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_mean = np.mean(mel.T, axis=0)
    mel_std = np.std(mel.T, axis=0)
    
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    
    rms = librosa.feature.rms(y=y)
    rms_mean, rms_std = np.mean(rms), np.std(rms)
    
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
    pitch_max = np.max(pitch_values) if len(pitch_values) > 0 else 0
    pitch_min = np.min(pitch_values) if len(pitch_values) > 0 else 0
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return np.hstack([
        mfccs_mean, mfccs_std, mfccs_max, mfccs_min,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        zcr_mean, zcr_std, sc_mean, sc_std,
        sr_mean, sr_std,
        mel_mean, mel_std,
        tonnetz_mean,
        rms_mean, rms_std,
        pitch_mean, pitch_std, pitch_max, pitch_min,
        tempo
    ])

def predict(audio_file):
    """Predict emotion from audio"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {audio_file}")
    print(f"{'='*60}\n")
    
    # Extract and predict
    features = extract_features(audio_file).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    emotion = encoder.inverse_transform([prediction])[0]
    probs = model.predict_proba(features_scaled)[0]
    
    # Display results
    print(f"🎯 EMOTION: {emotion.upper()}")
    print(f"📊 Confidence: {max(probs):.1%}\n")
    
    print("All Probabilities:")
    for e, p in zip(encoder.classes_, probs):
        bar = "█" * int(p * 30)
        print(f"  {e:10s}: {p:5.1%} {bar}")
    
    # Stress analysis
    if emotion in ['sad', 'fearful', 'angry']:
        stress = "🔴 HIGH STRESS"
        score = 0.85 * max(probs)
    elif emotion in ['neutral', 'calm']:
        stress = "🟢 LOW STRESS"
        score = 0.2 * max(probs)
    else:
        stress = "🟡 NORMAL"
        score = 0.45 * max(probs)
    
    print(f"\n{'='*60}")
    print(f"Stress Level: {stress}")
    print(f"Stress Score: {score:.2f}/1.00")
    print(f"{'='*60}\n")

# Test your audio files
if __name__ == "__main__":
    # Method 1: Direct file path
    predict("D:\Audio ML Project\Audio_Model\Sample_Audio\sample_audio.wav")
    
    # Method 2: Interactive
    # while True:
    #     file = input("\nEnter audio file (or 'quit'): ").strip()
    #     if file.lower() == 'quit':
    #         break
    #     try:
    #         predict(file)
    #     except Exception as e:
    #         print(f"Error: {e}")