def train_single_optimized_model(X_train, y_train, model_type='svm'):
    """
    Train a single optimized model (faster, less memory than ensemble)
    
    Parameters:
    -----------
    model_type : str
        'svm', 'rf', 'xgb', or 'mlp'
    """
    print(f"\n=== Training Single {model_type.upper()} Model ===")
    
    if model_type == 'svm':
        model = SVC(
            C=10, 
            gamma='scale', 
            kernel='rbf',
            probability=True, 
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        )
    elif model_type == 'xgb' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=1
        )
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    return model


"""
Enhanced Audio Emotion Recognition Model with Improved Accuracy
Target: 75%+ accuracy (from 68%)
"""

import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


# ===========================
# 1. ENHANCED FEATURE EXTRACTION
# ===========================

def extract_enhanced_features(file_path, sr=22050):
    """
    Extract comprehensive audio features for emotion recognition
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=3.0)
        
        if y.size == 0:
            raise ValueError("Empty audio file")

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # === MFCC Features (Voice Quality) ===
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_max = np.max(mfccs.T, axis=0)
        mfccs_min = np.min(mfccs.T, axis=0)
        
        # === Chroma Features (Pitch/Harmony) ===
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # === Spectral Contrast (Texture) ===
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)

        # === Zero Crossing Rate (Voice Quality) ===
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # === Spectral Centroid (Brightness) ===
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # === Spectral Rolloff (Shape) ===
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # === Mel Spectrogram ===
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        # === Tonnetz (Harmonic Features) ===
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        
        # === RMS Energy (Loudness) ===
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # === Pitch Features ===
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_max = np.max(pitch_values) if len(pitch_values) > 0 else 0
        pitch_min = np.min(pitch_values) if len(pitch_values) > 0 else 0

        # === Tempo ===
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Combine all features
        features = np.hstack([
            mfccs_mean, mfccs_std, mfccs_max, mfccs_min,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            zcr_mean, zcr_std,
            sc_mean, sc_std,
            sr_mean, sr_std,
            mel_mean, mel_std,
            tonnetz_mean,
            rms_mean, rms_std,
            pitch_mean, pitch_std, pitch_max, pitch_min,
            tempo
        ])
        
        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def extract_features_from_array(y, sr=22050):
    """
    Extract features from audio array (for augmentation)
    """
    try:
        if y.size == 0:
            return None

        # Trim silence
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

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # Mel
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        # Tonnetz
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        
        # RMS
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Pitch
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_max = np.max(pitch_values) if len(pitch_values) > 0 else 0
        pitch_min = np.min(pitch_values) if len(pitch_values) > 0 else 0

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        features = np.hstack([
            mfccs_mean, mfccs_std, mfccs_max, mfccs_min,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            zcr_mean, zcr_std,
            sc_mean, sc_std,
            sr_mean, sr_std,
            mel_mean, mel_std,
            tonnetz_mean,
            rms_mean, rms_std,
            pitch_mean, pitch_std, pitch_max, pitch_min,
            tempo
        ])
        
        return features

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None


# ===========================
# 2. DATA AUGMENTATION
# ===========================

def augment_audio(y, sr):
    """
    Generate augmented versions of audio for better training
    """
    augmented = [y]  # Original
    
    try:
        # Time stretch
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
        
        # Pitch shift
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
        
        # Add noise
        noise = np.random.randn(len(y))
        augmented.append(y + 0.005 * noise)
        
        # Volume change
        augmented.append(y * 1.25)
        augmented.append(y * 0.8)
        
    except Exception as e:
        print(f"Augmentation warning: {e}")
    
    return augmented


def load_and_augment_dataset(file_paths, labels, augment=True, use_cache=True):
    """
    Load dataset with optional augmentation and caching
    """
    cache_file = 'ravdess_features_cache.npz'
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print(f"Loading features from cache: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            print(f"Loaded {len(X)} cached samples")
            return X, y
        except Exception as e:
            print(f"Cache load failed: {e}. Processing from scratch...")
    
    X, y = [], []
    
    print(f"\nProcessing {len(file_paths)} audio files...")
    print("This will take 5-10 minutes on first run. Please wait...")
    print("(Features will be cached for faster future runs)\n")
    
    import time
    start_time = time.time()
    
    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        # Progress indicator every 50 files
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            if idx > 0:
                avg_time = elapsed / idx
                remaining = avg_time * (len(file_paths) - idx)
                print(f"Progress: {idx}/{len(file_paths)} ({idx/len(file_paths)*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f} minutes")
            else:
                print(f"Progress: {idx}/{len(file_paths)} ({idx/len(file_paths)*100:.1f}%)")
        
        try:
            audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
            
            if augment:
                augmented_audios = augment_audio(audio, sr)
            else:
                augmented_audios = [audio]
            
            for aug_audio in augmented_audios:
                features = extract_features_from_array(aug_audio, sr)
                if features is not None:
                    X.append(features)
                    y.append(label)
        
        except Exception as e:
            if idx % 100 == 0:  # Only show errors occasionally
                print(f"  Warning: Error loading {os.path.basename(file_path)}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    total_time = time.time() - start_time
    print(f"\n✓ Processing complete!")
    print(f"  Total samples: {len(X)}")
    print(f"  Processing time: {total_time/60:.1f} minutes")
    
    # Save to cache
    if use_cache:
        try:
            print(f"  Saving features to cache: {cache_file}")
            np.savez_compressed(cache_file, X=X, y=y)
            print(f"  ✓ Cache saved (next run will be instant!)")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")
    
    return X, y


# ===========================
# 3. MODEL TRAINING
# ===========================

def train_ensemble_model(X_train, y_train, use_grid_search=False):
    """
    Train ensemble model with multiple classifiers
    Fixed: Sequential training to avoid memory errors
    """
    print("\n=== Training Ensemble Model ===")
    
    # Optimized SVM
    if use_grid_search:
        print("Running Grid Search for SVM (this may take a while)...")
        svm_params = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.001, 0.01],
            'kernel': ['rbf']
        }
        svm = GridSearchCV(
            SVC(probability=True, class_weight='balanced', random_state=42),
            svm_params,
            cv=3,
            n_jobs=1,  # Changed from -1 to avoid memory issues
            verbose=1
        )
        svm.fit(X_train, y_train)
        print(f"Best SVM params: {svm.best_params_}")
        svm_classifier = svm.best_estimator_
    else:
        print("Training SVM...")
        svm_classifier = SVC(
            C=10, 
            gamma='scale', 
            kernel='rbf',
            probability=True, 
            class_weight='balanced',
            random_state=42
        )
        svm_classifier.fit(X_train, y_train)
        print("✓ SVM trained")
    
    # Random Forest
    print("Training Random Forest...")
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=1  # Changed from -1
    )
    rf_classifier.fit(X_train, y_train)
    print("✓ Random Forest trained")
    
    # Neural Network
    print("Training Neural Network...")
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    mlp_classifier.fit(X_train, y_train)
    print("✓ Neural Network trained")
    
    # Build ensemble
    estimators = [
        ('svm', svm_classifier),
        ('rf', rf_classifier),
        ('mlp', mlp_classifier)
    ]
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        xgb_classifier = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=1  # Changed from default
        )
        xgb_classifier.fit(X_train, y_train)
        print("✓ XGBoost trained")
        estimators.append(('xgb', xgb_classifier))
    
    # Create ensemble with n_jobs=1 to avoid parallel memory issues
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=1  # CRITICAL: Changed from -1 to 1 to prevent memory errors
    )
    
    print("Creating ensemble (no additional training needed)...")
    # Ensemble doesn't need to fit again since all estimators are already fitted
    # But we need to call fit for sklearn compatibility
    ensemble.fit(X_train, y_train)
    print("✓ Ensemble ready")
    
    return ensemble


def preprocess_data(X_train, y_train, X_test):
    """
    Preprocess data: handle imbalance, scale features
    """
    print("\n=== Preprocessing Data ===")
    
    # Handle class imbalance with SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training samples: {len(X_train)}")
    print(f"After SMOTE: {len(X_train_balanced)}")
    
    # Scale features (RobustScaler handles outliers better)
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train_balanced, X_test_scaled, scaler


# ===========================
# 4. COMPLETE TRAINING PIPELINE
# ===========================

def train_emotion_model(audio_files, labels, test_size=0.2, augment=True, grid_search=False, use_cache=True):
    """
    Complete training pipeline
    
    Parameters:
    -----------
    audio_files : list of str
        Paths to audio files
    labels : list of str
        Emotion labels for each audio file
    test_size : float
        Proportion of test set (default: 0.2)
    augment : bool
        Whether to use data augmentation (default: True)
    grid_search : bool
        Whether to use grid search for hyperparameters (default: False, slower)
    use_cache : bool
        Whether to cache extracted features (default: True, MUCH faster on reruns)
    
    Returns:
    --------
    model : trained model
    scaler : fitted scaler
    encoder : label encoder
    metrics : dict with accuracy and other metrics
    """
    
    # Label encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    
    print(f"Classes: {encoder.classes_}")
    print(f"Total samples: {len(audio_files)}")
    
    # Load and augment data
    print("\n=== Step 1: Loading Data ===")
    X, y = load_and_augment_dataset(audio_files, y_encoded, augment=augment, use_cache=use_cache)
    
    # Split data
    print("\n=== Step 2: Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Preprocess
    print("\n=== Step 3: Preprocessing ===")
    X_train_prep, y_train_prep, X_test_prep, scaler = preprocess_data(
        X_train, y_train, X_test
    )
    
    # Train model
    print("\n=== Step 4: Training Model ===")
    model = train_ensemble_model(X_train_prep, y_train_prep, use_grid_search=grid_search)
    
    # Evaluate
    print("\n=== Step 5: Evaluation ===")
    y_pred = model.predict(X_test_prep)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"TEST ACCURACY: {accuracy:.2%}")
    print(f"{'='*50}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    print("\n=== Cross-Validation ===")
    print("Running 5-fold cross-validation (this may take a few minutes)...")
    cv_scores = cross_val_score(model, X_train_prep, y_train_prep, cv=5, n_jobs=1)  # Changed from -1
    print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    metrics = {
        'test_accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_report(y_test, y_pred, target_names=encoder.classes_)
    }
    
    return model, scaler, encoder, metrics


def save_model(model, scaler, encoder, model_path='improved_emotion_model.pkl', 
               scaler_path='improved_scaler.pkl', encoder_path='emotion_encoder.pkl'):
    """
    Save trained model, scaler, and encoder
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Encoder saved to: {encoder_path}")


# ===========================
# 5. PREDICTION FUNCTIONS
# ===========================

def load_trained_model(model_path='improved_emotion_model.pkl',
                       scaler_path='improved_scaler.pkl',
                       encoder_path='emotion_encoder.pkl'):
    """
    Load pre-trained model, scaler, and encoder
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        return model, scaler, encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def predict_emotion(file_path, model, scaler, encoder):
    """
    Predict emotion from audio file
    """
    if model is None or scaler is None or encoder is None:
        return "Error: Model not loaded"
    
    features = extract_enhanced_features(file_path)
    if features is None:
        return "Error: Feature extraction failed"
    
    try:
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        emotion = encoder.inverse_transform([prediction])[0]
        
        # Get probability scores
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_probabilities': dict(zip(encoder.classes_, probabilities))
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error: Prediction failed"


def predict_stress_level(file_path, model, scaler, encoder):
    """
    Predict stress level from audio file
    Returns: (stress_score, emotion, stress_label, depression_indicator)
    """
    result = predict_emotion(file_path, model, scaler, encoder)
    
    if isinstance(result, str):  # Error occurred
        return (0.0, "Error", "Unknown", "Unknown")
    
    emotion = result['emotion'].lower()
    confidence = result['confidence']
    
    # Emotion to stress mapping
    if emotion in ['sad', 'fearful', 'angry', 'fear', 'sadness']:
        stress_label = "High stress"
        stress_score = 0.85 * confidence
    elif emotion in ['neutral', 'calm']:
        stress_label = "Low stress"
        stress_score = 0.2 * confidence
    elif emotion in ['happy', 'joy', 'surprised']:
        stress_label = "Low stress"
        stress_score = 0.15 * confidence
    else:
        stress_label = "Normal stress"
        stress_score = 0.45 * confidence
    
    depression_indicator = (
        "Possible depression indicator" if emotion in ['sad', 'sadness']
        else "No major depression sign"
    )
    
    return stress_score, emotion, stress_label, depression_indicator


# ===========================
# 6. RAVDESS DATASET LOADER
# ===========================

def load_ravdess_dataset(dataset_path="Datasets/Ravedess dataset"):
    """
    Load RAVDESS dataset from folder structure
    
    RAVDESS filename format: 03-01-06-01-02-01-12.wav
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    - Repetition (01 = 1st repetition, 02 = 2nd repetition)
    - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)
    
    Parameters:
    -----------
    dataset_path : str
        Path to RAVDESS dataset folder containing Actor_XX folders
    
    Returns:
    --------
    audio_files : list
        List of audio file paths
    labels : list
        List of emotion labels
    """
    
    # RAVDESS emotion mapping
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    audio_files = []
    labels = []
    
    print(f"Loading RAVDESS dataset from: {dataset_path}")
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return [], []
    
    # Iterate through Actor folders
    actor_folders = sorted([f for f in os.listdir(dataset_path) if f.startswith('Actor_')])
    
    if not actor_folders:
        print(f"Error: No Actor folders found in {dataset_path}")
        return [], []
    
    print(f"Found {len(actor_folders)} actor folders")
    
    for actor_folder in actor_folders:
        actor_path = os.path.join(dataset_path, actor_folder)
        
        if not os.path.isdir(actor_path):
            continue
        
        # Get all .wav files in actor folder
        wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            file_path = os.path.join(actor_path, wav_file)
            
            # Parse filename to get emotion
            # Format: 03-01-06-01-02-01-12.wav
            parts = wav_file.split('-')
            
            if len(parts) >= 3:
                emotion_code = parts[2]  # Third part is emotion
                
                if emotion_code in emotion_map:
                    audio_files.append(file_path)
                    labels.append(emotion_map[emotion_code])
    
    print(f"Total audio files loaded: {len(audio_files)}")
    
    # Print class distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for emotion, count in sorted(label_counts.items()):
        print(f"  {emotion}: {count}")
    
    return audio_files, labels


# ===========================
# 7. EXAMPLE USAGE
# ===========================

if __name__ == "__main__":
    """
    Complete example for training on RAVDESS dataset
    """
    
    print("="*60)
    print("TRAINING EMOTION RECOGNITION MODEL ON RAVDESS DATASET")
    print("="*60)
    
    # Load RAVDESS dataset
    dataset_path = "Datasets/Ravedess dataset"  # Adjust if needed
    audio_files, labels = load_ravdess_dataset(dataset_path)
    
    if len(audio_files) == 0:
        print("\nError: No audio files found. Please check:")
        print("1. Dataset path is correct")
        print("2. Folder structure: Datasets/Ravedess dataset/Actor_XX/*.wav")
        print("3. Actor folders exist (Actor_01, Actor_02, etc.)")
    else:
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        
        # Train the model
        model, scaler, encoder, metrics = train_emotion_model(
            audio_files, 
            labels,
            test_size=0.2,
            augment=True,        # Use data augmentation (recommended)
            grid_search=False    # Set True for hyperparameter tuning (much slower but better results)
        )
        
        # Save the trained model
        save_model(model, scaler, encoder)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"Cross-Validation Accuracy: {metrics['cv_mean']:.2%} (+/- {metrics['cv_std']:.2%})")
        
        # Test prediction on a random file
        print(f"\n{'='*60}")
        print("Testing prediction on sample file...")
        print(f"{'='*60}\n")
        
        test_file = audio_files[0]
        result = predict_emotion(test_file, model, scaler, encoder)
        print(f"Test file: {os.path.basename(test_file)}")
        print(f"Predicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"  {emotion}: {prob:.2%}")
        
        # Test stress prediction
        stress_score, emotion, stress_label, depression = predict_stress_level(
            test_file, model, scaler, encoder
        )
        print(f"\nStress Analysis:")
        print(f"  Emotion: {emotion}")
        print(f"  Stress Level: {stress_label}")
        print(f"  Stress Score: {stress_score:.2f}")
        print(f"  Depression Indicator: {depression}")
    
    print("\n" + "="*60)
    print("To use the trained model later:")
    print("="*60)
    print("""
# Load model
model, scaler, encoder = load_trained_model()

# Predict on new audio
result = predict_emotion("your_audio.wav", model, scaler, encoder)
print(result)
    """)