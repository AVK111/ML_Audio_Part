import os
import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1️⃣ AGGRESSIVE Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    
    # Expand ALL contractions aggressively
    text = text.replace("won't", "will not").replace("can't", "cannot")
    text = text.replace("n't", " not").replace("'re", " are")
    text = text.replace("'ve", " have").replace("'ll", " will")
    text = text.replace("'d", " would").replace("'m", " am")
    text = text.replace("i'm", "i am").replace("im", "i am")
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#', '', text)
    text = re.sub(r'[0-9]+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# -----------------------------
# 2️⃣ ULTRA SIMPLE: BINARY CLASSIFICATION
# -----------------------------
def group_emotions_binary(sentiment):
    """
    JUST 2 CLASSES for MAXIMUM accuracy!
    Positive vs Negative (remove neutral entirely for clearer separation)
    """
    positive = ['happiness', 'love', 'fun', 'relief', 'enthusiasm']
    negative = ['sadness', 'anger', 'hate', 'empty', 'worry']
    
    sentiment_lower = sentiment.lower()
    
    if sentiment_lower in positive:
        return 'positive'
    elif sentiment_lower in negative:
        return 'negative'
    else:
        return None  # We'll filter these out


# -----------------------------
# 3️⃣ Train Model - ULTRA OPTIMIZED
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "text_emotion_binary.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer_binary.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "label_encoder_binary.pkl")

if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path)):
    
    print("🚀 Training BINARY Classification Model (HIGHEST ACCURACY)...")
    print("Strategy: Positive vs Negative ONLY (removing neutral/unclear cases)")
    
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "tweet_emotions.csv"))
    data = data[['sentiment', 'content']].dropna()
    
    print(f"\n📊 Original dataset: {len(data)} samples")
    
    # Group to binary
    data['binary_sentiment'] = data['sentiment'].apply(group_emotions_binary)
    
    # ✅ REMOVE NEUTRAL - Keep only clear positive/negative
    data = data[data['binary_sentiment'].notna()].copy()
    
    print(f"✨ After filtering to binary: {len(data)} samples")
    print("\n📊 Class Distribution:")
    print(data['binary_sentiment'].value_counts())
    
    # Clean text
    data['clean_content'] = data['content'].apply(clean_text)
    
    # Remove very short or very long texts
    data = data[data['clean_content'].str.split().str.len().between(3, 50)].copy()
    
    print(f"📝 After text filtering: {len(data)} samples")
    
    # Remove duplicates
    data = data.drop_duplicates(subset=['clean_content'])
    print(f"🔍 After removing duplicates: {len(data)} samples")
    
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['binary_sentiment'])
    
    print(f"\n🎯 Final classes: {encoder.classes_}")
    
    # ✅ OPTIMIZED TF-IDF for binary classification
    vectorizer = TfidfVectorizer(
        max_features=15000,  # More features for binary
        ngram_range=(1, 4),  # Up to 4-grams
        min_df=3,
        max_df=0.8,
        stop_words='english',
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'\b\w+\b',
        lowercase=True
    )
    
    X = vectorizer.fit_transform(data['clean_content'])
    
    print(f"\n📊 Feature matrix: {X.shape}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # ✅ BEST MODEL: Logistic Regression with optimal params for binary
    print("\n🔄 Training Logistic Regression (optimized for binary)...")
    
    model = LogisticRegression(
        max_iter=3000,
        C=5.0,  # Higher C for better fit
        penalty='l2',
        solver='saga',
        class_weight='balanced',
        random_state=42,
        verbose=0,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train) * 100
    test_accuracy = accuracy_score(y_test, y_pred_test) * 100
    
    print(f"\n✅ Training Accuracy: {train_accuracy:.2f}%")
    print(f"✅ Test Accuracy: {test_accuracy:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=encoder.classes_))
    
    # Show top features for each class
    feature_names = vectorizer.get_feature_names_out()
    
    for class_idx, class_name in enumerate(encoder.classes_):
        coef = model.coef_[class_idx] if len(encoder.classes_) > 2 else model.coef_[0]
        if class_idx == 0 and len(encoder.classes_) == 2:
            coef = -coef  # Flip for first class in binary
        
        top_indices = np.argsort(coef)[-15:][::-1]
        print(f"\n🔝 Top 15 words for '{class_name}':")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
    
    # Save
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(encoder, encoder_path)
    print("\n💾 Model saved successfully!")
    
else:
    print("✅ Loading pre-trained binary model...")

# Load model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
encoder = joblib.load(encoder_path)


# -----------------------------
# 4️⃣ Helper Function
# -----------------------------
def get_stress_and_depression(emotion, confidence):
    """Map binary emotions to stress levels"""
    if emotion == 'negative':
        if confidence > 0.8:
            return "High Stress", "Significant Depression Risk"
        else:
            return "Moderate-High Stress", "Possible Depression Risk"
    else:  # positive
        if confidence > 0.8:
            return "Low Stress", "Stable Mental State"
        else:
            return "Low-Moderate Stress", "Generally Stable"


# -----------------------------
# 5️⃣ Prediction Function
# -----------------------------
def predict_text_stress(user_input=None):
    """
    Binary sentiment prediction (positive/negative only)
    """
    if user_input is None:
        user_input = input("Enter your text (how are you feeling?): ")

    try:
        text_clean = clean_text(user_input)
        
        if len(text_clean.split()) < 2:
            print("⚠️  Warning: Input text is very short. Results may be less accurate.")
        
        # Vectorize
        text_vec = vectorizer.transform([text_clean])
        
        # Predict
        emotion_pred = model.predict(text_vec)
        emotion_proba = model.predict_proba(text_vec)
        
        emotion_label = encoder.inverse_transform(emotion_pred)[0]
        confidence = np.max(emotion_proba)
        
        # Get both class probabilities
        all_probs = [(encoder.classes_[i], emotion_proba[0][i]) for i in range(len(encoder.classes_))]
        all_probs.sort(key=lambda x: x[1], reverse=True)
        
        stress_level, depression_risk = get_stress_and_depression(emotion_label, confidence)
        
        print(f"\n💬 Input Text: {user_input}")
        print(f"🪞 Predicted Emotion: {emotion_label.upper()} (Confidence: {confidence*100:.1f}%)")
        print(f"\n📊 Sentiment Breakdown:")
        for emotion, prob in all_probs:
            bar = "█" * int(prob * 50)
            print(f"  {emotion.capitalize()}: {prob*100:.1f}% {bar}")
        print(f"\n🧠 Stress Level: {stress_level}")
        print(f"💭 Depression Indicator: {depression_risk}")

        return emotion_label, stress_level, depression_risk, confidence

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", "Normal stress", "No major depression sign", 0.0


# -----------------------------
# 6️⃣ Test Function
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING BINARY EMOTION DETECTION MODEL (Positive vs Negative)")
    print("="*70)
    
    test_texts = [
        "I am so happy and excited about this!",
        "I feel terrible and sad today",
        "This makes me so angry and frustrated",
        "I'm scared and worried about the future",
        "Life is wonderful and beautiful",
        "I hate everything right now",
        "Having a great time with friends!",
        "I'm so depressed and lonely",
        "Absolutely love this amazing day!",
        "Everything is going wrong, I feel awful"
    ]
    
    correct = 0
    expected = ['positive', 'negative', 'negative', 'negative', 'positive', 
                'negative', 'positive', 'negative', 'positive', 'negative']
    
    for i, text in enumerate(test_texts):
        result = predict_text_stress(text)
        if result[0] == expected[i]:
            correct += 1
            print("✅ CORRECT")
        else:
            print(f"❌ WRONG (Expected: {expected[i]})")
        print("-" * 70)
    
    print(f"\n🎯 Manual Test Accuracy: {correct}/{len(test_texts)} = {correct/len(test_texts)*100:.1f}%")