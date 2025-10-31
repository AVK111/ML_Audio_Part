import os
import re
import string
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1️⃣ Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]+', '', text)
    text = re.sub(r'\#[\w]+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------
# 2️⃣ Train Model (only if not saved)
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "text_emotion_svc.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path)):
    print("🚀 Training new Text Emotion Model...")
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "tweet_emotions.csv"))
    data = data[['sentiment', 'content']].dropna()
    data['clean_content'] = data['content'].apply(clean_text)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data['sentiment'])

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(data['clean_content'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred) * 100)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save all components
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(encoder, encoder_path)
else:
    print("✅ Loading pre-trained Text Emotion Model...")

# -----------------------------
# 3️⃣ Load Model Components
# -----------------------------
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
encoder = joblib.load(encoder_path)


# -----------------------------
# 4️⃣ Helper Function
# -----------------------------
def get_stress_and_depression(emotion):
    high_stress = ['anger', 'fear', 'sadness']
    medium_stress = ['surprise', 'disgust']
    low_stress = ['joy', 'love']

    if emotion in high_stress:
        return "High Stress", "Possible Depression Risk"
    elif emotion in medium_stress:
        return "Moderate Stress", "Monitor Mental State"
    else:
        return "Low Stress", "Stable Mental State"


# -----------------------------
# 5️⃣ Prediction Function
# -----------------------------
def predict_text_stress(user_input=None):
    """
    Predicts text emotion, stress level, depression risk, and numeric score.
    Always returns a 4-tuple to avoid unpacking errors.
    """
    if user_input is None:
        user_input = input("Enter your text (how are you feeling?): ")

    try:
        text_clean = clean_text(user_input)
        text_vec = vectorizer.transform([text_clean])
        emotion_pred = model.predict(text_vec)
        emotion_label = encoder.inverse_transform(emotion_pred)[0]

        stress_level, depression_risk = get_stress_and_depression(emotion_label)

        print(f"\n💬 Input Text: {user_input}")
        print(f"🪞 Predicted Emotion: {emotion_label}")
        print(f"🧠 Stress Level: {stress_level}")
        print(f"💭 Depression Indicator: {depression_risk}")

        # Numeric score mapping
        if stress_level == "High Stress":
            score = 0.9
        elif stress_level == "Moderate Stress":
            score = 0.6
        else:
            score = 0.3

        # ✅ Return 4 values
        return emotion_label, stress_level, depression_risk, score

    except Exception as e:
        print("Text model error:", e)
        return "Error", "Normal stress", "No major depression sign", 0.0
