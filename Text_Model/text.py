# Import libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv(r"tweet_emotions.csv")

print("Columns in dataset:", data.columns)
print(data.head())

# Rename columns if necessary
data.rename(columns={'content': 'text', 'sentiment': 'emotion'}, inplace=True)

# Drop missing values
data.dropna(inplace=True)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # Remove URLs
    text = re.sub(r"@\w+", "", text)              # Remove mentions
    text = re.sub(r"#\w+", "", text)              # Remove hashtags
    text = re.sub(r"[0-9]+", "", text)            # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)           # Remove punctuation
    text = text.strip()
    return text

data['clean_text'] = data['text'].apply(clean_text)

# Train-test split
X = data['clean_text']
y = data['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# Apply SMOTE for class balance
X_tfidf = tfidf.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_tfidf, y_train)

# Logistic Regression model
model = LogisticRegression(max_iter=200, solver='saga', class_weight='balanced')

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
}
grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_res, y_res)

best_model = grid.best_estimator_
print("\n✅ Best parameters found:", grid.best_params_)

# Evaluate model
X_test_tfidf = tfidf.transform(X_test)
y_pred = best_model.predict(X_test_tfidf)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Save model and vectorizer
import joblib
joblib.dump(best_model, "text_emotion_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")
