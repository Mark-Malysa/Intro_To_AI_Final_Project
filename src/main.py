# src/main.py

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. NLTK setup
nltk.download('stopwords')
nltk.download('punkt')

# 2. Locate datasets folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "datasets"

# 3. List and load the two CSV files
print("Looking in:", DATA_DIR)
for f in DATA_DIR.iterdir():
    print("  ", f.name)

true_path = DATA_DIR / "True.csv"    
fake_path = DATA_DIR / "Fake.csv"   

# 4. Read and label
true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)

true_df["label"] = 'Real'
fake_df["label"] = 'Fake'

# 5. Merge & shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Loaded {len(df)} articles; label distribution:")
print(df["label"].value_counts())

# 6. Quick peek
print(df.head())
print(df.info())

# 7. EDA: plot class balance
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title('Real vs Fake News')
plt.xlabel('Real or Fake')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 8. Clean up: drop missing & duplicates
df.dropna(subset=['text'], inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)
print(f"After cleaning: {len(df)} articles remain")

# 9. Text preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove non-letters
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
print("✅ 'clean_text' column created")

# 10. Feature extraction with TF‑IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].values

print("TF‑IDF matrix shape:", X.shape)

# 11. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# 12.s Baseline Model Training & Evaluation 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 2. Predict on the test set
y_pred = clf.predict(X_test)

# 3. Evaluation metrics
print("Baseline Logistic Regression\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 4. Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# === Step 13: Hyperparameter Tuning with GridSearchCV ===
from sklearn.model_selection import GridSearchCV

# Define a small grid for C (inverse regularization strength)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

print("\nStarting Grid Search...")
grid.fit(X_train, y_train)

# Best parameters and score
print(f"Best params: {grid.best_params_}")
print(f"Best CV F1-score: {grid.best_score_:.4f}")

# Evaluate the best estimator on the test set
best_clf = grid.best_estimator_
y_pred_best = best_clf.predict(X_test)

print("\nTuned Logistic Regression\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Plot new confusion matrix
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (Tuned Model)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

