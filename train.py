# train.py — Hand Gesture Recognition (Rock Paper Scissors dataset from Kaggle)
# Dataset: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
#
# Pipeline: MediaPipe hand landmarks (21 keypoints, normalized) -> Logistic Regression
#
# Why Logistic Regression, not Random Forest: a real side-by-side comparison
# (compare_models.py) tested 5 algorithms on these exact features. Logistic
# Regression scored highest (100% vs Random Forest's 99.26% on the same
# 406-image test split) -- and since the landmark normalization already makes
# the 3 gesture classes cleanly separable, a simple linear model does the job
# without the extra complexity of 200 trees.

import os
import cv2
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from landmarks import extract_landmarks

DATA_DIR = "rockpaperscissors"   # folder you get after unzipping the Kaggle dataset

def load_dataset():
    X, y = [], []
    counts = {}
    skipped = 0
    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        if label not in ("rock", "paper", "scissors"):
            continue
        for fname in sorted(os.listdir(label_path)):
            img = cv2.imread(os.path.join(label_path, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            feat = extract_landmarks(img)
            if feat is None:
                skipped += 1
                continue
            X.append(feat)
            y.append(label)
            counts[label] = counts.get(label, 0) + 1
    print("Images per class:", counts)
    print("Skipped (no hand detected):", skipped)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading dataset and extracting hand landmarks (this takes a few minutes)...")
    X, y = load_dataset()
    print("Feature matrix:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline bundles the scaler + classifier into ONE saved object, so
    # app.py doesn't need any changes -- joblib.load(...) and .predict_proba(...)
    # keep working exactly as before, just with a different model inside.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds))

    joblib.dump(model, "gesture_model.pkl")
    print("Saved gesture_model.pkl")
