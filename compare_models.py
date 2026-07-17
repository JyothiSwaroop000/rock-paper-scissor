# compare_models.py — side-by-side comparison of 5 classifiers on the same
# MediaPipe landmark features used by train.py. Referenced in train.py's
# comment explaining why Logistic Regression was chosen for deployment.

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from landmarks import extract_landmarks

DATA_DIR = "rockpaperscissors"


def load_dataset():
    X, y = [], []
    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path) or label not in ("rock", "paper", "scissors"):
            continue
        for fname in sorted(os.listdir(label_path)):
            img = cv2.imread(os.path.join(label_path, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            feat = extract_landmarks(img)
            if feat is None:
                continue
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Loading dataset and extracting hand landmarks...")
    X, y = load_dataset()
    print("Feature matrix:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    for name, clf in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results.append({"model": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        print(f"{name:22} accuracy={acc:.4f}  macro_f1={f1:.4f}")

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    df.to_csv("model_comparison.csv", index=False)
    print("\nSaved model_comparison.csv")
    print(df.to_string(index=False))
