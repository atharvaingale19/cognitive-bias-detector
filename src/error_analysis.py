from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "clean_dataset.csv"
)

RESULTS_DIR = BASE_DIR / "results"


def build_model():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])


def create_similarity_groups(texts, threshold=0.85):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)

    similarity_matrix = (X @ X.T).toarray()

    n = len(texts)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                union(i, j)

    groups = {}

    for i in range(n):
        root = find(i)

        if root not in groups:
            groups[root] = len(groups)

    return np.array([
        groups[find(i)]
        for i in range(n)
    ])


def analyze_errors():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["label"]

    print("ERROR ANALYSIS")
    print("=" * 60)

    print("Creating similarity groups...")
    groups = create_similarity_groups(
        X.tolist(),
        threshold=0.85
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Use the same first grouped split as robust_evaluate.py
    train_idx, test_idx = next(
        cv.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    confidence = probabilities.max(axis=1)

    errors = []

    for i, (text, actual, predicted, conf) in enumerate(
        zip(
            X_test,
            y_test,
            predictions,
            confidence
        )
    ):
        if actual != predicted:
            errors.append({
                "text": text,
                "actual": actual,
                "predicted": predicted,
                "confidence": conf
            })

    errors_df = pd.DataFrame(errors)

    print(f"\nTest examples: {len(X_test)}")
    print(f"Incorrect predictions: {len(errors_df)}")

    if len(errors_df) > 0:

        print("\nERRORS BY ACTUAL CLASS")
        print("-" * 60)

        print(
            errors_df["actual"]
            .value_counts()
        )

        print("\nERRORS BY PREDICTED CLASS")
        print("-" * 60)

        print(
            errors_df["predicted"]
            .value_counts()
        )

        print("\nDETAILED ERRORS")
        print("-" * 60)

        for _, row in errors_df.iterrows():
            print(f"\nText: {row['text']}")
            print(f"Actual:    {row['actual']}")
            print(f"Predicted: {row['predicted']}")
            print(f"Confidence: {row['confidence']:.3f}")

    # Save errors for later analysis
    output_path = RESULTS_DIR / "error_analysis.csv"

    errors_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nError analysis saved to: {output_path}"
    )


if __name__ == "__main__":
    analyze_errors()