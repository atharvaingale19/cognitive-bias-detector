from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import clone
from bias_detector import build_model


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_dataset.csv"


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
    """
    Group highly similar texts so near-duplicates remain
    in the same cross-validation fold.
    """

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

    return np.array([groups[find(i)] for i in range(n)])


def evaluate():
    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["label"]

    print("LEAKAGE-RESISTANT EVALUATION")
    print("=" * 60)

    print(f"Examples: {len(df)}")
    print(f"Classes: {y.nunique()}")

    # --------------------------------------------------
    # Create similarity groups
    # --------------------------------------------------

    print("\nCreating similarity groups...")
    print("Similarity threshold: 0.85")

    groups = create_similarity_groups(
        X.tolist(),
        threshold=0.85
    )

    print(f"Similarity groups: {len(np.unique(groups))}")

    # --------------------------------------------------
    # Stratified group cross-validation
    # --------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = build_model()

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="f1_macro"
    )

    print("\n5-FOLD GROUPED CROSS-VALIDATION")
    print("-" * 60)

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.3f}")

    print(f"\nMean Macro F1: {scores.mean():.3f}")
    print(f"Std Macro F1:  {scores.std():.3f}")

    # --------------------------------------------------
    # Final grouped holdout
    # --------------------------------------------------

    print("\nGROUPED HOLDOUT TEST")
    print("-" * 60)

    splits = list(
        cv.split(X, y, groups=groups)
    )

    train_idx, test_idx = splits[0]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )
    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"\nAccuracy:     {accuracy:.3f}")
    print(f"Macro F1:     {macro_f1:.3f}")
    print(f"Weighted F1:  {weighted_f1:.3f}")

    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate()