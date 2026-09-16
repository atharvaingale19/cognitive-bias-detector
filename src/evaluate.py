from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.base import clone


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "clean_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "bias_model.joblib"
)

RESULTS_DIR = BASE_DIR / "results"

METRICS_PATH = RESULTS_DIR / "metrics.json"

CONFUSION_PATH = (
    RESULTS_DIR
    / "confusion_matrix.png"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df["text"].astype(str)
    y = df["label"]

    return df, X, y


# ============================================================
# CREATE SIMILARITY GROUPS
#
# IMPORTANT:
# Similar examples belong to the same group.
# This prevents near-duplicates leaking between train and test.
# ============================================================

def create_similarity_groups(texts, threshold=0.85):

    print("\nCreating similarity groups...")
    print(f"Similarity threshold: {threshold}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)

    similarity = (
        matrix @ matrix.T
    ).toarray()

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

            if similarity[i, j] >= threshold:

                union(i, j)


    groups = np.array(
        [find(i) for i in range(n)]
    )

    # Convert group IDs into clean sequential numbers
    _, groups = np.unique(
        groups,
        return_inverse=True
    )

    print(
        f"Similarity groups created: "
        f"{len(np.unique(groups))}"
    )

    return groups


# ============================================================
# EVALUATE
# ============================================================

def evaluate():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df, X, y = load_data()

    print("\nFINAL MODEL EVALUATION")
    print("=" * 60)

    print(f"\nExamples: {len(df)}")
    print(f"Classes: {y.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts())


    # ========================================================
    # CREATE LEAKAGE-RESISTANT GROUPS
    # ========================================================

    groups = create_similarity_groups(
        X,
        threshold=0.85
    )


    # ========================================================
    # LOAD FINAL MODEL
    # ========================================================

    print("\nLoading final trained model...")

    model = joblib.load(MODEL_PATH)

    print(
        f"Model loaded successfully from:\n{MODEL_PATH}"
    )


    # ========================================================
    # GROUPED CROSS VALIDATION
    # ========================================================

    print("\n" + "=" * 60)
    print("5-FOLD GROUPED CROSS-VALIDATION")
    print("=" * 60)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_scores = []

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]


        # Clone creates a fresh version of the model
        fold_model = clone(model)

        fold_model.fit(
            X_train,
            y_train
        )

        predictions = fold_model.predict(
            X_test
        )

        score = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        fold_scores.append(score)

        print(
            f"Fold {fold}: {score:.3f}"
        )


    mean_f1 = float(
        np.mean(fold_scores)
    )

    std_f1 = float(
        np.std(fold_scores)
    )


    print(
        f"\nMean Macro F1: "
        f"{mean_f1:.3f}"
    )

    print(
        f"Std Macro F1:  "
        f"{std_f1:.3f}"
    )


    # ========================================================
    # FINAL GROUPED HOLDOUT
    # ========================================================

    print("\n" + "=" * 60)
    print("GROUPED HOLDOUT EVALUATION")
    print("=" * 60)


    train_index, test_index = next(
        cv.split(X, y, groups)
    )


    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


    evaluation_model = clone(model)

    evaluation_model.fit(
        X_train,
        y_train
    )


    predictions = evaluation_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


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


    print(
        f"\nAccuracy:    {accuracy:.3f}"
    )

    print(
        f"Macro F1:    {macro_f1:.3f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.3f}"
    )


    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = sorted(y.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )


    fig, ax = plt.subplots(
        figsize=(9, 8)
    )


    display.plot(
        ax=ax,
        xticks_rotation=45
    )


    ax.set_title(
        "Final Cognitive Bias Classifier\n"
        "Leakage-Resistant Grouped Evaluation"
    )


    plt.tight_layout()


    plt.savefig(
        CONFUSION_PATH,
        dpi=200
    )


    plt.close()


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "dataset_size": int(len(df)),

        "number_of_classes": int(
            y.nunique()
        ),

        "similarity_threshold": 0.85,

        "similarity_groups": int(
            len(np.unique(groups))
        ),

        "grouped_cross_validation": {

            "folds": 5,

            "scores": [
                float(score)
                for score in fold_scores
            ],

            "mean_macro_f1": mean_f1,

            "std_macro_f1": std_f1,
        },

        "grouped_holdout": {

            "test_size": int(
                len(X_test)
            ),

            "accuracy": float(
                accuracy
            ),

            "macro_f1": float(
                macro_f1
            ),

            "weighted_f1": float(
                weighted_f1
            ),
        },
    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"\nMetrics saved to:\n{METRICS_PATH}"
    )

    print(
        f"\nConfusion matrix saved to:\n{CONFUSION_PATH}"
    )


if __name__ == "__main__":
    evaluate()