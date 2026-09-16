from pathlib import Path
import warnings

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")


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

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "model_comparison.csv"


# ============================================================
# CREATE SIMILARITY GROUPS
# ============================================================

def create_similarity_groups(texts, threshold=0.85):

    print("Creating similarity groups...")
    print(f"Similarity threshold: {threshold}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(matrix)

    n = len(texts)

    # Union-Find structure
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            parent[root_y] = root_x

    # Connect highly similar examples
    for i in range(n):

        for j in range(i + 1, n):

            if similarity_matrix[i, j] >= threshold:
                union(i, j)

    # Convert roots into group IDs
    groups = [find(i) for i in range(n)]

    # Compress IDs
    group_mapping = {}
    compressed_groups = []

    for group in groups:

        root = find(group)

        if root not in group_mapping:
            group_mapping[root] = len(group_mapping)

        compressed_groups.append(group_mapping[root])

    print(f"Similarity groups created: {len(set(compressed_groups))}")

    return compressed_groups


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    models = {

        "Word TF-IDF + Logistic Regression":

            Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95
                    )
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),


        "Word TF-IDF + Linear SVM":

            Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95
                    )
                ),

                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),


        "Character TF-IDF + Logistic Regression":

            Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        max_features=10000
                    )
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),


        "Word + Character TF-IDF + Linear SVM":

            Pipeline([
                (
                    "features",

                    FeatureUnion([

                        (
                            "word_tfidf",

                            TfidfVectorizer(
                                lowercase=True,
                                stop_words="english",
                                ngram_range=(1, 2),
                                min_df=1,
                                max_df=0.95
                            )
                        ),

                        (
                            "char_tfidf",

                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                max_features=10000
                            )
                        )

                    ])
                ),

                (
                    "classifier",

                    LinearSVC(
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ])
    }

    return models


# ============================================================
# MAIN MODEL COMPARISON
# ============================================================

def compare_models():

    print("\nMODEL SELECTION PIPELINE")
    print("=" * 65)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "label"]).copy()

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    print(f"\nExamples: {len(df)}")
    print(f"Classes: {y.nunique()}")

    print("\nClass distribution:")

    print(y.value_counts())


    # --------------------------------------------------------
    # Create leakage-resistant similarity groups
    # --------------------------------------------------------

    groups = create_similarity_groups(
        X.tolist(),
        threshold=0.85
    )


    # --------------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scorer = make_scorer(
        f1_score,
        average="macro"
    )


    # --------------------------------------------------------
    # Get models
    # --------------------------------------------------------

    models = get_models()

    results = []


    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 65)
        print(model_name)
        print("=" * 65)

        scores = cross_val_score(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scorer,
            n_jobs=-1
        )

        print("\nFold scores:")

        for i, score in enumerate(scores, start=1):

            print(
                f"Fold {i}: {score:.3f}"
            )

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"\nMean Macro F1: {mean_score:.3f}")
        print(f"Std Macro F1:  {std_score:.3f}")


        results.append({

            "model": model_name,

            "fold_1": round(scores[0], 4),
            "fold_2": round(scores[1], 4),
            "fold_3": round(scores[2], 4),
            "fold_4": round(scores[3], 4),
            "fold_5": round(scores[4], 4),

            "mean_macro_f1": round(mean_score, 4),

            "std_macro_f1": round(std_score, 4)

        })


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="mean_macro_f1",
        ascending=False
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Print ranking
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL MODEL RANKING")
    print("=" * 65)

    print(
        results_df[
            [
                "model",
                "mean_macro_f1",
                "std_macro_f1"
            ]
        ].to_string(index=False)
    )


    best_model = results_df.iloc[0]

    print("\n" + "=" * 65)
    print("BEST MODEL")
    print("=" * 65)

    print(f"Model: {best_model['model']}")

    print(
        f"Macro F1: "
        f"{best_model['mean_macro_f1']:.3f} "
        f"± {best_model['std_macro_f1']:.3f}"
    )

    print(f"\nResults saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":

    compare_models()