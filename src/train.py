from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC


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

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "bias_model.joblib"


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_model():

    print("\nFINAL MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "label"]).copy()

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    print(f"\nTraining examples: {len(df)}")
    print(f"Classes: {y.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts())


    # --------------------------------------------------------
    # Final winning architecture
    # --------------------------------------------------------

    print("\nBuilding final model...")

    model = Pipeline([

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


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining...")

    model.fit(X, y)


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )


    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nModel architecture:")
    print("Word TF-IDF + Character TF-IDF + Linear SVM")

    print(f"\nTraining examples: {len(df)}")

    print(f"\nModel saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":

    train_model()