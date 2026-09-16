from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_dataset.csv"


def audit_dataset():
    df = pd.read_csv(DATA_PATH)

    print("DATASET AUDIT")
    print("=" * 50)

    print(f"Total examples: {len(df)}")
    print(f"Unique texts: {df['text'].nunique()}")
    print(f"Unique labels: {df['label'].nunique()}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    # --------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------

    exact_duplicates = df["text"].duplicated().sum()

    print("\nExact duplicates:")
    print(exact_duplicates)

    # --------------------------------------------------
    # Near-duplicate analysis
    # --------------------------------------------------

    print("\nCalculating near-duplicate similarity...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    X = vectorizer.fit_transform(df["text"])

    similarities = cosine_similarity(X)

    near_duplicate_count = 0
    high_similarity_pairs = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = similarities[i, j]

            if similarity >= 0.85:
                near_duplicate_count += 1

                if len(high_similarity_pairs) < 20:
                    high_similarity_pairs.append(
                        (
                            similarity,
                            df.iloc[i]["text"],
                            df.iloc[j]["text"],
                            df.iloc[i]["label"],
                            df.iloc[j]["label"]
                        )
                    )

    print(f"\nPairs with similarity >= 0.85: {near_duplicate_count}")

    print("\nExamples of highly similar statements:")
    print("-" * 50)

    high_similarity_pairs.sort(reverse=True)

    for similarity, text1, text2, label1, label2 in high_similarity_pairs:
        print(f"\nSimilarity: {similarity:.3f}")
        print(f"Text 1 [{label1}]: {text1}")
        print(f"Text 2 [{label2}]: {text2}")

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    audit_dataset()