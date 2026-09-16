from pathlib import Path
import json
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "messy_bias_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLEAN_DATA_PATH = PROCESSED_DIR / "clean_dataset.csv"
REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"


VALID_LABELS = {
    "overgeneralization",
    "mind_reading",
    "catastrophizing",
    "personalization",
    "none",
}


def normalize_text(text):
    """Normalize whitespace without changing the meaning of the text."""

    if pd.isna(text):
        return None

    text = str(text)

    # Replace tabs, newlines and repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_label(label):
    """Convert inconsistent label formatting into canonical labels."""

    if pd.isna(label):
        return None

    label = str(label).strip().lower()

    # Convert hyphens to underscores.
    label = label.replace("-", "_")

    # Remove accidental non-letter characters.
    label = re.sub(r"[^a-z_]", "", label)

    # Handle known naming variation.
    if label == "over_generalization":
        label = "overgeneralization"

    return label


def clean_dataset():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)

    original_rows = len(df)

    # -------------------------
    # Basic validation
    # -------------------------

    required_columns = {"text", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # -------------------------
    # Missing values
    # -------------------------

    missing_text = int(df["text"].isna().sum())
    missing_labels = int(df["label"].isna().sum())

    df["text"] = df["text"].apply(normalize_text)
    df["label"] = df["label"].apply(normalize_label)

    # Remove rows with missing text or label.
    df = df.dropna(subset=["text", "label"])

    rows_after_missing = len(df)

    # -------------------------
    # Invalid labels
    # -------------------------

    invalid_labels = sorted(
        set(df["label"]) - VALID_LABELS
    )

    invalid_label_rows = int(
        (~df["label"].isin(VALID_LABELS)).sum()
    )

    df = df[df["label"].isin(VALID_LABELS)]

    rows_after_label_cleaning = len(df)

    # -------------------------
    # Duplicate detection
    # -------------------------

    duplicate_rows = int(
        df.duplicated(subset=["text"], keep="first").sum()
    )

    df = df.drop_duplicates(
        subset=["text"],
        keep="first"
    )

    final_rows = len(df)

    # -------------------------
    # Class distribution
    # -------------------------

    class_distribution = (
        df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # -------------------------
    # Save cleaned dataset
    # -------------------------

    df.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    # -------------------------
    # Quality report
    # -------------------------

    report = {
        "original_rows": original_rows,
        "missing_text_rows": missing_text,
        "missing_label_rows": missing_labels,
        "rows_after_missing_value_removal": rows_after_missing,
        "invalid_label_rows": invalid_label_rows,
        "invalid_labels_found": invalid_labels,
        "rows_after_label_cleaning": rows_after_label_cleaning,
        "duplicate_text_rows_removed": duplicate_rows,
        "final_unique_rows": final_rows,
        "number_of_classes": len(class_distribution),
        "class_distribution": class_distribution,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("DATA CLEANING COMPLETE")
    print("-" * 40)

    for key, value in report.items():
        print(f"{key}: {value}")

    print(f"\nClean dataset saved to:")
    print(CLEAN_DATA_PATH)

    return df


if __name__ == "__main__":
    clean_dataset()