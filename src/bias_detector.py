from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

BIAS_DESCRIPTIONS = {
    "catastrophizing": (
        "Expecting the worst possible outcome and imagining "
        "a situation as far worse than it actually is."
    ),

    "mind_reading": (
        "Assuming you know what other people are thinking, "
        "usually without sufficient evidence."
    ),

    "none": (
        "No clear cognitive distortion was detected."
    ),

    "overgeneralization": (
        "Drawing broad conclusions from a single event or "
        "a limited number of experiences."
    ),

    "personalization": (
        "Taking excessive personal responsibility for events "
        "that are outside your control."
    ),
}

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "bias_model.joblib"


def build_model():
    """
    Build the final NLP classification pipeline.

    Architecture selected through model comparison:

    1. Word-level TF-IDF features
    2. Character-level TF-IDF features
    3. Linear Support Vector Machine classifier
    """

    word_features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=15000
    )

    char_features = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=20000
    )

    features = FeatureUnion([
        ("word_features", word_features),
        ("char_features", char_features)
    ])

    classifier = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    )

    return Pipeline([
        ("features", features),
        ("classifier", classifier)
    ])


def load_model():
    """
    Load the trained model from disk.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def get_important_features(model, text, top_n=5):
    """
    Extract the most influential active features
    for the predicted class.
    """

    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    # Transform only the input text.
    # The result remains sparse.
    transformed = features.transform([text])

    feature_names = features.get_feature_names_out()

    prediction = model.predict([text])[0]

    class_index = list(
        classifier.classes_
    ).index(prediction)

    coefficients = classifier.coef_[class_index]

    # Only inspect features actually present
    # in the input text.
    feature_indices = transformed.indices
    tfidf_values = transformed.data

    contributions = (
        tfidf_values *
        coefficients[feature_indices]
    )

    sorted_indices = np.argsort(
        contributions
    )[::-1]

    important_features = []

    for i in sorted_indices:

        contribution = contributions[i]

        # Only positive contributions support
        # the predicted class.
        if contribution <= 0:
            continue

        feature_index = feature_indices[i]

        important_features.append(
            {
                "feature": str(
                    feature_names[feature_index]
                ),
                "importance": float(
                    contribution
                )
            }
        )

        if len(important_features) >= top_n:
            break

    return important_features


def analyze_text(text):
    """
    Analyze text and return the predicted
    cognitive distortion with confidence,
    class scores, and important features.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Text must be a non-empty string."
        )

    model = load_model()

    prediction = model.predict([text])[0]

    # Extract features contributing to
    # the predicted class.
    important_features = get_important_features(
        model,
        text,
        top_n=5
    )

    # LinearSVC does not provide predict_proba().
    # decision_function provides raw classification scores.
    scores = model.decision_function([text])[0]

    classes = model.classes_

    # Convert raw SVM scores into normalized
    # relative confidence scores.
    #
    # These are NOT calibrated probabilities.
    exp_scores = np.exp(
        scores - np.max(scores)
    )

    normalized_scores = (
        exp_scores / exp_scores.sum()
    )

    confidence = float(
        normalized_scores.max()
    )

    probability_dict = {
        label: float(score)
        for label, score in zip(
            classes,
            normalized_scores
        )
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
        "important_features": important_features
    }


def predict(text):
    """
    Backward-compatible prediction function.
    """

    return analyze_text(text)


if __name__ == "__main__":

    example = (
        "One mistake means my entire future is ruined."
    )

    result = analyze_text(example)

    print("\nANALYSIS RESULT")
    print("-" * 50)

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.3f}"
    )

    print("\nNormalized class scores:")

    for label, score in result[
        "probabilities"
    ].items():

        print(
            f"{label}: {score:.3f}"
        )

    print("\nImportant features:")

    for feature in result[
        "important_features"
    ]:

        print(
            f"{feature['feature']}: "
            f"{feature['importance']:.4f}"
        )