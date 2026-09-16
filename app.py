import streamlit as st

from src.bias_detector import (
    analyze_text,
    BIAS_DESCRIPTIONS,
)


# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="Cognitive Bias Analyzer",
    page_icon="🧠",
    layout="centered",
)


# -------------------------
# Header
# -------------------------

st.title("Cognitive Bias Analyzer")

st.write(
    "An interpretable NLP system for identifying "
    "possible cognitive-bias patterns in text."
)

st.caption(
    "This tool provides model-based predictions, "
    "not psychological or clinical assessments."
)


# -------------------------
# Input
# -------------------------

text = st.text_area(
    "Enter a statement",
    placeholder=(
        "Example: He didn't reply because "
        "he must be angry with me."
    ),
    height=150,
)


# -------------------------
# Analysis
# -------------------------

if st.button(
    "Analyze",
    type="primary",
    use_container_width=True,
):

    if not text.strip():

        st.warning(
            "Please enter a statement."
        )

    else:

        result = analyze_text(text)

        if result is None:

            st.error(
                "Unable to analyze the supplied text."
            )

        else:

            prediction = result["prediction"]

            confidence = result["confidence"]

            probabilities = result[
                "probabilities"
            ]

            important_features = result[
                "important_features"
            ]

            # -------------------------
            # Main result
            # -------------------------

            st.subheader("Analysis")

            display_prediction = (
                prediction
                .replace("_", " ")
                .title()
            )

            st.write(
                f"### Possible bias: {display_prediction}"
            )

            st.metric(
                "Model probability",
                f"{confidence:.1%}",
            )

            st.info(
                BIAS_DESCRIPTIONS[prediction]
            )

            # -------------------------
            # Probability distribution
            # -------------------------

            st.subheader(
                "Prediction distribution"
            )

            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for label, probability in (
                sorted_probabilities
            ):

                display_label = (
                    label
                    .replace("_", " ")
                    .title()
                )

                st.write(
                    f"**{display_label}**"
                )

                st.progress(
                    float(probability)
                )

                st.caption(
                    f"{probability:.1%}"
                )

            # -------------------------
            # Evidence
            # -------------------------

            if important_features:

                st.subheader(
                    "Model evidence"
                )

                st.write(
                    "These features contributed "
                    "positively to the predicted class:"
                )

                for feature in important_features:

                    st.code(
                        feature,
                        language=None
                    )

            # -------------------------
            # Disclaimer
            # -------------------------

            st.divider()

            st.caption(
                "Important: this system is an NLP "
                "classification experiment. Predictions "
                "can be incorrect, especially for text "
                "outside the training distribution."
            )