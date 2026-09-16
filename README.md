# Cognitive Bias Analyzer

An interpretable NLP system for identifying possible cognitive-bias patterns in short text statements.

**Live Demo:** https://cognitive-bias-detector.streamlit.app/

> **Disclaimer:** This project provides machine-learning-based linguistic predictions. It is not a psychological, medical, or clinical assessment.

---

## Overview

The **Cognitive Bias Analyzer** is an NLP classification system that analyzes short text statements and predicts whether they contain patterns associated with specific cognitive biases.

The system recognizes five categories:

- **Catastrophizing**
- **Mind Reading**
- **Overgeneralization**
- **Personalization**
- **None**

The application also provides interpretable model information, including the predicted category, relative model confidence, class-score distribution, and important features contributing to the prediction.

---

## Key Features

- 🧠 Five-class cognitive-bias classification
- 🔤 Word-level and character-level TF-IDF features
- ⚙️ Linear Support Vector Machine (LinearSVC)
- 🔎 Feature-level interpretability
- 📊 Prediction-score distribution
- 🌐 Interactive Streamlit web application
- 🧪 Similarity-grouped evaluation to reduce near-duplicate leakage
- 📁 Reproducible training and evaluation pipeline

---

## Methodology

The final classifier uses a **hybrid TF-IDF representation** combined with a linear Support Vector Machine.

### 1. Word-Level TF-IDF

The model extracts:

- Unigrams
- Bigrams

These features capture individual words as well as short word sequences.

### 2. Character-Level TF-IDF

Character n-grams are also extracted to capture:

- Sub-word patterns
- Word fragments
- Morphological patterns
- Variations in wording

### 3. Feature Combination

The word-level and character-level representations are combined using scikit-learn's `FeatureUnion`.

### 4. Classification

The combined feature representation is passed to a **LinearSVC** classifier with class balancing.

The final architecture was selected through comparative evaluation of multiple NLP model configurations.

---

## Dataset

The dataset contains text examples belonging to the five target categories.

The preprocessing pipeline performs:

- Missing-value removal
- Label normalization
- Duplicate removal
- Near-duplicate analysis
- Dataset consistency checks

After cleaning and deduplication:

- **1,111 usable examples**
- **5 target classes**

The dataset was also audited for highly similar examples because near-duplicate text across training and evaluation sets can lead to overly optimistic performance estimates.

---

## Evaluation

A standard random train/test split can be misleading for text classification when highly similar examples appear in both sets.

To address this, the project uses **similarity-grouped evaluation**, which attempts to keep highly similar examples within the same evaluation group.

### Final Grouped Holdout Results

| Metric | Score |
|---|---:|
| Accuracy | **90.1%** |
| Macro F1 | **89.7%** |
| Weighted F1 | **90.0%** |

### Grouped 5-Fold Cross-Validation

| Metric | Result |
|---|---:|
| Mean Macro F1 | **83.4%** |
| Standard Deviation | **4.9%** |

The grouped evaluation provides a more conservative estimate of generalization than a simple random split.

---

## Interpretability

The application is designed not only to produce a prediction, but also to provide insight into the model's decision.

For each input, the application displays:

- Predicted bias category
- Relative model confidence
- Score distribution across all classes
- Important active features contributing to the prediction
- Description of the predicted category

The feature contributions are derived from the TF-IDF representation and LinearSVC coefficients.

> The displayed confidence is a normalized model score and should not be interpreted as a calibrated probability.

---

## Example

**Input:**

> "One mistake means my entire future is ruined."

**Possible prediction:**

> Catastrophizing

The application then displays the model's relative confidence, class-score distribution, and influential features.

---

## Project Structure

```text
cognitive-bias-detector/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed/
│       └── clean_dataset.csv
│
├── models/
│   └── bias_model.joblib
│
├── results/
│   ├── confusion_matrix.png
│   ├── error_analysis.csv
│   ├── metrics.json
│   └── model_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── bias_detector.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── robust_evaluate.py
│   ├── model_selection.py
│   ├── audit_dataset.py
│   └── error_analysis.py
│

## Technologies

- **Python**
- **Streamlit**
- **scikit-learn**
- **NumPy**
- **Pandas**
- **Joblib**

---

## Running Locally

Clone the repository and install the required dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    streamlit run app.py

The application will then be available locally through Streamlit.

---

## Limitations

This system identifies linguistic patterns associated with the categories present in its training data. It does **not** determine whether an individual actually has a cognitive distortion or psychological condition.

Performance may be affected by:

- Short or ambiguous statements
- Unusual language or phrasing
- Text outside the training distribution
- Overlap between different cognitive-bias categories
- Dataset size and composition
- Limitations of TF-IDF-based representations

The model's confidence values are normalized decision scores rather than calibrated probabilities.

---

## Future Improvements

Potential extensions include:

- Expanding the dataset with larger and more diverse samples
- Human-annotated training data
- Transformer-based NLP models
- Probability calibration
- Multilingual support
- Improved handling of ambiguous statements
- More advanced explainability techniques
- External validation on an independent dataset

---

## Disclaimer

This project is intended for **educational and research purposes**.

Its predictions should not be used for psychological diagnosis, medical decision-making, or clinical assessment.

---

## Author

**Atharva Ingale**
