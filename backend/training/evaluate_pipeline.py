"""
AwaazDesk - Scikit-Learn Multilingual Complaint Classification Pipeline
Performs preprocessing, stratified train/test split, 5-fold cross-validation,
training, evaluation, classification report generation, and confusion matrix calculation.

Policy: Empirically tested metrics only. Sample data and limitations are explicitly labeled.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import json
import logging
import hashlib
from datetime import datetime
import numpy as np
import joblib
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from backend.config import (
    BASE_DIR,
    CATEGORY_MODEL_PATH,
    VECTORIZER_PATH,
    EVALUATION_MODEL_PATH,
    EVALUATION_VECTORIZER_PATH,
    EVALUATION_REPORT_PATH,
    GRIEVANCE_CATEGORIES,
    ML_MAX_FEATURES,
    ML_NGRAM_RANGE,
    ML_SUBLINEAR_TF,
    ML_TOKEN_PATTERN,
    ML_LOGISTIC_REG_C,
    ML_LOGISTIC_REG_MAX_ITER,
    ML_LOGISTIC_REG_CLASS_WEIGHT,
    ML_RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("awaazdesk.ml_pipeline")

DATASET_PATH = BASE_DIR / "backend" / "training" / "dataset.json"
REPORT_PATH = EVALUATION_REPORT_PATH

def preprocess_multilingual_text(text: str) -> str:
    """
    Normalizes grievance text across Hindi (Devanagari), English, and Hinglish.
    - Strips leading/trailing whitespace
    - Lowercases Latin characters
    - Normalizes punctuation while strictly preserving Devanagari script (U+0900 - U+097F)
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    # Normalize excessive whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Lowercase Latin characters while leaving Devanagari intact
    cleaned = cleaned.lower()
    # Remove URL or phone patterns if any
    cleaned = re.sub(r"http\S+|www\S+", "", cleaned)
    cleaned = re.sub(r"\+?\d[\d\s-]{8,}\d", "", cleaned)
    return cleaned.strip()

def load_and_preprocess_dataset():
    """Loads and preprocesses sample grievance dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_texts = []
    cleaned_texts = []
    labels = []
    languages = []

    for item in data:
        t = item.get("text", "").strip()
        c = item.get("category", "").strip()
        lang = item.get("language", "en").strip()
        if t and c in GRIEVANCE_CATEGORIES:
            cleaned = preprocess_multilingual_text(t)
            if cleaned:
                raw_texts.append(t)
                cleaned_texts.append(cleaned)
                labels.append(c)
                languages.append(lang)

    logger.info("Loaded and preprocessed %d total samples across %d categories.", len(cleaned_texts), len(set(labels)))
    return cleaned_texts, labels, languages

def run_pipeline_evaluation():
    """
    Executes the full Scikit-learn Pipeline with:
    1. Stratified 75/25 Train-Test Split
    2. 5-Fold Stratified Cross Validation
    3. Evaluation: Accuracy, Balanced Accuracy, Precision, Recall, F1
    4. Full Classification Report
    5. Confusion Matrix calculation
    6. Artifact and report serialization
    """
    texts, labels, languages = load_and_preprocess_dataset()

    X = np.array(texts)
    y = np.array(labels)
    langs = np.array(languages)
    classes = sorted(list(set(y)))

    # Stratified Train/Test Split (75% Train, 25% Test)
    X_train, X_test, y_train, y_test, l_train, l_test = train_test_split(
        X, y, langs, test_size=0.25, random_state=ML_RANDOM_STATE, stratify=y
    )
    logger.info("Dataset split: %d Training samples, %d Held-out Testing samples.", len(X_train), len(X_test))

    # Construct Scikit-learn Pipeline with Word + Character FeatureUnion
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=ML_NGRAM_RANGE,
            token_pattern=ML_TOKEN_PATTERN,
            sublinear_tf=ML_SUBLINEAR_TF,
            max_features=ML_MAX_FEATURES
        )),
        ("char", TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            sublinear_tf=ML_SUBLINEAR_TF
        ))
    ])
    classifier = LogisticRegression(
        C=ML_LOGISTIC_REG_C,
        max_iter=ML_LOGISTIC_REG_MAX_ITER,
        class_weight=ML_LOGISTIC_REG_CLASS_WEIGHT,
        random_state=ML_RANDOM_STATE
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])

    # 1. Stratified 5-Fold Cross Validation on the training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=ML_RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    logger.info("5-Fold CV Accuracy: %.4f (+/- %.4f) [Folds: %s]",
                cv_scores.mean(), cv_scores.std(), [round(s, 4) for s in cv_scores])

    # 2. Fit pipeline on training set
    pipeline.fit(X_train, y_train)

    # 3. Evaluate on unseen held-out test set
    y_pred = pipeline.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Compute language-specific test performance
    language_breakdown = {}
    for lang_code in ["en", "hi", "hinglish"]:
        mask = (l_test == lang_code)
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            correct_cnt = int(np.sum(y_test[mask] == y_pred[mask]))
            acc = round(float(correct_cnt / n_samples), 4)
            macro_f1_lang = round(float(f1_score(y_test[mask], y_pred[mask], average="macro", zero_division=0)), 4)
            weighted_f1_lang = round(float(f1_score(y_test[mask], y_pred[mask], average="weighted", zero_division=0)), 4)
            language_breakdown[lang_code] = {
                "sample_count": n_samples,
                "correct_predictions": correct_cnt,
                "accuracy": acc,
                "macro_f1": macro_f1_lang,
                "weighted_f1": weighted_f1_lang,
                "disclosure": f"Small sample size limitation (N={n_samples} held-out test samples). High variance expected; Indic scripts exhibit token sparsity with n-gram models."
            }

    # 4. Generate Classification Report (Text & Structured Dict)
    clf_report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

    # 5. Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_list = cm.tolist()

    # 6. Export EVALUATION artifacts strictly trained on training set (never train on held-out test set)
    # Production artifacts (CATEGORY_MODEL_PATH and VECTORIZER_PATH) are kept separate and untouched.
    EVALUATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["clf"], EVALUATION_MODEL_PATH)
    joblib.dump(pipeline.named_steps["tfidf"], EVALUATION_VECTORIZER_PATH)
    logger.info("Saved serialized evaluation model (N=%d) to %s and vectorizer to %s", len(X_train), EVALUATION_MODEL_PATH, EVALUATION_VECTORIZER_PATH)

    # 7. Compile Evaluation Report & Versioning
    dataset_sha256 = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()

    report_data = {
        "metadata": {
            "dataset_version": "1.0.0",
            "dataset_sha256": dataset_sha256,
            "evaluation_timestamp": datetime.now().isoformat(),
            "scikit_learn_version": sklearn.__version__,
            "random_seed": ML_RANDOM_STATE,
            "model_type": "Scikit-Learn TF-IDF + LogisticRegression",
            "model_configuration": {
                "vectorizer": {
                    "ngram_range": list(ML_NGRAM_RANGE),
                    "max_features": ML_MAX_FEATURES,
                    "sublinear_tf": ML_SUBLINEAR_TF,
                    "token_pattern": ML_TOKEN_PATTERN
                },
                "classifier": {
                    "C": ML_LOGISTIC_REG_C,
                    "max_iter": ML_LOGISTIC_REG_MAX_ITER,
                    "class_weight": ML_LOGISTIC_REG_CLASS_WEIGHT,
                    "random_state": ML_RANDOM_STATE
                }
            },
            "total_samples": len(texts),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "classes": classes,
            "class_count": len(classes)
        },
        "empirical_metrics": {
            "cv_5fold_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_5fold_std": round(float(cv_scores.std()), 4),
            "cv_fold_scores": [round(float(s), 4) for s in cv_scores],
            "test_accuracy": round(float(test_acc), 4),
            "test_balanced_accuracy": round(float(test_balanced_acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4)
        },
        "language_breakdown": language_breakdown,
        "classification_report": clf_report_dict,
        "classification_report_text": clf_report_text,
        "confusion_matrix": {
            "matrix": cm_list,
            "labels": classes
        },
        "sample_data_labeling_and_limitations": {
            "dataset_nature": "Curated multilingual grievance sample dataset (Hindi, English, Hinglish).",
            "sample_size_limitation": (
                f"Sample size is limited to {len(texts)} samples (approx 15 per class). "
                "In production municipal deployments with 10,000+ complaints, variance will stabilize."
            ),
            "probability_calibration_status": "Uncalibrated Raw Softmax Scores",
            "calibration_assessment": (
                "Model confidence outputs are raw multinomial logistic softmax scores, NOT calibrated probabilities. "
                "Fitting post-hoc calibration (Platt scaling or Isotonic regression) requires a sizable independent validation split. "
                "With only 120 total samples (90 train / 30 test) across 8 classes, an additional validation split would starve the "
                "training set (<8 samples per class). Fitting a calibrator on the held-out test set would violate evaluation integrity. "
                "Therefore, raw scores are retained and the 0.35 confidence threshold is treated as an experimental triage heuristic."
            ),
            "spelling_variation_challenge": (
                "Informal phonetic Hinglish exhibits heavy orthographic variation "
                "(e.g., 'paani' vs 'pani' vs 'paani nahi aa raha'). Character n-grams and sublinear TF mitigate this, "
                "but extreme dialectal slang may require phonetic normalization."
            ),
            "semantic_overlap_limitation": (
                "Real-world grievances frequently cross domains (e.g. 'broken road with flooded sewage' overlaps "
                "Civic Infrastructure and Water Drainage). The model outputs full probability distributions "
                "to provide transparency into borderline cases."
            ),
            "continuous_learning_recommendation": (
                "Use the '/api/ml/retrain' endpoint to continuously incorporate verified citizen tickets "
                "into the training corpus as new complaints are resolved."
            )
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation report to %s", REPORT_PATH)

    print("\n" + "=" * 70)
    print("SCIKIT-LEARN ML PIPELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Samples: {len(texts)} | Train: {len(X_train)} | Held-out Test: {len(X_test)}")
    print(f"5-Fold CV Mean Accuracy: {report_data['empirical_metrics']['cv_5fold_mean_accuracy']:.4f} (+/- {report_data['empirical_metrics']['cv_5fold_std']:.4f})")
    print(f"Held-out Test Accuracy:  {report_data['empirical_metrics']['test_accuracy']:.4f}")
    print(f"Macro F1-Score:          {report_data['empirical_metrics']['macro_f1']:.4f}")
    print(f"Weighted F1-Score:       {report_data['empirical_metrics']['weighted_f1']:.4f}")
    print("\n--- CLASSIFICATION REPORT ---")
    print(clf_report_text)
    print("--- CONFUSION MATRIX (Labels on Y, Predictions on X) ---")
    print(np.array(cm))
    print("=" * 70)

    return report_data

if __name__ == "__main__":
    run_pipeline_evaluation()
