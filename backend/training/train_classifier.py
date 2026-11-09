"""
AwaazDesk - ML Classifier Trainer (TF-IDF + Logistic Regression)
Trains the multilingual grievance categorization model for Hindi, English, and Hinglish.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import joblib
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from backend.config import (
    BASE_DIR,
    CATEGORY_MODEL_PATH,
    VECTORIZER_PATH,
    EVALUATION_MODEL_PATH,
    EVALUATION_VECTORIZER_PATH,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("awaazdesk.ml_trainer")

DATASET_PATH = BASE_DIR / "backend" / "training" / "dataset.json"

def load_training_data():
    """Loads dataset samples from json and optionally pulls from SQLite complaints."""
    texts = []
    labels = []

    # 1. Load from JSON
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                text = item.get("text", "").strip()
                cat = item.get("category", "").strip()
                if text and cat in GRIEVANCE_CATEGORIES:
                    texts.append(text)
                    labels.append(cat)
        logger.info("Loaded %d samples from %s", len(texts), DATASET_PATH.name)

    # 2. Check if database has extra complaints
    # CRITICAL ML INTEGRITY:
    # Only human-verified complaint categories are allowed to augment the retraining dataset.
    # Unverified model predictions (category_ml) are strictly excluded to avoid confirmation-bias feedback loops.
    try:
        from backend.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT raw_text, verified_category 
            FROM complaints 
            WHERE raw_text IS NOT NULL 
              AND verified_category IS NOT NULL 
              AND verified_category != ''
        """)
        db_rows = cursor.fetchall()
        conn.close()
        verified_count = 0
        for r in db_rows:
            txt = r["raw_text"].strip()
            cat = r["verified_category"].strip()
            if txt and cat in GRIEVANCE_CATEGORIES and txt not in texts:
                texts.append(txt)
                labels.append(cat)
                verified_count += 1
        logger.info("Augmented retraining corpus with %d human-verified citizen tickets.", verified_count)
    except Exception as e:
        logger.debug("Database augment skipped: %s", e)

    return texts, labels

def train_and_export_model():
    """Trains TF-IDF + LogisticRegression pipeline and exports joblib artifacts."""
    texts, labels = load_training_data()
    if not texts:
        raise ValueError("No training data found to train classifier.")

    logger.info("Initializing Word + Character FeatureUnion Vectorizer...")
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

    X = vectorizer.fit_transform(texts)
    y = labels

    classifier = LogisticRegression(
        C=ML_LOGISTIC_REG_C,
        max_iter=ML_LOGISTIC_REG_MAX_ITER,
        class_weight=ML_LOGISTIC_REG_CLASS_WEIGHT,
        random_state=ML_RANDOM_STATE
    )

    # Stratified cross validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=ML_RANDOM_STATE)
    scores = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")
    logger.info("Stratified 3-Fold CV Accuracy: %.4f (+/- %.4f)", scores.mean(), scores.std())

    # Fit full model for production inference
    classifier.fit(X, y)
    train_preds = classifier.predict(X)
    train_acc = accuracy_score(y, train_preds)
    logger.info("Production Full-Data Refit Training Accuracy: %.4f", train_acc)

    # Ensure output directories exist
    CATEGORY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save ONLY production artifacts (NEVER overwrite evaluation artifacts)
    joblib.dump(classifier, CATEGORY_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Successfully exported production full-data refit model to %s and vectorizer to %s", CATEGORY_MODEL_PATH, VECTORIZER_PATH)

    return {
        "artifact_type": "production_full_data_refit",
        "samples": len(texts),
        "cv_accuracy": float(scores.mean()),
        "train_accuracy": float(train_acc),
        "categories": list(classifier.classes_)
    }

if __name__ == "__main__":
    train_and_export_model()
