"""
AwaazDesk - ML Grievance Category Classifier
Uses Scikit-learn TF-IDF Vectorizer + Logistic Regression.
Explicitly distinguished as [ML Model: Scikit-learn TF-IDF + LogisticRegression].
"""
import logging
from typing import Dict, Any, Tuple
import joblib
from backend.config import CATEGORY_MODEL_PATH, VECTORIZER_PATH, GRIEVANCE_CATEGORIES

logger = logging.getLogger("awaazdesk.ml_classifier")

class MLClassifierService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads trained model and vectorizer from disk."""
        if CATEGORY_MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            try:
                self.model = joblib.load(CATEGORY_MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("Loaded ML model and vectorizer successfully.")
            except Exception as e:
                logger.error("Failed to load ML artifacts: %s", e)
                self.model = None
                self.vectorizer = None
        else:
            logger.warning("ML artifacts not found at %s. Please train the model.", CATEGORY_MODEL_PATH)

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict_category(self, text: str) -> Dict[str, Any]:
        """
        Classifies the grievance text and returns predicted category, confidence score,
        and full probability distribution across all grievance categories.
        """
        if not self.is_ready():
            self.load_artifacts()

        if not self.is_ready():
            # Fallback if model hasn't been trained yet
            return {
                "category_ml": "Other Grievances",
                "category_confidence": 0.50,
                "category_top_probabilities": {"Other Grievances": 1.0},
                "attribution": "[ML Model: Scikit-learn TF-IDF + LogisticRegression (Model uninitialized fallback)]"
            }

        try:
            cleaned_text = text.strip()
            # Vectorize text
            X = self.vectorizer.transform([cleaned_text])

            # Probabilities from Logistic Regression
            probabilities = self.model.predict_proba(X)[0]
            classes = self.model.classes_

            # Build class -> probability map
            prob_dict = {classes[i]: round(float(probabilities[i]), 4) for i in range(len(classes))}
            # Sort descending
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

            # Best prediction
            top_category = max(prob_dict, key=prob_dict.get)
            top_confidence = prob_dict[top_category]

            return {
                "category_ml": top_category,
                "category_confidence": top_confidence,
                "category_top_probabilities": sorted_probs,
                "attribution": f"[ML Model: Scikit-learn TF-IDF + LogisticRegression (Confidence: {int(top_confidence * 100)}%)]"
            }
        except Exception as e:
            logger.error("Error during ML category prediction: %s", e)
            return {
                "category_ml": "Other Grievances",
                "category_confidence": 0.50,
                "category_top_probabilities": {"Other Grievances": 1.0},
                "attribution": "[ML Model: Scikit-learn TF-IDF + LogisticRegression (Error Fallback)]"
            }

# Global singleton
ml_classifier = MLClassifierService()
