"""
AwaazDesk - Grievance Sentiment Analysis Engine
Lexicon-based polarity and frustration scoring for Hindi, English, and Hinglish complaints.
Explicitly distinguished as [Lexical Sentiment Engine].
"""
import re
import logging
from typing import Dict, Any

logger = logging.getLogger("awaazdesk.sentiment_engine")

# Negative Sentiment Indicators
NEGATIVE_TERMS = [
    # English
    "pathetic", "terrible", "worst", "harassment", "corrupt", "disaster", "fail",
    "failed", "refusing", "delay", "damaged", "danger", "fraud", "scam", "shameful",
    "unacceptable", "broken", "dirty", "unhygienic", "suffering", "careless",
    "incompetent", "overcharged", "illegal", "clogged", "overflowing", "useless",
    # Hindi & Hinglish
    "bura", "ganda", "kharab", "pareshan", "pareshani", "musibat", "badtameez",
    "badtameezi", "dhokha", "lut", "loot", "ghatiya", "barbad", "tang", "tadap",
    "dhakka", "bhedbhav", "laaparwahi", "dhokhadhadi", "bhrashtachar", "sharmnak",
    "trasdi", "bura haal", "bina baat", "pareshan kar rahe hain"
]

POSITIVE_CONSTRUCTIVE_TERMS = [
    "thank you", "dhanyawad", "shukriya", "appreciate", "helpful", "resolved",
    "good work", "kripya", "kindly", "request", "prarthana", "anurodh", "please"
]

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Computes sentiment polarity and frustration score using lexical heuristics.
    Grievances range from Mildly Negative (-0.30) to Highly Frustrated (-0.95).
    """
    text_lower = text.lower()

    neg_count = sum(1 for w in NEGATIVE_TERMS if re.search(rf"\b{re.escape(w)}\b", text_lower))
    pos_count = sum(1 for w in POSITIVE_CONSTRUCTIVE_TERMS if re.search(rf"\b{re.escape(w)}\b", text_lower))

    # Base grievance offset (complaints have inherent negative baseline)
    score = -0.40 - (neg_count * 0.15) + (pos_count * 0.08)

    # Exclamation marks and uppercase reflect heightened frustration
    exclamations = text.count("!")
    if exclamations > 1:
        score -= min(exclamations * 0.05, 0.20)

    # Bound score between -1.0 and 1.0
    score = round(max(-1.0, min(1.0, score)), 2)

    if score <= -0.70:
        polarity = "Strongly Negative"
    elif score <= -0.25:
        polarity = "Negative"
    elif score <= 0.15:
        polarity = "Neutral"
    else:
        polarity = "Constructive / Inquisitive"

    return {
        "sentiment_polarity": polarity,
        "sentiment_score": score,
        "attribution": f"[Lexical Sentiment Engine (Score: {score}, Polarity: {polarity})]"
    }
