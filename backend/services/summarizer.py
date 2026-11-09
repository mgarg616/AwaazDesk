"""
AwaazDesk - Extractive Summarization Engine
Generates a concise, high-salience executive summary from raw grievance statements.
Explicitly distinguished as [Extractive Rule-Based Summarizer].
"""
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("awaazdesk.summarizer")

def generate_summary(text: str) -> Dict[str, Any]:
    """
    Extracts salient problem sentences and reformulates a clean executive summary.
    """
    cleaned = text.strip()
    # Split by standard sentence delimiters (English periods/newlines, Devanagari danda ।)
    sentences = re.split(r"[।\.\n\?!]+", cleaned)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 8]

    if not sentences:
        return {
            "summary": cleaned[:150] + ("..." if len(cleaned) > 150 else ""),
            "attribution": "[Extractive Rule-Based Summarizer (Truncation Fallback)]"
        }

    # Score sentences based on problem markers and length
    scored_sentences = []
    salience_markers = [
        "par", "me", "se", "hai", "nahi", "problem", "delay", "leak", "power", "water",
        "road", "drainage", "transformer", "bill", "hospital", "pension", "atm", "fail",
        "कचरा", "सड़क", "पानी", "बिजली", "अस्पताल", "खाते", "पैसे", "शिकायत", "तुरंत"
    ]

    for s in sentences:
        score = 0
        s_lower = s.lower()
        for m in salience_markers:
            if m in s_lower:
                score += 1
        # Moderate sentence length preferred (10 to 30 words)
        word_count = len(s.split())
        if 8 <= word_count <= 35:
            score += 2
        scored_sentences.append((score, s))

    # Sort descending by score
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # Take top 1 or 2 most descriptive sentences
    top_picks = [item[1] for item in scored_sentences[:2]]
    summary_text = ". ".join(top_picks)
    if not summary_text.endswith("."):
        summary_text += "."

    return {
        "summary": summary_text,
        "attribution": f"[Extractive Rule-Based Summarizer (Salience Ranking over {len(sentences)} sentences)]"
    }
