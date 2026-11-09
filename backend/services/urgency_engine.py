"""
AwaazDesk - Urgency & Severity Detection Engine
Multi-factor Rule-Based Heuristic Engine with multilingual keyword, duration & contextual weighting.
Explicitly distinguished as [Rule-Based Heuristic Engine].
"""
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("awaazdesk.urgency_engine")

# 1. Critical Urgency Lexicon (Base Score: 8.5 - 10.0)
# Life Safety, Electrical & Fire Hazards, Structural Collapses, Severe Health Emergencies
CRITICAL_TRIGGERS = [
    # Electrical & Fire
    "shock", "electric shock", "spark", "sparks", "fire", "aag", "short circuit",
    "transformer blast", "hanging wire", "loose wire", "khula taar", "khula tar", "tuta taar",
    "बिजली का झटका", "करंट", "बिजली का करंट", "आग", "शॉर्ट सर्किट", "ट्रांसफार्मर ब्लास्ट",
    "खुला तार", "टूटा तार", "तार टूट", "स्पार्क", "चिंगारी",

    # Life Safety & Hazards
    "dangerous", "khatarnak", "life threatening", "mritu", "accident", "durghatna",
    "open manhole", "khula sewer", "khula gutter", "collapse", "girne wala", "stampede",
    "poison", "zahar", "gas leak", "bleeding", "emergency ward", "no oxygen",
    "खतरनाक", "जान का खतरा", "मृत्यु", "दुर्घटना", "हादसा", "खुला मैनहोल", "खुला सीवर",
    "खुला गटर", "गैस लीक", "गैस रिसाव", "ऑक्सीजन खत्म", "जानलेवा", "दीवार गिर", "छत गिर",

    # Immediate & Emergency markers
    "urgent", "tatkal", "turant", "immediate danger", "life danger",
    "तत्काल", "तुरंत", "आपातकालीन", "अति गंभीर", "प्राण संकट"
]

# 2. High Urgency Lexicon (Base Score: 6.5 - 8.4)
# Severe Public Utility Disruptions, Sanitation Outbreaks, Health Disruption
HIGH_TRIGGERS = [
    # Water & Sanitation
    "no drinking water", "no water", "water cut", "water supply stopped", "water crisis",
    "paani nahi", "pani nahi", "paani band", "pani band", "paani nahi aa raha", "pani nahi aa raha",
    "severely contaminated", "ganda paani", "ganda pani", "dusit pani", "peene ka pani nahi",
    "peene ka paani nahi", "diarrhea", "cholera", "sewage in house", "sewage overflow", "sewer overflow",
    "naali overflow", "sewer block",
    "पीने का पानी नहीं", "पानी नहीं आ रहा", "पानी बंद", "गंदा पानी", "दूषित पानी",
    "हैजा", "डायरिया", "सीवर का पानी", "सीवर ओवरफ्लो", "नाली ओवरफ्लो", "सीवर जाम",
    "घर में सीवर", "जल संकट", "पेयजल नहीं", "पानी की किल्लत", "नल में पानी नहीं",

    # Electricity & Public Infrastructure
    "transformer burnt", "transformer phuk gaya", "48 ghante se bijli nahi", "power cut 2 days",
    "bijli nahi", "bijlee nahi", "power cut", "bijli gul", "batti gul", "blackout",
    "बिजली नहीं आ रही", "बिजली गुल", "बत्ती गुल", "ट्रांसफार्मर जल गया", "ट्रांसफार्मर फुंक गया",
    "दो दिन से बिजली नहीं", "बिजली संकट",

    # Roads & Hazards
    "deep pothole", "potholes", "bada khadda", "bada gaddha", "road collapsed",
    "बड़ा गड्ढा", "गहरा गड्ढा", "सड़क धंस गई", "सड़क टूट गई",

    # Health & Medical Disruption
    "hospital no doctor", "doctor absent", "vaccine expired", "no ambulance",
    "अस्पताल में डॉक्टर नहीं", "डॉक्टर नदारद", "एम्बुलेंस नहीं", "दवाइयां नहीं",

    # Fraud & Social Security
    "senior citizen harassed", "fraudulent", "unauthorized withdrawal",
    "dhoka", "cheating", "pension withheld", "pension band",
    "धोखाधड़ी", "धोखा", "पेंशन रोकी", "पेंशन बंद", "पैसे हड़प लिए"
]

# 3. Medium Urgency Lexicon (Base Score: 4.0 - 6.4)
# Financial Inconvenience, Routine Municipal Maintenance & Telecommunications
MEDIUM_TRIGGERS = [
    # Banking & Digital
    "atm fail", "atm cash", "failed transaction", "upi failed", "money debited",
    "pese kat gaye", "paise kat gaye", "meter reading", "overcharged", "extra bill", "heavy bill",
    "एटीएम खराब", "पैसे कट गए", "यूपीआई फेल", "लेनदेन विफल", "मीटर रीडिंग",
    "बिजली का बिल ज्यादा", "ज्यादा बिल", "अतिरिक्त बिल",

    # Municipal & Sanitation Maintenance
    "street light off", "street light kharab", "street light", "pothole", "minor pothole", "kachra nahi uthaya", "kachra", "garbage dump", "badboo",
    "कचरा नहीं उठाया", "कचरे का ढेर", "गंदगी", "बदबू", "कूड़ा नहीं उठाया",
    "स्ट्रीट लाइट बंद", "स्ट्रीट लाइट खराब", "स्ट्रीट लाइट", "सड़क की बत्ती बंद", "सड़क की बत्ती",

    # Telecom & Transport
    "broadband down", "wifi down", "signal drop", "call drop", "train delay",
    "bus frequency", "badtameezi", "rude behavior",
    "इंटरनेट बंद", "ब्रॉडबैंड धीमा", "सिग्नल नहीं", "कॉल ड्रॉप", "ट्रेन लेट", "बस नहीं आई",
    "बदतमीजी", "अभद्र व्यवहार"
]

# Contextual Multipliers & Intensifiers
INTENSIFIER_KEYWORDS = [
    "urgently", "turant", "immediately", "jaldi", "asap", "bahut zyada",
    "terrible", "disaster", "severe", "critical", "bahut pareshan", "behaal",
    "haahakaar", "traahi", "bohot zyada",
    "तुरंत", "तत्काल", "जल्दी", "बहुत ज्यादा", "गंभीर", "बहुत परेशान", "बुरा हाल", "हाहाकार", "त्राहि"
]

# Vulnerable Population & Critical Location Multipliers
VULNERABLE_KEYWORDS = [
    "chhota bachha", "chote bacche", "bachhe", "children", "kids", "infant",
    "senior citizen", "buzoorg", "bujurg", "elderly", "pregnant", "bimar",
    "patient", "hospital", "school", "dispensary",
    "छोटे बच्चे", "बच्चे", "बुजुर्ग", "वृद्ध", "गर्भवती", "बीमार", "मरीज",
    "अस्पताल", "स्कूल", "इलाज"
]

# Essential Public Utility Keywords (used to determine if prolonged outage applies)
ESSENTIAL_SERVICES = [
    "paani", "pani", "water", "पानी", "जल", "नल", "पेयजल",
    "bijli", "electricity", "power", "बिजली", "करंट", "बत्ती", "light",
    "sewage", "sewer", "drainage", "naali", "सीवर", "नाली", "गटर",
    "hospital", "doctor", "ambulance", "medicine", "अस्पताल", "दवा", "इलाज"
]

# Number mappings for duration parsing
DURATION_NUMBERS = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "12": 12, "14": 14, "15": 15, "20": 20, "25": 25, "30": 30,
    "१": 1, "२": 2, "३": 3, "४": 4, "५": 5, "६": 6, "७": 7, "८": 8, "९": 9, "१०": 10,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "पंद्रह": 15, "बीस": 20, "तीस": 30,
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "pandrah": 15, "bees": 20,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20
}


def _matches_keyword(text_lower: str, keyword: str) -> bool:
    """Robust substring/word-boundary matcher across Latin and Devanagari."""
    kw = keyword.strip().lower()
    if not kw:
        return False
    if kw.isascii() and kw.isalnum():
        return bool(re.search(rf"\b{re.escape(kw)}\b", text_lower))
    elif kw.isascii():
        return bool(re.search(rf"(?:^|\b){re.escape(kw)}(?:\b|$)", text_lower))
    else:
        return kw in text_lower


def extract_disruption_duration_in_days(text: str) -> Optional[float]:
    """
    Extracts the duration of a reported public disruption in days.
    Understands Hindi (Devanagari), Hinglish, and English numerals and time units.
    """
    text_lower = text.lower()

    # 1. Check days: e.g. "10 din", "3 days", "१० दिन", "दस दिन", "तीन दिन"
    m_days = re.search(r"(\d+)\s*(?:दिनों|दिन|dino|din|days?)\b", text_lower)
    if m_days:
        return float(m_days.group(1))

    for word, val in sorted(DURATION_NUMBERS.items(), key=lambda x: -len(x[0])):
        p = rf"(?:^|[\s\W]){re.escape(word)}\s*(?:दिनों|दिन|dino|din|days?)(?:[\s\W]|$)"
        if re.search(p, text_lower):
            return float(val)

    # 2. Check weeks: e.g. "1 hafte", "2 weeks", "दो हफ्ते", "हफ्ते से"
    m_weeks = re.search(r"(\d+)\s*(?:सप्ताह|हफ़्ते|हफ्ते|hafton|hafte|weeks?)\b", text_lower)
    if m_weeks:
        return float(m_weeks.group(1)) * 7.0

    for word, val in sorted(DURATION_NUMBERS.items(), key=lambda x: -len(x[0])):
        p = rf"(?:^|[\s\W]){re.escape(word)}\s*(?:सप्ताह|हफ़्ते|हफ्ते|hafton|hafte|weeks?)(?:[\s\W]|$)"
        if re.search(p, text_lower):
            return float(val) * 7.0

    if re.search(r"(?:हफ्ते से|हफ़्ते से|एक हफ्ते से|hafto se|hafte se|weeks together)", text_lower):
        return 7.0

    # 3. Check months: e.g. "1 mahina", "महीने से"
    m_months = re.search(r"(\d+)\s*(?:महीने|महीनों|mahine|mahino|months?)\b", text_lower)
    if m_months:
        return float(m_months.group(1)) * 30.0

    if re.search(r"(?:महीने से|mahine se|months together)", text_lower):
        return 30.0

    # 4. Check hours: e.g. "48 ghante", "24 hours"
    m_hours = re.search(r"(\d+)\s*(?:घंटे|ghanto|ghante|hours?)\b", text_lower)
    if m_hours:
        return round(float(m_hours.group(1)) / 24.0, 1)

    return None


def evaluate_urgency(text: str) -> Dict[str, Any]:
    """
    Evaluates grievance text using rule-based lexical heuristics with
    duration scaling, public safety hazard detection, and multilingual awareness.
    Returns urgency_level_rule, urgency_score, matched triggers, and attribution.
    """
    text_lower = text.lower()
    matched_triggers: List[str] = []
    base_score = 3.0  # Default starting baseline for routine inquiries

    # Check Critical Triggers (Life Safety, Hazards, Fire, Shock)
    critical_matches = [w for w in CRITICAL_TRIGGERS if _matches_keyword(text_lower, w)]
    if critical_matches:
        matched_triggers.extend(critical_matches)
        base_score = 8.5 + min(len(critical_matches) * 0.4, 1.5)

    # Check High Triggers (Water cut, sewage, power cut, hospital issue)
    high_matches = [w for w in HIGH_TRIGGERS if _matches_keyword(text_lower, w)]
    if high_matches:
        matched_triggers.extend(high_matches)
        if not critical_matches:
            base_score = 6.8 + min(len(high_matches) * 0.3, 1.4)

    # Check Medium Triggers
    if not critical_matches and not high_matches:
        medium_matches = [w for w in MEDIUM_TRIGGERS if _matches_keyword(text_lower, w)]
        if medium_matches:
            matched_triggers.extend(medium_matches)
            base_score = 4.8 + min(len(medium_matches) * 0.3, 1.4)

    # --- Duration-Based Escalation (Prolonged Civic Disruption) ---
    duration_days = extract_disruption_duration_in_days(text_lower)
    has_essential_service = any(_matches_keyword(text_lower, s) for s in ESSENTIAL_SERVICES) or bool(high_matches)

    if duration_days is not None and has_essential_service:
        int_days = int(duration_days) if duration_days >= 1.0 else duration_days
        if duration_days >= 7.0:
            # 7+ to 10+ days without essential water/power is an acute public emergency -> Critical
            duration_score = 8.6 + min((duration_days - 7.0) * 0.1, 1.2)
            base_score = max(base_score, duration_score)
            matched_triggers.append(f"Prolonged Outage ({int_days} Days)")
        elif duration_days >= 3.0:
            # 3 to 6 days without water/power -> High Severity
            duration_score = 7.2 + min((duration_days - 3.0) * 0.3, 0.9)
            base_score = max(base_score, duration_score)
            matched_triggers.append(f"Multi-day Outage ({int_days} Days)")
        elif duration_days >= 1.0:
            # 1 to 2 days
            base_score = max(base_score, 6.6)
            matched_triggers.append(f"Daily Outage ({int_days} Day)")

    # Check Intensifiers (urgently, immediately, jaldi, bahut pareshan)
    intensifiers_found = [w for w in INTENSIFIER_KEYWORDS if _matches_keyword(text_lower, w)]
    if intensifiers_found:
        base_score += min(len(intensifiers_found) * 0.3, 0.8)
        matched_triggers.extend(intensifiers_found)

    # Check Vulnerable Population (small kids, hospital, senior citizens)
    vulnerable_found = [w for w in VULNERABLE_KEYWORDS if _matches_keyword(text_lower, w)]
    if vulnerable_found:
        base_score += min(len(vulnerable_found) * 0.3, 0.6)
        matched_triggers.extend(vulnerable_found)

    # Bound score between 1.0 and 10.0
    final_score = round(max(1.0, min(10.0, base_score)), 1)

    # Classify into 4 Standard Tiers
    if final_score >= 8.5:
        level = "Critical"
    elif final_score >= 6.5:
        level = "High"
    elif final_score >= 4.0:
        level = "Medium"
    else:
        level = "Low"

    # Remove duplicates from triggers while preserving order
    unique_triggers = list(dict.fromkeys(matched_triggers))

    return {
        "urgency_level_rule": level,
        "urgency_score": final_score,
        "urgency_triggers": unique_triggers,
        "attribution": f"[Rule-Based Heuristic Engine (Severity Score: {final_score}/10, Triggers: {len(unique_triggers)})]"
    }
