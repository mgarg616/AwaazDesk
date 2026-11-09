# AwaazDesk: Voice-First Civic Grievance Intelligence System
**The Great Indian AI Internship Challenge (Gnani.ai)**  
**Author:** Muskan | **Target Categories:** 02 Mixed Vibes Award, 08 Small Town Big Build, 09 Demo Day Drop  
**GitHub Repository:** [https://github.com/mgarg616/AwaazDesk](https://github.com/mgarg616/AwaazDesk)  

---

### 1. Problem Statement
Citizens across Tier-2, Tier-3 cities, and rural Bharat face urgent civic breakdowns daily—ranging from sparking electrical transformers and pipeline contamination to road craters and sanitation failures. However, conventional public grievance redressal mechanisms remain text-heavy, English-dominated, and bureaucratic. Citizens from vernacular and non-English backgrounds encounter steep friction with rigid online forms, resulting in delayed emergency hazard reporting, unresolved public complaints, and severe sorting backlogs for municipal officers.

---

### 2. The Solution: AwaazDesk
**AwaazDesk** is an accessible, voice-first multilingual civic dispute resolution platform built for the linguistic reality of Bharat. Citizens simply speak or type their grievance naturally in **हिन्दी (Hindi)**, **English**, or code-mixed **Hinglish**. The platform transcribes the statement in real-time, categorizes the civic domain with high-accuracy machine learning, scores public urgency, and provides triage officers with actionable root-cause guidance.

---

### 3. Core Architecture & Gnani.ai Indic AI Stack Integration
AwaazDesk integrates the sovereign Indic AI capabilities developed by **Gnani.ai**:

1. **Gnani PRISMA STT (Speech-to-Text - v3.0):**  
   Processes raw citizen audio via the official `api.vachana.ai` endpoint with automated dialect detection (`indic-auto`). Accurately transcribes phonetic Indian speech and code-mixed vernacular expressions (e.g., *"Transformer mein spark ho raha hai, short circuit ka risk hai"*).
2. **Gnani TIMBRE TTS (Text-to-Speech - v3.0):**  
   Synthesizes clear Indic neural speech for instant grievance confirmation and accessible status-tracking audio readouts, ensuring digital inclusion for low-literacy citizens.
3. **Gnani EVON v3.3 Indic LLM (Artha Sovereign Stack):**  
   Integrated with Gnani's sovereign Indic model endpoint (`api.vachana.ai/v1/chat/completions`) using `X-API-Key-ID` authentication. It evaluates linguistic nuance, identifies root causes, and recommends next-best actions for field officers, backed by a transparent Indic fallback mode to guarantee zero-downtime evaluation.

---

### 4. Machine Learning & Responsible AI Design
- **Empirically Evaluated ML Pipeline:** Scikit-Learn `TfidfVectorizer` + `LogisticRegression` delivering **90.2% Held-Out Test Accuracy** and **92.3% 5-Fold Cross-Validation Accuracy** (±3.09%, peak fold **97.8%**) across 8 civic classes with a 0.903 macro F1-score (verified in `backend/models/evaluation_report.json`).
- **Strict ML vs Rule Separation:** Physical hazard detection is governed by a deterministic multi-factor rule engine (Critical / High / Medium / Low), preventing black-box hallucinations during public safety emergencies.
- **Sentiment & Extractive Summarization:** Quantifies citizen frustration polarity and generates executive briefs without external cloud dependencies.
- **Officer Operations Console:** Role-based access control (RBAC) with JWT auth, ticket lifecycle tracking (`Open` → `In Progress` → `Resolved`), and audited SQLite database logging.

---

### 5. Measured Impact & Engineering Rigor
- **Zero Digital Literacy Barrier:** Voice-enabled complaint submission in under 30 seconds.
- **Automated Instant Triage:** Replaces manual ticket reading with real-time classification, hazard scoring, and instant root-cause domain routing upon submission.
- **Production-Ready & Fully Tested:** Validated by **106 genuinely executed and passing pytest test suites** across API, Auth, ML, and Speech modules (`pytest backend/tests/`).

---
*Built with ❤️ for citizen empowerment using Gnani.ai’s sovereign Indic voice intelligence.*
