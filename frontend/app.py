"""
AwaazDesk - AI Voice Grievance Intelligence System
Frontend Dashboard built with Streamlit, Plotly, and clean civic UI components.
Powered by Gnani.ai Voice Technologies.
"""
import sys
import base64
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from frontend.components.api_client import api_client
from frontend.components.charts import (
    render_category_donut,
    render_urgency_bar,
    render_language_donut,
    render_status_pipeline,
    render_sentiment_bar,
    render_confusion_matrix_heatmap
)
from frontend.components.ui_helpers import (
    render_model_attribution_badge,
    render_urgency_badge,
    render_integration_badge,
    render_ticket_card_html,
    render_timeline_html
)
from backend.config import (
    GRIEVANCE_CATEGORIES,
    URGENCY_LEVELS,
    STATUS_OPTIONS,
    SUPPORTED_LANGUAGES
)

# Page Setup
st.set_page_config(
    page_title="AwaazDesk – Civic Voice Grievance Redressal Portal",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Styles
CSS_PATH = PROJECT_ROOT / "frontend" / "styles" / "custom.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Civic Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎙️ AwaazDesk</h1>
    <p>AI-Powered Civic Voice Grievance Redressal System • Integrated with <b>Gnani.ai Voice Models</b> (Prisma STT & Timbre TTS)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Diagnostics
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/microphone.png", width=64)
st.sidebar.title("AwaazDesk")

NAV_SUBMIT = "📝 Submit Grievance"
NAV_TRACK = "🔍 Track Complaint"
NAV_HISTORY = "📋 Grievance History"
NAV_OFFICER = "🏛️ Officer Portal"
NAV_ANALYTICS = "📊 Civic Analytics"
NAV_DIAGNOSTICS = "⚡ Voice Engine Health"

NAV_LIST = [
    NAV_SUBMIT,
    NAV_TRACK,
    NAV_HISTORY,
    NAV_OFFICER,
    NAV_ANALYTICS,
    NAV_DIAGNOSTICS
]

if "nav_choice" not in st.session_state:
    st.session_state["nav_choice"] = NAV_SUBMIT

current_nav_index = NAV_LIST.index(st.session_state["nav_choice"]) if st.session_state["nav_choice"] in NAV_LIST else 0

nav_choice = st.sidebar.radio(
    "Navigation",
    NAV_LIST,
    index=current_nav_index,
    key="sidebar_navigation_radio"
)
st.session_state["nav_choice"] = nav_choice

# Health Indicator in Sidebar
health_data = api_client.get_health()
is_backend_online = health_data.get("status") == "Healthy"
if is_backend_online:
    st.sidebar.success("🟢 Backend & ML Active")
else:
    st.sidebar.error("🔴 Backend Disconnected")
    st.sidebar.info("Start backend via command: `python run_servers.py`")

st.sidebar.markdown("---")
st.sidebar.caption("AwaazDesk • Powered by Gnani.ai Voice Technologies")

# =============================================================================
# VIEW 1: SUBMIT GRIEVANCE
# =============================================================================
if nav_choice == NAV_SUBMIT:
    st.subheader("📝 Lodge a Citizen Grievance")
    st.write("Submit public service complaints or community disputes using **Voice Recording** or **Typed Text**:")

    col_citizen1, col_citizen2 = st.columns(2)
    with col_citizen1:
        citizen_name = st.text_input("Full Name *", placeholder="e.g. Ramesh Chandra")
    with col_citizen2:
        citizen_phone = st.text_input("Mobile Number (Optional)", placeholder="e.g. +91 98765 43210")

    st.markdown("---")

    # Tabs for Voice vs Text
    voice_tab, text_tab = st.tabs([
        "🎙️ Voice Complaint (Microphone Recording & Gnani STT)",
        "⌨️ Typed Grievance (Text)"
    ])

    transcribed_text = ""
    audio_confidence = 1.0
    submission_channel = "text"
    selected_lang = "hinglish"

    # ----------------- VOICE TAB -----------------
    with voice_tab:
        st.markdown("#### 🎙️ Voice Grievance Recording")

        # Language selection specifically for voice recording
        voice_lang = st.radio(
            "Select the language you will speak in:",
            options=["en", "hi", "hinglish"],
            format_func=lambda code: {
                "en": "🇬🇧 English",
                "hi": "🇮🇳 Hindi (हिन्दी)",
                "hinglish": "🗣️ Hinglish (Hindi + English)"
            }[code],
            horizontal=True,
            index=2,
            key="voice_spoken_language_selector"
        )
        selected_lang = voice_lang

        # Direct in-browser microphone recording
        recorded_mic_audio = st.audio_input(
            "Record your complaint voice (tap microphone to start/stop):",
            key="citizen_live_mic"
        )

        audio_source = None
        audio_filename = "live_mic_recording.wav"

        if recorded_mic_audio is not None:
            audio_source = recorded_mic_audio
            submission_channel = "voice"

        # Alternative Audio File Upload
        with st.expander("📁 Or upload a pre-recorded audio file (optional)"):
            uploaded_file = st.file_uploader(
                "Upload Audio File (WAV, MP3, M4A, OGG)",
                type=["wav", "mp3", "m4a", "ogg"],
                key="citizen_file_audio"
            )
            if uploaded_file is not None and audio_source is None:
                audio_source = uploaded_file
                audio_filename = uploaded_file.name
                submission_channel = "voice"
                st.audio(uploaded_file)

        # Transcribe Action Button
        if audio_source is not None:
            lang_label = {"hi": "Hindi", "en": "English", "hinglish": "Hinglish"}.get(voice_lang, "Hinglish")
            if st.button(f"✨ Transcribe Voice with Gnani Prisma STT ({lang_label})", type="primary", key="btn_transcribe_voice"):
                with st.spinner("Transcribing speech via Gnani Prisma STT..."):
                    audio_bytes = audio_source.getvalue() if hasattr(audio_source, 'getvalue') else audio_source.read()
                    stt_result = api_client.transcribe_audio(
                        audio_bytes=audio_bytes,
                        filename=audio_filename,
                        language=voice_lang
                    )

                    if stt_result and not stt_result.get("_client_error"):
                        transcribed_str = (stt_result.get("text") or "").strip()
                        st.session_state["voice_transcription"] = transcribed_str
                        st.session_state["voice_text_input"] = transcribed_str
                        st.session_state["voice_confidence"] = stt_result.get("confidence", 0.95)
                        st.session_state["stt_provider"] = stt_result.get("provider", "Gnani Prisma STT")
                        st.session_state["stt_placeholder"] = stt_result.get("is_placeholder", False)
                        st.session_state["stt_diag"] = stt_result.get("diagnostic_message", "")
                        if transcribed_str:
                            st.success("✅ Voice successfully transcribed! Review your complaint text below and click 'Submit Citizen Grievance' to generate your Ticket ID.")
                            st.rerun()
                        else:
                            st.warning("Speech was inaudible or empty. Please record again clearly.")
                    elif stt_result and stt_result.get("_client_error"):
                        st.error(f"Transcription Error: {stt_result.get('diagnostic_message', 'Unknown error.')}")
                    else:
                        st.error("Failed to reach backend server. Please verify FastAPI is running on port 8000.")

        # Editable Review Area in Voice Tab
        current_voice_text = st.session_state.get("voice_text_input", st.session_state.get("voice_transcription", ""))
        if "stt_provider" in st.session_state:
            badge_html = render_integration_badge(
                st.session_state.get("stt_provider", "Gnani Prisma STT"),
                st.session_state.get("stt_placeholder", True)
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            if st.session_state.get("stt_diag"):
                st.caption(st.session_state["stt_diag"])

        final_voice_text = st.text_area(
            "Complaint Statement (Transcribed text - editable):",
            value=current_voice_text,
            placeholder="Your spoken words will appear here once transcribed. You can edit or add details directly...",
            height=110,
            key="voice_text_input"
        )
        if final_voice_text and final_voice_text.strip():
            transcribed_text = final_voice_text.strip()
            audio_confidence = st.session_state.get("voice_confidence", 0.92)
            submission_channel = "voice"

    # ----------------- TEXT TAB -----------------
    with text_tab:
        st.markdown("#### ⌨️ Type Grievance in English, Hindi, or Hinglish")

        col_tlang, _ = st.columns([1, 1])
        with col_tlang:
            text_lang = st.selectbox(
                "Statement Language",
                options=list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda k: SUPPORTED_LANGUAGES[k]["display"],
                index=2  # Default Hinglish
            )
        if not audio_source:
            selected_lang = text_lang

        st.write("You can select a realistic sample scenario or type your complaint:")

        sample_col1, sample_col2, sample_col3 = st.columns(3)
        if sample_col1.button("📋 Sample 1 (Water Crisis - 10 Days Outage)"):
            st.session_state["typed_complaint_text"] = "हमारे वार्ड 4 में पिछले 10 दिन से पीने का पानी नहीं आ रहा है। छोटे बच्चे और बुजुर्ग बहुत परेशान हैं। कृपया तुरंत पानी का टैंकर भेजें।"
            st.session_state["sample_lang"] = "hi"
            st.rerun()
        if sample_col2.button("📋 Sample 2 (Electrical Hazard - Loose Sparking Wire)"):
            st.session_state["typed_complaint_text"] = "Colony ke main road par open transformer sparks nikal raha hai, dangerous electricity issue hai kisi ko electric shock lag sakta hai, please urgently repair karwaiye."
            st.session_state["sample_lang"] = "hinglish"
            st.rerun()
        if sample_col3.button("📋 Sample 3 (Healthcare Clinic - Doctor Absent)"):
            st.session_state["typed_complaint_text"] = "The primary healthcare clinic in Ward 12 lacks emergency medicines and the doctor has been absent for three days. Urgent inspection required."
            st.session_state["sample_lang"] = "en"
            st.rerun()

        if "sample_lang" in st.session_state and not audio_source:
            selected_lang = st.session_state["sample_lang"]

        current_typed_text = st.session_state.get("typed_complaint_text", "")
        typed_text = st.text_area(
            "Complaint Statement *",
            value=current_typed_text,
            placeholder="Type your complaint statement here (include duration, location, affected public)...",
            height=120,
            key="typed_complaint_text"
        )
        if typed_text and typed_text.strip():
            # If the user typed text or chose a sample, use it if voice text is not active
            if not transcribed_text or audio_source is None:
                transcribed_text = typed_text.strip()
                submission_channel = "text"

    # Effective grievance statement
    final_complaint_text = transcribed_text.strip()

    # ----------------- REAL-TIME INTELLIGENCE PREVIEW -----------------
    if len(final_complaint_text) >= 10:
        st.markdown("---")
        st.markdown("### ⚡ Live AI Grievance Intelligence Preview")

        with st.spinner("Analyzing complaint intelligence..."):
            preview = api_client.preview_intelligence(final_complaint_text, language=selected_lang)

        if preview:
            prev_c1, prev_c2, prev_c3 = st.columns(3)

            with prev_c1:
                st.markdown("**🏛️ Suggested Category:**")
                st.markdown(f"### {preview['category_ml']}")
                conf_pct = int(preview['category_confidence'] * 100)
                st.progress(preview['category_confidence'])
                st.caption(f"{preview['model_attribution']['category']}")

            with prev_c2:
                st.markdown("**⚖️ Urgency & Severity Level:**")
                urgency_badge = render_urgency_badge(preview['urgency_level_rule'], preview['urgency_score'])
                st.markdown(f"<h3>{urgency_badge}</h3>", unsafe_allow_html=True)
                st.caption(f"{preview['model_attribution']['urgency']}")

            with prev_c3:
                st.markdown("**🧠 Citizen Frustration Sentiment:**")
                st.markdown(f"### {preview['sentiment_polarity']}")
                st.caption(f"{preview['model_attribution']['sentiment']}")

            # Extractive Summary Preview
            st.markdown(f"""
            <div style="background:#1e293b; border-left:4px solid #10b981; padding:0.9rem 1.1rem; border-radius:8px; margin-top:0.8rem; border:1px solid #334155;">
                <b style="color:#ffffff;">📌 Executive Brief:</b><br/>
                <span style="color:#f1f5f9; font-size:0.95rem;">{preview['summary']}</span><br/>
                <span style="font-size:0.75rem; color:#94a3b8; margin-top:4px; display:inline-block;">{preview['model_attribution']['summary']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Gnani Evon Sovereign Indic Reasoning & Officer Action Recommendation
            evon_res = api_client.analyze_with_evon(final_complaint_text, language=selected_lang)
            if evon_res:
                evon_badge = "LIVE API" if not evon_res.get("is_placeholder") else "SIMULATED / ACTIVE"
                badge_bg = "#065f46" if not evon_res.get("is_placeholder") else "#1e3a8a"
                badge_col = "#a7f3d0" if not evon_res.get("is_placeholder") else "#bfdbfe"
                st.markdown(f"""
                <div style="background:#0f172a; border-left:4px solid #3b82f6; padding:0.9rem 1.1rem; border-radius:8px; margin-top:0.8rem; border:1px solid #334155;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <b style="color:#ffffff;">🇮🇳 Gnani Evon v3.3 Sovereign Indic Reasoning & Triage:</b>
                        <span style="background:{badge_bg}; color:{badge_col}; font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:700;">{evon_res.get('provider', 'Gnani Evon')} ({evon_badge})</span>
                    </div>
                    <div style="color:#93c5fd; font-size:0.9rem; margin-bottom:0.35rem;"><b>Root Cause Domain:</b> {evon_res.get('root_cause_domain', 'Civic Operations')}</div>
                    <div style="color:#e2e8f0; font-size:0.88rem; margin-bottom:0.35rem;"><b>Indic Nuance Reasoning:</b> {evon_res.get('reasoning', '')}</div>
                    <div style="color:#34d399; font-size:0.88rem;"><b>Actionable Officer Recommendation:</b> {evon_res.get('actionable_recommendation', '')}</div>
                </div>
                """, unsafe_allow_html=True)

    # ----------------- SUBMIT BUTTON -----------------
    st.markdown("---")
    submit_col1, submit_col2 = st.columns([1.5, 2.5])
    with submit_col1:
        submitted = st.button("🚀 Submit Grievance Ticket", type="primary", use_container_width=True)

    if submitted:
        # Check if user provided audio but did not explicitly click the transcribe button first
        if len(final_complaint_text) < 10 and audio_source is not None:
            with st.spinner("Auto-transcribing your recorded voice audio via Gnani Prisma STT..."):
                audio_bytes = audio_source.getvalue() if hasattr(audio_source, 'getvalue') else audio_source.read()
                auto_stt = api_client.transcribe_audio(
                    audio_bytes=audio_bytes,
                    filename=audio_filename,
                    language=voice_lang
                )
                if auto_stt and not auto_stt.get("_client_error") and (auto_stt.get("text") or "").strip():
                    final_complaint_text = auto_stt.get("text", "").strip()
                    st.session_state["voice_transcription"] = final_complaint_text
                    st.session_state["voice_text_input"] = final_complaint_text
                    st.session_state["voice_confidence"] = auto_stt.get("confidence", 0.95)
                    audio_confidence = st.session_state["voice_confidence"]
                    submission_channel = "voice"
                elif auto_stt and auto_stt.get("_client_error"):
                    st.error(f"Voice Transcription Error: {auto_stt.get('diagnostic_message')}")

        if not citizen_name.strip():
            st.error("Please provide citizen's full name.")
        elif len(final_complaint_text) < 10:
            st.error("Please enter a valid complaint of at least 10 characters or record voice audio.")
        else:
            payload = {
                "citizen_name": citizen_name.strip(),
                "citizen_phone": citizen_phone.strip() if citizen_phone else None,
                "language": selected_lang,
                "submission_channel": submission_channel,
                "raw_text": final_complaint_text,
                "transcription_confidence": audio_confidence
            }

            with st.spinner("Submitting grievance and registering ticket..."):
                created = api_client.submit_complaint(payload)

            if created:
                st.balloons()
                new_ticket_id = created['ticket_id']
                st.session_state["last_submitted_ticket"] = new_ticket_id
                st.session_state["track_target_id"] = new_ticket_id
                if "submitted_tickets_history" not in st.session_state:
                    st.session_state["submitted_tickets_history"] = []
                if new_ticket_id not in st.session_state["submitted_tickets_history"]:
                    st.session_state["submitted_tickets_history"].insert(0, new_ticket_id)

                st.success(f"✅ Grievance successfully registered! Unique Ticket ID: **{new_ticket_id}**")

                # Quick Direct Button to Track this Ticket immediately!
                col_btn_track, _ = st.columns([2, 2])
                with col_btn_track:
                    if st.button(f"🔍 Click Here to Track Status of Ticket {new_ticket_id}", type="secondary", key="btn_quick_track"):
                        st.session_state["nav_choice"] = NAV_TRACK
                        st.session_state["track_target_id"] = new_ticket_id
                        st.rerun()

                # Display Ticket Details Card
                st.markdown(render_ticket_card_html(created), unsafe_allow_html=True)

                # Synthesize Audible Confirmation via Gnani Timbre TTS
                st.markdown("#### 🔊 Audio Confirmation (Gnani Timbre TTS)")
                tts_text = f"Your grievance has been successfully lodged. Your ticket number is {new_ticket_id}."
                tts_result = api_client.synthesize_speech(tts_text, language="en", voice_gender="female")

                if tts_result and tts_result.get("audio_base64"):
                    st.markdown(render_integration_badge(
                        tts_result.get("provider", "Gnani Timbre TTS"),
                        tts_result.get("is_placeholder", True)
                    ), unsafe_allow_html=True)
                    audio_raw = base64.b64decode(tts_result["audio_base64"])
                    st.audio(audio_raw, format="audio/wav")
                    if tts_result.get("diagnostic_message"):
                        st.caption(tts_result.get("diagnostic_message", ""))

                # Clear input state for future grievances
                st.session_state.pop("voice_transcription", None)
                st.session_state.pop("voice_text_input", None)
                st.session_state.pop("typed_complaint_text", None)
            else:
                st.error("Failed to submit grievance. Please verify backend server is running.")

# =============================================================================
# VIEW 2: TRACK COMPLAINT
# =============================================================================
elif nav_choice == NAV_TRACK:
    st.subheader("🔍 Track Grievance Status")
    st.write("Enter your unique Ticket ID or speak/upload audio to view resolution progress and dispute audit history:")

    if "voice_track_success_msg" in st.session_state:
        st.success(st.session_state.pop("voice_track_success_msg"))

    # Voice / Audio Ticket Search Option
    with st.expander("🎙️ Or Track using Voice / Audio (Upload audio file or speak your Ticket ID)", expanded=False):
        st.write("Record your voice or upload an audio clip stating your Ticket ID (e.g. *'AWA 2026 1922'* or *'mera ticket 1922 hai'*):")
        track_voice_lang = st.radio(
            "Spoken Language:",
            options=["hinglish", "hi", "en"],
            format_func=lambda c: {"hinglish": "🗣️ Hinglish", "hi": "🇮🇳 Hindi", "en": "🇬🇧 English"}[c],
            horizontal=True,
            key="track_voice_lang_selector"
        )
        track_mic_audio = st.audio_input("Record voice to track ticket:", key="track_mic_audio_input")
        track_uploaded_file = st.file_uploader("Or upload an audio file (WAV, MP3, M4A, OGG):", type=["wav", "mp3", "m4a", "ogg"], key="track_file_audio_upload")

        track_audio_src = track_mic_audio if track_mic_audio is not None else track_uploaded_file
        if track_audio_src is not None:
            if st.button("✨ Transcribe & Locate Ticket (Gnani Prisma STT)", type="primary", key="btn_track_voice_search"):
                with st.spinner("Analyzing speech via Gnani Prisma STT..."):
                    ab = track_audio_src.getvalue() if hasattr(track_audio_src, 'getvalue') else track_audio_src.read()
                    fn = "track_audio.wav" if track_mic_audio is not None else track_uploaded_file.name
                    stt_res = api_client.transcribe_audio(ab, filename=fn, language=track_voice_lang)
                    if stt_res and not stt_res.get("_client_error"):
                        txt = (stt_res.get("text") or "").strip()
                        st.info(f"🎙️ Transcribed Audio: *\"{txt}\"*")
                        import re
                        match = re.search(r"AWA[- ]?2026[- ]?(\d+)", txt, re.IGNORECASE)
                        detected_id = None
                        if match:
                            detected_id = f"AWA-2026-{match.group(1)}"
                        else:
                            dmatch = re.search(r"\b(\d{4})\b", txt)
                            if dmatch:
                                cand = f"AWA-2026-{dmatch.group(1)}"
                                check_c = api_client.get_complaint(cand)
                                if check_c and "complaint" in check_c:
                                    detected_id = cand

                        if not detected_id:
                            recent = api_client.list_complaints(limit=10)
                            for item in recent:
                                tid = item.get("ticket_id", "")
                                if tid and (tid in txt.upper() or (item.get("citizen_name") and item["citizen_name"].lower() in txt.lower())):
                                    detected_id = tid
                                    break

                        if detected_id:
                            st.session_state["track_target_id"] = detected_id
                            st.session_state["txt_track_ticket_input"] = detected_id
                            st.session_state["voice_track_success_msg"] = f"✅ Voice audio processed successfully! Detected Ticket Number: **{detected_id}**"
                            st.rerun()
                        else:
                            fallback_tid = st.session_state.get("last_submitted_ticket")
                            if fallback_tid:
                                st.session_state["track_target_id"] = fallback_tid
                                st.session_state["txt_track_ticket_input"] = fallback_tid
                                st.session_state["voice_track_success_msg"] = f"✅ Audio received! Loaded your latest active Ticket Number: **{fallback_tid}**"
                                st.rerun()
                            else:
                                st.warning("Could not identify a Ticket ID from the audio. Please state your ticket number clearly (e.g. 'AWA-2026-1024').")
                    else:
                        st.error("Audio transcription failed. Please verify the backend connection.")

    # Default ticket resolution (clean, non-hardcoded)
    default_ticket = (
        st.session_state.get("track_target_id")
        or st.session_state.get("last_submitted_ticket")
        or ""
    )

    # Show recent tickets submitted in this session as clickable quick-access pills
    recent_history = st.session_state.get("submitted_tickets_history", [])
    if recent_history:
        st.write("💡 **Recently Submitted in this Session:**")
        cols_hist = st.columns(min(len(recent_history[:5]), 5))
        for idx, tid in enumerate(recent_history[:5]):
            if cols_hist[idx].button(f"📌 {tid}", key=f"chip_track_{tid}"):
                st.session_state["track_target_id"] = tid
                st.session_state["txt_track_ticket_input"] = tid
                st.rerun()

    search_c1, search_c2 = st.columns([3, 1])
    with search_c1:
        query_ticket = st.text_input(
            "Ticket ID",
            value=default_ticket,
            placeholder="e.g. AWA-2026-1024",
            key="txt_track_ticket_input"
        ).strip().upper()
    with search_c2:
        track_btn = st.button("Track Status", type="primary", use_container_width=True)

    if query_ticket:
        with st.spinner("Fetching ticket record..."):
            ticket_data = api_client.get_complaint(query_ticket)

        if ticket_data and "complaint" in ticket_data:
            c = ticket_data["complaint"]
            
            # Prominent success banner with Ticket ID
            st.success(f"✅ Grievance verified and loaded successfully! Ticket Number: **{c['ticket_id']}**")
            st.markdown(render_ticket_card_html(c), unsafe_allow_html=True)

            # Audio Speech Vocalization of Status using Timbre TTS
            st.markdown("##### 🔊 Listen to Status (Gnani Timbre TTS)")
            speech_prompt = f"Ticket {c['ticket_id']} status is {c['status']}. Assigned to {c.get('assigned_department') or 'Municipal Operations'}."
            tts_res = api_client.synthesize_speech(speech_prompt, language=c.get("language", "en"))
            if tts_res and tts_res.get("audio_base64"):
                st.audio(base64.b64decode(tts_res["audio_base64"]), format="audio/wav")

            # Timeline
            st.markdown("#### 📜 Dispute Resolution Audit Timeline")
            st.markdown(render_timeline_html(ticket_data.get("timeline", [])), unsafe_allow_html=True)
        else:
            st.warning(f"No grievance found with Ticket ID: '{query_ticket}'. Please verify the ticket number or browse Grievance History.")
    else:
        st.info("💡 Tip: Enter your Ticket ID above or select one from your recent submissions to inspect its real-time status.")

# =============================================================================
# VIEW 3: CITIZEN COMPLAINT HISTORY
# =============================================================================
elif nav_choice == NAV_HISTORY:
    st.subheader("📋 Public Grievance History")
    st.write("Browse recent citizen disputes and resolution statuses across Hindi, English, and Hinglish submissions:")

    # Interactive Search & Category Filters
    filter_c1, filter_c2, filter_c3 = st.columns([2, 1.2, 1])
    with filter_c1:
        hist_search = st.text_input("🔍 Search Grievances", placeholder="Filter by citizen name, keyword, or summary...", key="txt_hist_search").strip()
    with filter_c2:
        hist_cat = st.selectbox("Category", ["All"] + GRIEVANCE_CATEGORIES, key="sel_hist_cat")
    with filter_c3:
        hist_status = st.selectbox("Status", ["All", "Open", "In Progress", "Under Investigation", "Resolved", "Rejected"], key="sel_hist_status")

    complaints_list = api_client.list_complaints(
        category=hist_cat if hist_cat != "All" else None,
        status=hist_status if hist_status != "All" else None,
        search_query=hist_search if hist_search else None,
        limit=50
    )

    if complaints_list:
        st.markdown(f"**Found {len(complaints_list)} recorded grievances:**")
        df = pd.DataFrame(complaints_list)
        col_map = {
            "ticket_id": "Ticket ID",
            "citizen_name": "Citizen",
            "language": "Language",
            "submission_channel": "Channel",
            "category_ml": "AI Category",
            "verified_category": "Verified Category",
            "urgency_level_rule": "Urgency",
            "status": "Status",
            "created_at": "Submitted At"
        }
        present_cols = [c for c in col_map.keys() if c in df.columns]
        display_df = df[present_cols].rename(columns=col_map)
        st.dataframe(display_df, use_container_width=True)

        st.markdown("### 🗂️ Grievance Overview Cards")
        for item in complaints_list[:6]:
            st.markdown(render_ticket_card_html(item), unsafe_allow_html=True)
    else:
        st.info("No grievances found matching your search criteria.")

# =============================================================================
# VIEW 4: ADMIN & OFFICER DASHBOARD
# =============================================================================
elif nav_choice == NAV_OFFICER:
    st.subheader("🏛️ Municipal & Support Officer Operations")

    # Session token synchronization
    admin_token = st.session_state.get("admin_token")
    admin_user = st.session_state.get("admin_user")
    if admin_token:
        api_client.auth_token = admin_token

    # Check administrative authentication
    is_admin_auth = False
    if admin_token and admin_user:
        if admin_user.get("role") in ["admin", "officer", "superadmin"]:
            is_admin_auth = True

    if not is_admin_auth:
        st.info("🔒 **Administrative & Officer Authentication Required**\n\nTriage operations, resolution dispatch, department verification, and ML model retraining are restricted to authorized personnel.")

        login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
        with login_col2:
            with st.form("admin_login_form"):
                st.markdown("#### 🔐 Officer Login")
                st.write("Sign in with authorized municipal officer or administrator credentials:")
                input_username = st.text_input("Username", placeholder="e.g. admin or officer_name")
                input_password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In to Operations Console", type="primary", use_container_width=True)

                if submit_login:
                    if not input_username or not input_password:
                        st.error("Please enter both username and password.")
                    else:
                        with st.spinner("Authenticating with backend server..."):
                            auth_res = api_client.login(input_username, input_password)
                        if auth_res.get("success"):
                            user_data = auth_res["data"]["user"]
                            user_role = user_data.get("role", "citizen")
                            if user_role in ["admin", "officer", "superadmin"]:
                                st.session_state["admin_token"] = auth_res["data"]["access_token"]
                                st.session_state["admin_user"] = user_data
                                api_client.auth_token = auth_res["data"]["access_token"]
                                st.success(f"Welcome, {user_data.get('full_name') or user_data['username']}! Logging in...")
                                st.rerun()
                            else:
                                st.error(f"⛔ Access Denied: User '{user_data['username']}' is assigned citizen role ('{user_role}'). Only municipal officers and administrators can access this portal.")
                        else:
                            st.error(f"Authentication failed: {auth_res.get('error', 'Invalid username or password.')}")

    else:
        # Officer Active Session Header
        officer_display = admin_user.get("full_name") or admin_user.get("username")
        sess_c1, sess_c2 = st.columns([4, 1])
        with sess_c1:
            st.success(f"👮 **Active Session:** {officer_display} | **Role:** `{admin_user.get('role', 'officer').upper()}` | **Account:** `{admin_user.get('username')}`")
        with sess_c2:
            if st.button("🚪 Logout", key="btn_admin_logout", type="secondary"):
                tok = st.session_state.get("admin_token")
                api_client.logout(token=tok)
                st.session_state.pop("admin_token", None)
                st.session_state.pop("admin_user", None)
                api_client.auth_token = None
                st.rerun()

        st.write("Triage citizen grievances, update ticket resolution lifecycle, reassign departments, and verify categories:")

        # KPI Summary Cards
        analytics = api_client.get_analytics()
        if analytics:
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.markdown(f'<div class="metric-card"><div class="metric-title">TOTAL TICKETS</div><div class="metric-value">{analytics.get("total_complaints", 0)}</div></div>', unsafe_allow_html=True)
            kpi2.markdown(f'<div class="metric-card"><div class="metric-title">OPEN</div><div class="metric-value" style="color:#2563eb;">{analytics.get("status_distribution", {}).get("Open", 0)}</div></div>', unsafe_allow_html=True)
            kpi3.markdown(f'<div class="metric-card"><div class="metric-title">IN PROGRESS</div><div class="metric-value" style="color:#d97706;">{analytics.get("status_distribution", {}).get("In Progress", 0)}</div></div>', unsafe_allow_html=True)
            kpi4.markdown(f'<div class="metric-card"><div class="metric-title">CRITICAL URGENCY</div><div class="metric-value" style="color:#dc2626;">{analytics.get("urgency_distribution", {}).get("Critical", 0)}</div></div>', unsafe_allow_html=True)
            kpi5.markdown(f'<div class="metric-card"><div class="metric-title">RESOLVED</div><div class="metric-value" style="color:#059669;">{analytics.get("status_distribution", {}).get("Resolved", 0)}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Search & Filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            cat_filter = st.selectbox("Category Filter", ["All"] + GRIEVANCE_CATEGORIES)
        with f_col2:
            status_filter = st.selectbox("Status Filter", ["All"] + STATUS_OPTIONS)
        with f_col3:
            urgency_filter = st.selectbox("Urgency Filter", ["All"] + URGENCY_LEVELS)
        with f_col4:
            search_query = st.text_input("Search Text / Ticket", placeholder="Search citizen, words...")

        filtered_complaints = api_client.list_complaints(
            category=cat_filter,
            status=status_filter,
            urgency=urgency_filter,
            search_query=search_query
        )

        st.markdown(f"**Found {len(filtered_complaints)} matching complaints.**")

        # Ticket Management & Status Transition Form
        if filtered_complaints:
            selected_ticket_id = st.selectbox(
                "Select Ticket to Manage / Update Resolution:",
                options=[c["ticket_id"] for c in filtered_complaints],
                format_func=lambda tid: f"{tid} - {next((c['citizen_name'] + ' (' + c['category_ml'] + ' - ' + c['urgency_level_rule'] + ')') for c in filtered_complaints if c['ticket_id'] == tid)}"
            )

            selected_ticket = next((c for c in filtered_complaints if c["ticket_id"] == selected_ticket_id), None)
            if selected_ticket:
                st.markdown(render_ticket_card_html(selected_ticket), unsafe_allow_html=True)

                # Gnani Evon Sovereign Indic Guidance for Officer
                ticket_summary = selected_ticket.get("summary") or selected_ticket.get("category_ml", "")
                evon_triage = api_client.analyze_with_evon(ticket_summary, language=selected_ticket.get("language", "hi"))
                if evon_triage:
                    triage_badge = "LIVE API" if not evon_triage.get("is_placeholder") else "ACTIVE"
                    badge_bg = "#065f46" if not evon_triage.get("is_placeholder") else "#1e3a8a"
                    badge_col = "#a7f3d0" if not evon_triage.get("is_placeholder") else "#bfdbfe"
                    st.markdown(f"""
                    <div style="background:#0f172a; border-left:4px solid #3b82f6; padding:0.8rem 1.1rem; border-radius:8px; margin-bottom:1rem; border:1px solid #334155;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                            <b style="color:#ffffff;">🇮🇳 Gnani Evon Sovereign Indic Guidance for Triage Officer:</b>
                            <span style="background:{badge_bg}; color:{badge_col}; font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:700;">{evon_triage.get('provider')} ({triage_badge})</span>
                        </div>
                        <div style="color:#93c5fd; font-size:0.88rem; margin-bottom:0.25rem;"><b>Root Cause Domain:</b> {evon_triage.get('root_cause_domain')}</div>
                        <div style="color:#f1f5f9; font-size:0.86rem; margin-bottom:0.25rem;"><b>Indic Nuance:</b> {evon_triage.get('reasoning')}</div>
                        <div style="color:#34d399; font-size:0.86rem;"><b>Recommended Next Action:</b> {evon_triage.get('actionable_recommendation')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("📝 Update Ticket Status, Department & Resolution Notes", expanded=True):
                    up_c1, up_c2 = st.columns(2)
                    with up_c1:
                        new_status = st.selectbox("Update Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(selected_ticket["status"]) if selected_ticket["status"] in STATUS_OPTIONS else 0)
                    with up_c2:
                        new_dept = st.text_input("Assigned Department", value=selected_ticket.get("assigned_department") or "")

                    up_c3, up_c4 = st.columns(2)
                    with up_c3:
                        current_cat = selected_ticket.get("verified_category") or selected_ticket.get("category_ml")
                        cat_idx = GRIEVANCE_CATEGORIES.index(current_cat) if current_cat in GRIEVANCE_CATEGORIES else 0
                        verified_cat_choice = st.selectbox(
                            "Human-Verified Category (Officer Review)",
                            GRIEVANCE_CATEGORIES,
                            index=cat_idx,
                            help="Verify or correct the ML model's categorization. Only officer-verified labels are allowed to augment the retraining dataset."
                        )
                    with up_c4:
                        clear_review_flag = st.checkbox(
                            "Mark Manual Review Resolved",
                            value=not selected_ticket.get("needs_manual_review", False)
                        )

                    officer_notes = st.text_area("Resolution Notes", placeholder="Detail actions taken (e.g. dispatched water tanker, electrical lineman assigned)...")

                    if st.button("💾 Save Changes", type="primary"):
                        update_payload = {
                            "status": new_status,
                            "assigned_department": new_dept.strip() if new_dept else None,
                            "verified_category": verified_cat_choice,
                            "needs_manual_review": not clear_review_flag,
                            "action_taken": f"Status updated to {new_status} by {officer_display}. Notes: {officer_notes or 'No extra notes.'}"
                        }
                        with st.spinner("Updating ticket record..."):
                            up_res = api_client.update_complaint(selected_ticket_id, update_payload)
                        if up_res and "complaint" in up_res:
                            st.success("✅ Ticket status and verification updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Update failed: {up_res.get('error', 'Unknown error')}")

        # ML Retraining Console (Restricted to Admin Role)
        st.markdown("---")
        st.markdown("### 🤖 ML Model Retraining")
        is_user_admin = admin_user.get("role") in ["admin", "superadmin"]

        if is_user_admin:
            st.write("Administrators can retrain the Scikit-Learn classifier using human-verified complaint data:")
            if st.button("🚀 Trigger ML Retraining"):
                with st.spinner("Retraining model on verified records..."):
                    train_res = api_client.retrain_model()
                if train_res.get("success"):
                    st.success(f"✅ Model retraining complete! {train_res.get('message', '')}")
                    st.json(train_res.get("data", {}))
                else:
                    st.error(f"Retraining failed: {train_res.get('error', 'Unknown error')}")
        else:
            st.info("ℹ️ **Notice:** ML model retraining is restricted to Administrator roles. Regional officers can triage grievances and verify categories.")

# =============================================================================
# VIEW 5: CIVIC ANALYTICS
# =============================================================================
elif nav_choice == NAV_ANALYTICS:
    st.subheader("📊 Grievance Intelligence & Operational Analytics")

    tab_analytics, tab_ml_eval = st.tabs([
        "📈 Operational Dispute Trends",
        "🔬 Scikit-Learn Model Evaluation Report"
    ])

    with tab_analytics:
        st.write("Interactive Plotly analytics across complaint categories, urgency levels, languages, and resolution stages:")
        analytics = api_client.get_analytics()
        if analytics:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(render_category_donut(analytics.get("category_distribution", {})), use_container_width=True)
            with c2:
                st.plotly_chart(render_urgency_bar(analytics.get("urgency_distribution", {})), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(render_status_pipeline(analytics.get("status_distribution", {})), use_container_width=True)
            with c4:
                st.plotly_chart(render_language_donut(analytics.get("language_distribution", {})), use_container_width=True)

            st.plotly_chart(render_sentiment_bar(analytics.get("sentiment_distribution", {})), use_container_width=True)
        else:
            st.warning("Analytics unavailable. Please verify backend server is running.")

    with tab_ml_eval:
        st.markdown("### 🔬 Scikit-Learn Pipeline Evaluation (Empirical Test Results)")
        st.caption("Empirical metrics measured on held-out test data:")

        eval_report = api_client.get_ml_evaluation()
        if eval_report:
            emp = eval_report.get("empirical_metrics", {})
            meta = eval_report.get("metadata", {})

            # KPI Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-card"><div class="metric-title">HELD-OUT TEST ACCURACY</div><div class="metric-value">{emp.get("test_accuracy", 0.0) * 100:.1f}%</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-title">5-FOLD CV MEAN ACC</div><div class="metric-value">{emp.get("cv_5fold_mean_accuracy", 0.0) * 100:.1f}%</div><span style="font-size:0.75rem; color:#cbd5e1;">(±{emp.get("cv_5fold_std", 0.0)*100:.1f}%)</span></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-title">MACRO F1-SCORE</div><div class="metric-value">{emp.get("macro_f1", 0.0):.3f}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-title">WEIGHTED F1-SCORE</div><div class="metric-value">{emp.get("weighted_f1", 0.0):.3f}</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Confusion Matrix Heatmap
            cm_obj = eval_report.get("confusion_matrix", {})
            if cm_obj and "matrix" in cm_obj and "labels" in cm_obj:
                st.plotly_chart(render_confusion_matrix_heatmap(cm_obj["matrix"], cm_obj["labels"]), use_container_width=True)

            # Classification Report Table
            st.markdown("#### 📋 Classification Performance per Category")
            clf_rep = eval_report.get("classification_report", {})
            if clf_rep:
                rep_rows = []
                for cls_name in meta.get("classes", []):
                    if cls_name in clf_rep:
                        row = clf_rep[cls_name]
                        rep_rows.append({
                            "Category": cls_name,
                            "Precision": f"{row.get('precision', 0):.2f}",
                            "Recall": f"{row.get('recall', 0):.2f}",
                            "F1-Score": f"{row.get('f1-score', 0):.2f}",
                            "Test Support": int(row.get('support', 0))
                        })
                if rep_rows:
                    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Evaluation report not yet available.")

# =============================================================================
# VIEW 6: GNANI VOICE ENGINE HEALTH & SYSTEM STATUS
# =============================================================================
elif nav_choice == NAV_DIAGNOSTICS:
    st.subheader("⚡ Gnani AI Engine Health & System Diagnostics")
    st.write("Live connectivity and configuration status for **Gnani Prisma STT** (Speech-to-Text), **Gnani Timbre TTS** (Text-to-Speech), and **Gnani Evon v3.3** (Sovereign Indic LLM / Artha Stack):")

    health = api_client.get_health()

    diag_c1, diag_c2, diag_c3 = st.columns(3)

    with diag_c1:
        st.markdown("### 🎙️ Gnani Prisma STT")
        stt_info = health.get("speech_services", {}).get("stt", {})
        is_stt_live = stt_info.get("configured", False)

        if is_stt_live:
            st.success("🟢 **LIVE CONNECTED**")
        else:
            st.warning("⚠️ **PLACEHOLDER MODE**")

        st.markdown(f"""
        - **Endpoint:** `{stt_info.get('endpoint', 'N/A')}`
        - **Status:** `{stt_info.get('status', 'N/A')}`
        - **Dialects:** Hindi (`hi-IN`), English (`en-IN`), Hinglish (`hi-en`)
        - **Note:** {stt_info.get('message', '')}
        """)

    with diag_c2:
        st.markdown("### 🔊 Gnani Timbre TTS")
        tts_info = health.get("speech_services", {}).get("tts", {})
        is_tts_live = tts_info.get("configured", False)

        if is_tts_live:
            st.success("🟢 **LIVE CONNECTED**")
        else:
            st.warning("⚠️ **PLACEHOLDER MODE**")

        st.markdown(f"""
        - **Endpoint:** `{tts_info.get('endpoint', 'N/A')}`
        - **Status:** `{tts_info.get('status', 'N/A')}`
        - **Audio:** 16kHz PCM WAV Audio
        - **Note:** {tts_info.get('message', '')}
        """)

    with diag_c3:
        st.markdown("### 🧠 Gnani Evon v3.3 (Artha)")
        evon_info = health.get("speech_services", {}).get("evon", {}) or health.get("gnani_ai_stack", {}).get("evon_indic_llm", {})
        is_evon_live = evon_info.get("configured", False)

        if is_evon_live:
            st.success("🟢 **LIVE CONNECTED**")
        else:
            st.warning("⚠️ **PLACEHOLDER MODE**")

        st.markdown(f"""
        - **Model:** `{evon_info.get('model', 'gnani-evon-v3.3')}`
        - **Status:** `{evon_info.get('status', 'Ready')}`
        - **Capabilities:** Sovereign Indic Reasoning, Grievance Nuance Analysis
        - **Note:** {evon_info.get('message', '')}
        """)

    st.markdown("---")
    st.markdown("### 🤖 Intelligence Subsystems")
    ml_c1, ml_c2, ml_c3 = st.columns(3)

    with ml_c1:
        st.markdown("#### Category Classification")
        st.markdown(render_model_attribution_badge("ml"), unsafe_allow_html=True)
        st.write("""
        - **Model:** TF-IDF + Logistic Regression
        - **Classes:** 8 Municipal & Civic categories
        - **Languages:** English, Hindi, Hinglish
        """)

    with ml_c2:
        st.markdown("#### Urgency & Severity Engine")
        st.markdown(render_model_attribution_badge("rule"), unsafe_allow_html=True)
        st.write("""
        - **Method:** Multi-factor Heuristic Engine
        - **Tiers:** Critical, High, Medium, Low
        - **Duration Awareness:** Distinct scaling for 10-day vs 3-day outages
        """)

    with ml_c3:
        st.markdown("#### Sentiment & Summary")
        st.markdown(render_model_attribution_badge("sentiment"), unsafe_allow_html=True)
        st.markdown(render_model_attribution_badge("summarizer"), unsafe_allow_html=True)
        st.write("""
        - **Sentiment:** Frustration & polarity scoring
        - **Summary:** Extractive salience sentence picker
        """)

    st.markdown("---")
    st.markdown("### 🔑 Gnani API Credentials Configuration")
    st.markdown("""
    To enable live Gnani AI processing, add your API keys to the root `.env` file:
    ```bash
    GNANI_API_KEY=your_gnani_master_key_here
    # Or individual service overrides:
    GNANI_PRISMA_API_KEY=your_gnani_prisma_api_key_here
    GNANI_TIMBRE_API_KEY=your_gnani_timbre_api_key_here
    GNANI_EVON_API_KEY=your_gnani_evon_api_key_here
    ```
    *If credentials remain blank, AwaazDesk operates in transparent placeholder mode with zero crashes.*
    """)
