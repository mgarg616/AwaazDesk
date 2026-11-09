"""
AwaazDesk - UI Components & HTML Render Helpers
Provides reusable, humanized visual components, accessible status tags, and attribution labels.
Configured with High-Visibility White Typography and modern contrast styling.
"""
from typing import Dict, Any, List

def render_model_attribution_badge(engine_type: str, detail: str = "") -> str:
    """
    Renders distinct, human-readable visual attribution badges:
    Clearly distinguishes ML predictions from rule-based calculations.
    """
    if engine_type == "ml":
        return f'<span class="badge-ml">🤖 AI Model: TF-IDF Classifier {detail}</span>'
    elif engine_type == "rule":
        return f'<span class="badge-rule">⚖️ Priority Rules Matrix {detail}</span>'
    elif engine_type == "sentiment":
        return f'<span class="badge-sentiment">🧠 Sentiment Analysis {detail}</span>'
    elif engine_type == "summarizer":
        return f'<span class="badge-summarizer">📝 Executive Summary {detail}</span>'
    return ""

def render_urgency_badge(level: str, score: float = 0.0) -> str:
    """Renders humanized high-contrast urgency tag."""
    lvl = (level or "Low").capitalize()
    css_class = f"urgency-{lvl.lower()}"

    labels = {
        "Critical": "🔴 Critical Priority",
        "High": "🟠 High Priority",
        "Medium": "🟡 Medium Priority",
        "Low": "🟢 Routine / Low"
    }
    label = labels.get(lvl, f"⚪ {lvl}")
    return f'<span class="{css_class}">{label} ({score}/10)</span>'

def render_integration_badge(provider: str, is_placeholder: bool) -> str:
    """
    Renders Gnani STT/TTS transparency badge.
    Maintains transparency: clearly indicates live connection vs offline test mode.
    """
    if is_placeholder:
        return f'<span class="badge-placeholder">⚠️ {provider} • Integration Placeholder</span>'
    return f'<span class="badge-live">🟢 {provider} • Live Connected</span>'

def render_ticket_card_html(ticket: Dict[str, Any]) -> str:
    """Renders formatted, accessible HTML card for a grievance ticket with high-contrast text."""
    triggers_html = ""
    if ticket.get("urgency_triggers"):
        trig_list = ticket["urgency_triggers"]
        if isinstance(trig_list, str):
            try:
                import json
                trig_list = json.loads(trig_list)
            except Exception:
                trig_list = []
        if isinstance(trig_list, list) and trig_list:
            items = "".join([f'<span style="background:#334155; color:#f8fafc; padding:3px 8px; margin:2px; border-radius:6px; font-size:0.78rem; border:1px solid #475569; font-weight:500;">{t}</span>' for t in trig_list[:5]])
            triggers_html = f'<div style="margin-top:0.6rem; font-size:0.83rem; color:#cbd5e1;"><b style="color:#ffffff;">Key Indicators:</b> {items}</div>'

    verified_cat = ticket.get("verified_category")
    verified_by = ticket.get("verified_by")
    needs_review = ticket.get("needs_manual_review")
    conf_pct = int(ticket.get("category_confidence", 0) * 100)

    if verified_cat:
        category_html = f"""
        <div style="margin-bottom:0.75rem;">
            <span style="font-size:0.9rem; color:#cbd5e1;">Category:</span>
            <span style="color:#34d399; font-weight:800; font-size:1.05rem; margin-left:0.3rem;">{verified_cat}</span>
            <span style="background:#064e3b; color:#a7f3d0; border:1px solid #059669; padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.76rem; margin-left:0.5rem;">
                ✅ Verified by {verified_by or 'Officer'}
            </span>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:3px;">
                Initial Model Prediction: <span style="color:#ffffff;">{ticket.get('category_ml')}</span> (Confidence: {conf_pct}%)
            </div>
        </div>
        """
    else:
        if needs_review:
            review_badge = '<span style="background:#451a03; color:#fde68a; border:1px solid #d97706; padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.76rem; margin-left:0.4rem;">⚠️ Pending Officer Review (Low Confidence &lt; 35%)</span>'
        else:
            review_badge = '<span style="background:#1e293b; color:#cbd5e1; border:1px solid #475569; padding:3px 10px; border-radius:6px; font-size:0.76rem; margin-left:0.4rem;">🤖 Auto-Predicted (Awaiting Officer Verification)</span>'

        category_html = f"""
        <div style="margin-bottom:0.75rem;">
            <span style="font-size:0.9rem; color:#cbd5e1;">Predicted Category:</span>
            <span style="color:#60a5fa; font-weight:800; font-size:1.05rem; margin-left:0.3rem;">{ticket.get('category_ml')}</span>
            <span style="margin-left:0.4rem;">{render_model_attribution_badge('ml', f'({conf_pct}%)')}</span>
            {review_badge}
        </div>
        """

    status_colors = {
        "Open": "#60a5fa",
        "In Progress": "#fbbf24",
        "Resolved": "#34d399",
        "Rejected": "#f87171"
    }
    status_val = ticket.get("status", "Open")
    status_color = status_colors.get(status_val, "#60a5fa")

    html = f"""
    <div class="ticket-box">
        <div class="ticket-header">
            <div>
                <span class="ticket-id">{ticket.get('ticket_id')}</span>
                <span style="margin-left:0.8rem; font-weight:700; color:#ffffff; font-size:1.1rem;">{ticket.get('citizen_name')}</span>
                <span style="color:#94a3b8; font-size:0.85rem; margin-left:0.5rem;">({ticket.get('language', '').upper()} • {ticket.get('submission_channel', '').capitalize()})</span>
            </div>
            <div>
                {render_urgency_badge(ticket.get('urgency_level_rule', 'Low'), ticket.get('urgency_score', 0.0))}
            </div>
        </div>
        {category_html}
        <div style="background:#0f172a; border:1px solid #334155; padding:0.9rem 1.1rem; border-radius:10px; margin-bottom:0.75rem;">
            <div style="font-size:0.82rem; font-weight:700; color:#94a3b8; margin-bottom:0.3rem;">
                📌 EXECUTIVE SUMMARY {render_model_attribution_badge('summarizer')}
            </div>
            <div style="color:#f8fafc; font-size:0.95rem; line-height:1.5;">{ticket.get('summary')}</div>
        </div>
        <div style="font-size:0.88rem; color:#cbd5e1; margin-bottom:0.4rem; display:flex; flex-wrap:wrap; gap:1.2rem;">
            <div><b style="color:#94a3b8;">Assigned Department:</b> <span style="color:#ffffff; font-weight:600;">{ticket.get('assigned_department') or 'Municipal Operations / Unassigned'}</span></div>
            <div><b style="color:#94a3b8;">Status:</b> <span style="font-weight:800; color:{status_color};">{status_val}</span></div>
            <div><b style="color:#94a3b8;">Created:</b> <span style="color:#ffffff;">{ticket.get('created_at', '')[:16].replace('T', ' ')}</span></div>
        </div>
        {triggers_html}
    </div>
    """
    return html

def render_timeline_html(events: List[Dict[str, Any]]) -> str:
    """Renders vertical audit trail nodes for a complaint ticket with bright white text."""
    if not events:
        return "<p style='color:#94a3b8; padding:1rem;'>No previous actions recorded on this ticket.</p>"

    html = '<div style="margin-top:1.2rem;">'
    for ev in events:
        timestamp = ev.get("timestamp", "")[:19].replace("T", " ")
        html += f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-time" style="color:#94a3b8; font-size:0.82rem;">{timestamp} • Action by: <b style="color:#ffffff;">{ev.get('performed_by', 'System')}</b></div>
            <div class="timeline-title" style="color:#ffffff; font-weight:700; font-size:0.95rem; margin-top:0.2rem;">{ev.get('action', 'Action')}</div>
            <div class="timeline-desc" style="color:#cbd5e1; font-size:0.88rem; margin-top:0.2rem;">{ev.get('details', '')}</div>
        </div>
        """
    html += '</div>'
    return html
