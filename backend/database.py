"""
AwaazDesk - Database Management & Persistent SQLite Storage
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import (
    DB_PATH,
    GRIEVANCE_CATEGORIES,
    ADMIN_INITIAL_USERNAME,
    ADMIN_INITIAL_PASSWORD
)

logger = logging.getLogger("awaazdesk.database")

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database with row_factory configured."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(seed_sample_data: bool = True):
    """Initializes tables in SQLite database if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create complaints table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id TEXT UNIQUE NOT NULL,
        citizen_name TEXT NOT NULL,
        citizen_phone TEXT,
        citizen_email TEXT,
        language TEXT NOT NULL,
        submission_channel TEXT NOT NULL,
        original_audio_path TEXT,
        raw_text TEXT NOT NULL,
        transcription_confidence REAL DEFAULT 1.0,
        category_ml TEXT NOT NULL,
        category_confidence REAL NOT NULL,
        category_top_probabilities TEXT,
        urgency_level_rule TEXT NOT NULL,
        urgency_score REAL NOT NULL,
        urgency_triggers TEXT,
        sentiment_polarity TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        summary TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Open',
        assigned_department TEXT,
        admin_notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        verified_category TEXT,
        verified_by TEXT,
        verified_at TEXT,
        needs_manual_review INTEGER DEFAULT 0
    );
    """)

    # Backward-compatible migration: Check and add verification and review columns if missing
    cursor.execute("PRAGMA table_info(complaints);")
    existing_columns = {row["name"] for row in cursor.fetchall()}
    if "verified_category" not in existing_columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN verified_category TEXT;")
    if "verified_by" not in existing_columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN verified_by TEXT;")
    if "verified_at" not in existing_columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN verified_at TEXT;")
    if "needs_manual_review" not in existing_columns:
        cursor.execute("ALTER TABLE complaints ADD COLUMN needs_manual_review INTEGER DEFAULT 0;")

    # Create timeline table for ticket audit logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticket_timeline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id TEXT NOT NULL,
        action TEXT NOT NULL,
        performed_by TEXT NOT NULL,
        details TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (ticket_id) REFERENCES complaints(ticket_id) ON DELETE CASCADE
    );
    """)

    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticket_id ON complaints(ticket_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON complaints(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON complaints(category_ml);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_verified_category ON complaints(verified_category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_needs_manual_review ON complaints(needs_manual_review);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_urgency ON complaints(urgency_level_rule);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON complaints(created_at);")

    # Create users table for authentication & RBAC
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        full_name TEXT,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'citizen',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")

    # Create revoked_tokens table for server-side JWT invalidation on logout
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS revoked_tokens (
        jti TEXT PRIMARY KEY,
        revoked_at TEXT NOT NULL,
        expires_at INTEGER NOT NULL
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_revoked_jti ON revoked_tokens(jti);")

    conn.commit()

    # Check if empty, seed if requested
    cursor.execute("SELECT COUNT(*) as count FROM complaints;")
    count = cursor.fetchone()["count"]
    if count == 0 and seed_sample_data:
        _seed_realistic_data(conn)

    # Ensure at least one admin account exists if credentials explicitly supplied in environment
    cursor.execute("SELECT COUNT(*) as admin_count FROM users WHERE role IN ('admin', 'superadmin');")
    admin_count = cursor.fetchone()["admin_count"]
    if admin_count == 0:
        _bootstrap_admin_user(conn)

    conn.close()
    logger.info("Database initialized successfully at %s", DB_PATH)

def _bootstrap_admin_user(conn: sqlite3.Connection):
    """
    Bootstraps an initial admin account ONLY IF credentials are explicitly supplied in .env.
    Never creates hardcoded or fallback admin accounts with default passwords.
    """
    if not ADMIN_INITIAL_USERNAME or not ADMIN_INITIAL_PASSWORD or len(ADMIN_INITIAL_PASSWORD) < 8:
        logger.warning(
            "No administrative account found in database and no valid ADMIN_INITIAL_PASSWORD "
            "provided in environment. Please provision an initial administrator via CLI: "
            "'python -m backend.security.create_admin'"
        )
        return

    from backend.security.auth import hash_password
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    hashed = hash_password(ADMIN_INITIAL_PASSWORD)
    cursor.execute("""
    INSERT OR IGNORE INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
    VALUES (?, ?, ?, ?, ?, 1, ?)
    """, (
        ADMIN_INITIAL_USERNAME,
        "admin@awaazdesk.gov.in",
        "Lead Grievance Officer",
        hashed,
        "admin",
        now_iso
    ))
    conn.commit()
    logger.info("Admin bootstrap complete: initial admin '%s' provisioned.", ADMIN_INITIAL_USERNAME)


def _seed_realistic_data(conn: sqlite3.Connection):
    """Seeds rich, realistic multilingual complaints across Hindi, English, and Hinglish."""
    sample_complaints = [
        {
            "ticket_id": "AWA-2026-8812",
            "citizen_name": "Rajesh Kumar Sharma",
            "citizen_phone": "+91 98112 43210",
            "citizen_email": "rajesh.sharma@example.com",
            "language": "hi",
            "submission_channel": "voice",
            "original_audio_path": None,
            "raw_text": "हमारे इलाके सेक्टर 4 में पिछले तीन दिनों से पीने का पानी बिल्कुल नहीं आ रहा है। छोटे बच्चे और बुजुर्ग बहुत परेशान हैं। पाइपलाइन में लीकेज की वजह से गंदा पानी भी सड़क पर बह रहा है। कृपया जल्द से जल्द पानी का टैंकर भेजें।",
            "transcription_confidence": 0.94,
            "category_ml": "Water Supply & Drainage",
            "category_confidence": 0.96,
            "category_top_probabilities": json.dumps({"Water Supply & Drainage": 0.96, "Civic Infrastructure & Sanitation": 0.03, "Other Grievances": 0.01}),
            "urgency_level_rule": "High",
            "urgency_score": 7.8,
            "urgency_triggers": json.dumps(["तीन दिनों से पानी नहीं", "गंदा पानी", "परेशान"]),
            "sentiment_polarity": "Negative",
            "sentiment_score": -0.72,
            "summary": "सेक्टर 4 में 3 दिनों से पेयजल आपूर्ति ठप है और पाइपलाइन लीकेज के कारण सड़क पर दूषित जल बह रहा है। तत्काल पानी के टैंकर की मांग की गई है।",
            "status": "In Progress",
            "assigned_department": "Delhi Jal Board / Water Works",
            "admin_notes": "Water tanker dispatched to Sector 4. Maintenance crew en route to repair main pipeline.",
            "created_at": "2026-11-03T10:15:00",
            "updated_at": "2026-11-03T14:30:00"
        },
        {
            "ticket_id": "AWA-2026-9104",
            "citizen_name": "Pooja Verma",
            "citizen_phone": "+91 97234 11890",
            "citizen_email": "pooja.v@example.com",
            "language": "hinglish",
            "submission_channel": "voice",
            "original_audio_path": None,
            "raw_text": "Main road par open transformer sparks nikal raha hai, aur nearby school students walk karte hain. Bohot dangerous situation hai, kisi ko severe electric shock lag sakta hai. Please urgently repair karwaiye electricity board se!",
            "transcription_confidence": 0.91,
            "category_ml": "Electricity & Power Supply",
            "category_confidence": 0.98,
            "category_top_probabilities": json.dumps({"Electricity & Power Supply": 0.98, "Civic Infrastructure & Sanitation": 0.02}),
            "urgency_level_rule": "Critical",
            "urgency_score": 9.5,
            "urgency_triggers": json.dumps(["open transformer", "sparks", "dangerous", "severe electric shock", "urgently"]),
            "sentiment_polarity": "Negative",
            "sentiment_score": -0.85,
            "summary": "Main road par transformer se sparks nikal rahe hain, school ke paas hone se life-threatening danger hai. Urgent electrical line intervention required.",
            "status": "Under Investigation",
            "assigned_department": "Electricity Distribution Board (DISCOM)",
            "admin_notes": "Emergency lineman team alerted. Feeder cut off temporarily to isolate sparking transformer.",
            "created_at": "2026-11-04T08:20:00",
            "updated_at": "2026-11-04T09:00:00"
        },
        {
            "ticket_id": "AWA-2026-6421",
            "citizen_name": "Dr. Anita Desai",
            "citizen_phone": "+91 94000 55678",
            "citizen_email": "anita.desai@example.org",
            "language": "en",
            "submission_channel": "text",
            "original_audio_path": None,
            "raw_text": "The primary healthcare center in Ward 12 lacks basic emergency medicines and anti-rabies vaccines. Yesterday a patient bitten by a stray dog had to be referred 25 km away because no doctor was on duty. Kindly inspect and ensure medical supplies.",
            "transcription_confidence": 1.0,
            "category_ml": "Healthcare & Civil Supplies",
            "category_confidence": 0.93,
            "category_top_probabilities": json.dumps({"Healthcare & Civil Supplies": 0.93, "Civic Infrastructure & Sanitation": 0.05, "Other Grievances": 0.02}),
            "urgency_level_rule": "High",
            "urgency_score": 8.0,
            "urgency_triggers": json.dumps(["emergency medicines", "anti-rabies vaccines", "no doctor on duty"]),
            "sentiment_polarity": "Negative",
            "sentiment_score": -0.65,
            "summary": "Primary Health Centre in Ward 12 lacks critical emergency medicines and vaccines, with no doctor on duty during dog-bite emergency. Urgent replenishment demanded.",
            "status": "Open",
            "assigned_department": "District Health Department",
            "admin_notes": None,
            "created_at": "2026-11-04T11:45:00",
            "updated_at": "2026-11-04T11:45:00"
        },
        {
            "ticket_id": "AWA-2026-3398",
            "citizen_name": "Manpreet Singh",
            "citizen_phone": "+91 98881 22910",
            "citizen_email": "m.singh@example.com",
            "language": "hinglish",
            "submission_channel": "text",
            "original_audio_path": None,
            "raw_text": "Mera SBI account se ATM withdrawal fail ho gaya, Rs 10000 debit ho chuka hai but ATM ne cash dispense nahi kiya. Transaction id 8839210 hai. Bank helpline par koi pick nahi kar raha hai.",
            "transcription_confidence": 1.0,
            "category_ml": "Banking & Financial Services",
            "category_confidence": 0.97,
            "category_top_probabilities": json.dumps({"Banking & Financial Services": 0.97, "Telecommunications & Internet": 0.02, "Other Grievances": 0.01}),
            "urgency_level_rule": "Medium",
            "urgency_score": 5.8,
            "urgency_triggers": json.dumps(["ATM withdrawal fail", "cash dispense nahi kiya", "debit"]),
            "sentiment_polarity": "Negative",
            "sentiment_score": -0.58,
            "summary": "ATM par 10000 rupaye debit hue par cash dispense nahi hua. Bank helpline unresponsive.",
            "status": "Resolved",
            "assigned_department": "Banking Ombudsman & Customer Grievances",
            "admin_notes": "Auto-reversal completed by switch network. Amount credited back to citizen account on 03-Nov.",
            "created_at": "2026-11-01T15:20:00",
            "updated_at": "2026-11-03T16:10:00"
        },
        {
            "ticket_id": "AWA-2026-5120",
            "citizen_name": "Venkatesh Iyer",
            "citizen_phone": "+91 99401 77654",
            "citizen_email": "v.iyer@example.com",
            "language": "en",
            "submission_channel": "voice",
            "original_audio_path": None,
            "raw_text": "Deep potholes on the 100 Feet Ring Road between 14th Main and Silk Board junction are causing terrible traffic jams and two-wheeler accidents daily. Please patch the asphalt immediately before heavy monsoon rains.",
            "transcription_confidence": 0.95,
            "category_ml": "Civic Infrastructure & Sanitation",
            "category_confidence": 0.95,
            "category_top_probabilities": json.dumps({"Civic Infrastructure & Sanitation": 0.95, "Public Transport & Railways": 0.04, "Other Grievances": 0.01}),
            "urgency_level_rule": "High",
            "urgency_score": 7.5,
            "urgency_triggers": json.dumps(["deep potholes", "accidents daily", "traffic jams"]),
            "sentiment_polarity": "Negative",
            "sentiment_score": -0.68,
            "summary": "Deep potholes on 100 Feet Ring Road causing daily accidents and traffic congestion; urgent asphalt road repairs requested.",
            "status": "In Progress",
            "assigned_department": "Municipal Public Works Department (PWD)",
            "admin_notes": "Tender work order issued for asphalt cold-patching. Work scheduled for night hours.",
            "created_at": "2026-11-02T13:00:00",
            "updated_at": "2026-11-03T18:20:00"
        }
    ]

    cursor = conn.cursor()
    for item in sample_complaints:
        cursor.execute("""
        INSERT INTO complaints (
            ticket_id, citizen_name, citizen_phone, citizen_email, language,
            submission_channel, original_audio_path, raw_text, transcription_confidence,
            category_ml, category_confidence, category_top_probabilities,
            urgency_level_rule, urgency_score, urgency_triggers,
            sentiment_polarity, sentiment_score, summary, status,
            assigned_department, admin_notes, created_at, updated_at
        ) VALUES (
            :ticket_id, :citizen_name, :citizen_phone, :citizen_email, :language,
            :submission_channel, :original_audio_path, :raw_text, :transcription_confidence,
            :category_ml, :category_confidence, :category_top_probabilities,
            :urgency_level_rule, :urgency_score, :urgency_triggers,
            :sentiment_polarity, :sentiment_score, :summary, :status,
            :assigned_department, :admin_notes, :created_at, :updated_at
        )
        """, item)

        # Add initial timeline event
        cursor.execute("""
        INSERT INTO ticket_timeline (ticket_id, action, performed_by, details, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (
            item["ticket_id"],
            "Complaint Created",
            f"Citizen ({item['citizen_name']})",
            f"Submitted via {item['submission_channel'].capitalize()} in {item['language'].upper()}",
            item["created_at"]
        ))

        if item["status"] != "Open":
            cursor.execute("""
            INSERT INTO ticket_timeline (ticket_id, action, performed_by, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """, (
                item["ticket_id"],
                f"Status updated to '{item['status']}'",
                "Admin Desk",
                item["admin_notes"] or "Status transitioned by authority.",
                item["updated_at"]
            ))

    conn.commit()
    logger.info("Seeded %d realistic sample complaints.", len(sample_complaints))

def create_complaint(data: Dict[str, Any]) -> str:
    """Inserts a new complaint and logs the creation in ticket_timeline."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now_iso = datetime.now().isoformat()
    data.setdefault("created_at", now_iso)
    data.setdefault("updated_at", now_iso)
    data.setdefault("status", "Open")
    data.setdefault("verified_category", None)
    data.setdefault("verified_by", None)
    data.setdefault("verified_at", None)
    data.setdefault("needs_manual_review", 0)

    cursor.execute("""
    INSERT INTO complaints (
        ticket_id, citizen_name, citizen_phone, citizen_email, language,
        submission_channel, original_audio_path, raw_text, transcription_confidence,
        category_ml, category_confidence, category_top_probabilities,
        urgency_level_rule, urgency_score, urgency_triggers,
        sentiment_polarity, sentiment_score, summary, status,
        assigned_department, admin_notes, created_at, updated_at,
        verified_category, verified_by, verified_at, needs_manual_review
    ) VALUES (
        :ticket_id, :citizen_name, :citizen_phone, :citizen_email, :language,
        :submission_channel, :original_audio_path, :raw_text, :transcription_confidence,
        :category_ml, :category_confidence, :category_top_probabilities,
        :urgency_level_rule, :urgency_score, :urgency_triggers,
        :sentiment_polarity, :sentiment_score, :summary, :status,
        :assigned_department, :admin_notes, :created_at, :updated_at,
        :verified_category, :verified_by, :verified_at, :needs_manual_review
    )
    """, data)

    # Insert timeline audit
    cursor.execute("""
    INSERT INTO ticket_timeline (ticket_id, action, performed_by, details, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (
        data["ticket_id"],
        "Grievance Lodged",
        f"Citizen ({data.get('citizen_name', 'Anonymous')})",
        f"Channel: {data.get('submission_channel', 'text')}, Language: {data.get('language', 'en')}",
        data["created_at"]
    ))

    conn.commit()
    conn.close()
    return data["ticket_id"]

def get_complaint_by_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single complaint by ticket ID (case-insensitive & trimmed)."""
    clean_id = (ticket_id or "").strip()
    if not clean_id:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints WHERE UPPER(ticket_id) = UPPER(?)", (clean_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    res = dict(row)
    if res.get("category_top_probabilities"):
        try:
            res["category_top_probabilities"] = json.loads(res["category_top_probabilities"])
        except Exception:
            pass
    if res.get("urgency_triggers"):
        try:
            res["urgency_triggers"] = json.loads(res["urgency_triggers"])
        except Exception:
            pass
    return res

def list_complaints(
    category: Optional[str] = None,
    status: Optional[str] = None,
    urgency: Optional[str] = None,
    language: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Queries complaints with optional multi-attribute filters and search."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM complaints WHERE 1=1"
    params: List[Any] = []

    if category and category != "All":
        query += " AND category_ml = ?"
        params.append(category)
    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
    if urgency and urgency != "All":
        query += " AND urgency_level_rule = ?"
        params.append(urgency)
    if language and language != "All":
        query += " AND language = ?"
        params.append(language)
    if search_query:
        query += " AND (ticket_id LIKE ? OR citizen_name LIKE ? OR raw_text LIKE ? OR summary LIKE ?)"
        like_param = f"%{search_query}%"
        params.extend([like_param, like_param, like_param, like_param])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        item = dict(row)
        if item.get("category_top_probabilities"):
            try:
                item["category_top_probabilities"] = json.loads(item["category_top_probabilities"])
            except Exception:
                pass
        if item.get("urgency_triggers"):
            try:
                item["urgency_triggers"] = json.loads(item["urgency_triggers"])
            except Exception:
                pass
        result.append(item)
    return result

def update_complaint(ticket_id: str, updates: Dict[str, Any], actor: str = "Admin") -> bool:
    """Updates status, department assignment, or notes of a ticket and records an audit log."""
    conn = get_db_connection()
    cursor = conn.cursor()

    valid_fields = [
        "status", "assigned_department", "admin_notes",
        "verified_category", "verified_by", "verified_at", "needs_manual_review"
    ]
    set_clauses = []
    params = []

    for k, v in updates.items():
        if k in valid_fields:
            set_clauses.append(f"{k} = ?")
            params.append(v)

    if not set_clauses:
        conn.close()
        return False

    set_clauses.append("updated_at = ?")
    now_iso = datetime.now().isoformat()
    params.append(now_iso)

    params.append(ticket_id)
    cursor.execute(f"UPDATE complaints SET {', '.join(set_clauses)} WHERE ticket_id = ?", params)
    rows_affected = cursor.rowcount

    if rows_affected > 0:
        # Log to timeline
        details = []
        if "status" in updates:
            details.append(f"Status -> {updates['status']}")
        if "assigned_department" in updates:
            details.append(f"Dept -> {updates['assigned_department']}")
        if "admin_notes" in updates:
            details.append(f"Notes added: '{updates['admin_notes']}'")
        if "verified_category" in updates:
            details.append(f"Category verified -> '{updates['verified_category']}'")
        if "needs_manual_review" in updates:
            rev_label = "Flagged for manual review" if updates["needs_manual_review"] else "Manual review cleared"
            details.append(rev_label)

        cursor.execute("""
        INSERT INTO ticket_timeline (ticket_id, action, performed_by, details, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (
            ticket_id,
            "Ticket Updated",
            actor,
            "; ".join(details) if details else "Ticket modified",
            now_iso
        ))
        conn.commit()

    conn.close()
    return rows_affected > 0

def get_ticket_timeline(ticket_id: str) -> List[Dict[str, Any]]:
    """Returns timeline event entries for a ticket (case-insensitive & trimmed)."""
    clean_id = (ticket_id or "").strip()
    if not clean_id:
        return []
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ticket_timeline WHERE UPPER(ticket_id) = UPPER(?) ORDER BY id ASC", (clean_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_analytics_summary() -> Dict[str, Any]:
    """Computes comprehensive aggregated metrics and distributions for Plotly analytics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total counts
    cursor.execute("SELECT COUNT(*) as total FROM complaints;")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT status, COUNT(*) as cnt FROM complaints GROUP BY status;")
    status_dist = {r["status"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT category_ml, COUNT(*) as cnt FROM complaints GROUP BY category_ml ORDER BY cnt DESC;")
    category_dist = {r["category_ml"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT urgency_level_rule, COUNT(*) as cnt FROM complaints GROUP BY urgency_level_rule;")
    urgency_dist = {r["urgency_level_rule"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT language, COUNT(*) as cnt FROM complaints GROUP BY language;")
    lang_dist = {r["language"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT sentiment_polarity, COUNT(*) as cnt FROM complaints GROUP BY sentiment_polarity;")
    sentiment_dist = {r["sentiment_polarity"]: r["cnt"] for r in cursor.fetchall()}

    cursor.execute("SELECT AVG(urgency_score) as avg_urgency, AVG(category_confidence) as avg_conf FROM complaints;")
    avg_row = cursor.fetchone()

    conn.close()

    return {
        "total_complaints": total,
        "status_distribution": status_dist,
        "category_distribution": category_dist,
        "urgency_distribution": urgency_dist,
        "language_distribution": lang_dist,
        "sentiment_distribution": sentiment_dist,
        "avg_urgency_score": round(avg_row["avg_urgency"] or 0.0, 2),
        "avg_ml_confidence": round(avg_row["avg_conf"] or 0.0, 2)
    }

# ---------------------------------------------------------------------
# User Management & Authentication Queries
# ---------------------------------------------------------------------
def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user record by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a user record by primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def create_user(
    username: str,
    email: Optional[str],
    full_name: Optional[str],
    hashed_password: str,
    role: str = "citizen"
) -> Optional[Dict[str, Any]]:
    """Creates a new user record in the SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    try:
        cursor.execute("""
        INSERT INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        """, (username.strip(), email.strip() if email else None, full_name.strip() if full_name else None, hashed_password, role.strip().lower(), now_iso))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError as e:
        conn.close()
        logger.warning("Failed to create user '%s': %s", username, e)
        return None

def list_users() -> List[Dict[str, Any]]:
    """Lists all users without returning their password hashes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, full_name, role, is_active, created_at FROM users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------
# Server-Side JWT Token Revocation (Blacklist) Queries
# ---------------------------------------------------------------------
def revoke_token(jti: str, expires_at: int) -> bool:
    """Records a JWT identifier (jti) in the revoked_tokens blacklist."""
    if not jti:
        return False
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    try:
        cursor.execute("""
        INSERT OR REPLACE INTO revoked_tokens (jti, revoked_at, expires_at)
        VALUES (?, ?, ?)
        """, (jti, now_iso, int(expires_at)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        logger.error("Failed to revoke token jti '%s': %s", jti, e)
        return False

def is_token_revoked(jti: str) -> bool:
    """Checks whether a JWT identifier (jti) has been recorded as revoked."""
    if not jti:
        return False
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def cleanup_expired_revoked_tokens() -> int:
    """Prunes expired tokens from the blacklist table to conserve space."""
    import time
    now_ts = int(time.time())
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM revoked_tokens WHERE expires_at < ?", (now_ts,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted

# Ensure database and tables exist upon module import
try:
    init_db(seed_sample_data=True)
except Exception as _e:
    logger.debug("Database auto-init deferred: %s", _e)
