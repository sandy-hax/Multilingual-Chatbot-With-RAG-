"""
db.py
====
Database Manager supporting PostgreSQL with local SQLite fallback.
Handles citizen grievance storage, tracking, and PACS resolution workflow.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "grievances.db")


def get_db_connection():
    """Return a database connection (PostgreSQL if DATABASE_URL is set, else SQLite)."""
    if DATABASE_URL:
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(DATABASE_URL)
            return conn, "postgres"
        except Exception as e:
            print(f"⚠️ PostgreSQL connection failed ({e}). Falling back to local SQLite.")

    # SQLite fallback
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


def init_db():
    """Initialize database tables for grievances."""
    conn, db_type = get_db_connection()
    cursor = conn.cursor()

    if db_type == "postgres":
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS grievances (
            id SERIAL PRIMARY KEY,
            tracking_id VARCHAR(50) UNIQUE NOT NULL,
            citizen_name VARCHAR(255) NOT NULL,
            phone_number VARCHAR(20) NOT NULL,
            department VARCHAR(255) NOT NULL,
            district VARCHAR(255) NOT NULL,
            pacs_centre VARCHAR(255) NOT NULL,
            issue_category VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            desired_resolution TEXT,
            status VARCHAR(50) DEFAULT 'Pending',
            pacs_remarks TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    else:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS grievances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id TEXT UNIQUE NOT NULL,
            citizen_name TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            department TEXT NOT NULL,
            district TEXT NOT NULL,
            pacs_centre TEXT NOT NULL,
            issue_category TEXT NOT NULL,
            description TEXT NOT NULL,
            desired_resolution TEXT,
            status TEXT DEFAULT 'Pending',
            pacs_remarks TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT
        );
        """

    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()
    print(f"✅ Grievance database initialized ({db_type}).")


class DatabaseService:
    def __init__(self):
        init_db()

    def create_grievance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new grievance ticket into DB."""
        conn, db_type = get_db_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        tracking_id = data.get("tracking_id") or f"GRV-{datetime.now().year}-{os.urandom(2).hex().upper()}"

        sql = """
        INSERT INTO grievances 
        (tracking_id, citizen_name, phone_number, department, district, pacs_centre, issue_category, description, desired_resolution, status, pacs_remarks, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """ if db_type == "postgres" else """
        INSERT INTO grievances 
        (tracking_id, citizen_name, phone_number, department, district, pacs_centre, issue_category, description, desired_resolution, status, pacs_remarks, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            tracking_id,
            data.get("citizen_name", "Anonymous"),
            data.get("phone_number", "N/A"),
            data.get("department", "PACS Cooperative Society"),
            data.get("district", "General"),
            data.get("pacs_centre", "Central PACS"),
            data.get("issue_category", "General Query"),
            data.get("description", ""),
            data.get("desired_resolution", ""),
            "Pending",
            "",
            now,
            now
        )

        cursor.execute(sql, params)
        conn.commit()
        conn.close()

        res_data = dict(data)
        res_data["tracking_id"] = tracking_id
        res_data["status"] = "Pending"
        res_data["created_at"] = now
        print(f"📌 Grievance recorded in DB: {tracking_id}")
        return res_data

    def get_grievances(self, pacs_centre: str = None, department: str = None, status: str = None) -> List[Dict[str, Any]]:
        """Retrieve grievances with optional filtering."""
        conn, db_type = get_db_connection()
        
        query = "SELECT * FROM grievances WHERE 1=1"
        params = []

        if pacs_centre and pacs_centre != "all":
            query += " AND pacs_centre = %s" if db_type == "postgres" else " AND pacs_centre = ?"
            params.append(pacs_centre)

        if department and department != "all":
            query += " AND department = %s" if db_type == "postgres" else " AND department = ?"
            params.append(department)

        if status and status != "all":
            query += " AND status = %s" if db_type == "postgres" else " AND status = ?"
            params.append(status)

        query += " ORDER BY id DESC"

        if db_type == "postgres":
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
        else:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]

        conn.close()
        return result

    def get_grievance_by_tracking_id(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single grievance detail by tracking ID."""
        conn, db_type = get_db_connection()
        query = "SELECT * FROM grievances WHERE tracking_id = %s" if db_type == "postgres" else "SELECT * FROM grievances WHERE tracking_id = ?"
        
        cursor = conn.cursor()
        cursor.execute(query, (tracking_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def update_grievance_status(self, tracking_id: str, status: str, pacs_remarks: str = "") -> bool:
        """Update grievance status and PACS operator resolution remarks."""
        conn, db_type = get_db_connection()
        now = datetime.now().isoformat()

        query = """
        UPDATE grievances 
        SET status = %s, pacs_remarks = %s, updated_at = %s 
        WHERE tracking_id = %s
        """ if db_type == "postgres" else """
        UPDATE grievances 
        SET status = ?, pacs_remarks = ?, updated_at = ? 
        WHERE tracking_id = ?
        """

        cursor = conn.cursor()
        cursor.execute(query, (status, pacs_remarks, now, tracking_id))
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

    def get_pacs_stats(self) -> Dict[str, int]:
        """Return analytics counts for PACS Operator Dashboard."""
        conn, db_type = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM grievances")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM grievances WHERE status = 'Pending'")
        pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM grievances WHERE status = 'In Progress'")
        in_progress = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM grievances WHERE status = 'Resolved'")
        resolved = cursor.fetchone()[0]

        conn.close()
        return {
            "total": total,
            "pending": pending,
            "in_progress": in_progress,
            "resolved": resolved
        }


db_service = DatabaseService()
