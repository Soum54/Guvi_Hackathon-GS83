# backend/init_db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("backend/patients.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        weight REAL,
        phone TEXT,
        language TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS diagnoses (
        diag_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        input_text TEXT,
        vitals_json TEXT,
        demographics_json TEXT,
        results_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """)
    conn.commit()
    conn.close()
    print("DB initialized at", DB_PATH)

if __name__ == "__main__":
    init_db()
