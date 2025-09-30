# backend/app.py
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from model_utils import MedicalKB
from init_db import init_db, DB_PATH
import sqlite3
from pathlib import Path

app = Flask(__name__, static_folder="../frontend", static_url_path="/")

# ensure DB exists
Path("backend").mkdir(parents=True, exist_ok=True)
if not DB_PATH.exists():
    init_db()

# initialize model KB (load model - may take ~several seconds)
print("Loading local BioBERT model (local_files_only=True must have files downloaded)...")
kb = MedicalKB(kb_path="backend/diseases.json")
print("Model and KB loaded.")

def save_diagnosis_to_db(patient_info, input_text, vitals, demographics, results):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # find or create patient record (simple matching by phone if available)
    phone = patient_info.get("phone")
    patient_id = None
    if phone:
        c.execute("SELECT patient_id FROM patients WHERE phone = ?", (phone,))
        row = c.fetchone()
        if row:
            patient_id = row[0]
    if not patient_id:
        c.execute("INSERT INTO patients (name, age, gender, weight, phone, language) VALUES (?, ?, ?, ?, ?, ?)",
                  (patient_info.get("name"), patient_info.get("age"), patient_info.get("gender"),
                   patient_info.get("weight"), phone, patient_info.get("language")))
        patient_id = c.lastrowid
    c.execute("INSERT INTO diagnoses (patient_id, input_text, vitals_json, demographics_json, results_json) VALUES (?, ?, ?, ?, ?)",
              (patient_id, input_text, json.dumps(vitals), json.dumps(demographics), json.dumps(results)))
    conn.commit()
    conn.close()

@app.route("/diagnose", methods=["POST"])
def diagnose():
    payload = request.json or {}
    symptom_text = payload.get("symptoms", "")
    vitals = payload.get("vitals", {})
    demographics = payload.get("demographics", {})
    patient_info = payload.get("patient", {})
    if not symptom_text:
        return jsonify({"error": "Provide 'symptoms' text"}), 400
    try:
        results = kb.diagnose(symptom_text, vitals=vitals, demographics=demographics, top_k=6)
        # Save for knowledge base improvement
        save_diagnosis_to_db(patient_info, symptom_text, vitals, demographics, results)
        return jsonify({"status": "ok", "diagnoses": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/diseases", methods=["GET"])
def list_diseases():
    return jsonify(kb.diseases)

# serve frontend
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

