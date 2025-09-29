# backend/model_utils.py
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from pathlib import Path

MODEL_NAME = os.environ.get("BIOBERT_MODEL", "dmis-lab/biobert-base-cased-v1.1")
# NOTE: model should be downloaded during initial setup for offline use.

class MedicalKB:
    def __init__(self, kb_path="backend/diseases.json", device=None):
        self.kb_path = kb_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(self.device)
        self.diseases = self._load_kb()
        # Precompute embeddings for disease templates
        self._precompute_disease_embeddings()

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _encode_text(self, text: str) -> np.ndarray:
        # returns mean-pooled last hidden states (cpu numpy)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=False)
            last = out.last_hidden_state  # (1, seq_len, hidden)
            attn_mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (last * attn_mask).sum(1)
            counts = attn_mask.sum(1).clamp(min=1)
            mean_pooled = summed / counts
            vec = mean_pooled.cpu().numpy()[0]
        return vec

    def _precompute_disease_embeddings(self):
        for d in self.diseases:
            templates = d.get("symptom_templates", [])
            vectors = []
            for t in templates:
                try:
                    v = self._encode_text(t)
                    vectors.append(v)
                except Exception as e:
                    vectors.append(np.zeros(self.model.config.hidden_size))
            if vectors:
                d["template_vecs"] = np.vstack(vectors)  # (k, hidden)
            else:
                d["template_vecs"] = np.zeros((1, self.model.config.hidden_size))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray):
        # a: (hidden,), b: (k, hidden) or (hidden,)
        if b.ndim == 1:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)
        # else compute max similarity across templates
        denom = (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=1)) + 1e-8
        # compute pairwise
        sims = (b @ a) / denom
        return float(np.max(sims))

    def diagnose(self, symptom_text: str, vitals: Dict = None, demographics: Dict = None, top_k=5):
        """
        Returns differential diagnoses with confidence scores (0-1),
        suggested protocol snippets, referral flag, and basic dosage hints.
        """
        if vitals is None:
            vitals = {}
        if demographics is None:
            demographics = {}
        qvec = self._encode_text(symptom_text)
        scores = []
        for d in self.diseases:
            tv = d.get("template_vecs")
            sim = self._cosine(qvec, tv)
            # heuristics: boost scores if vitals suggest severity matches disease
            score = sim
            # severity heuristics (simple)
            sev = d.get("severity", "low")
            if sev == "high":
                # if tachycardia / hypotension mention in vitals boost small
                hr = vitals.get("pulse")
                bp = vitals.get("systolic")
                if hr and hr > 120:
                    score += 0.05
                if bp and bp < 90:
                    score += 0.05
            scores.append((d, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        results = []
        for d, s in top:
            conf = min(max((s + 1) / 2, 0.0), 1.0)  # map cosine[-1,1] to [0,1]
            results.append({
                "disease_id": d["id"],
                "name": d["name"],
                "confidence": round(conf, 3),
                "common_symptoms": d.get("common_symptoms", []),
                "immediate_protocol": d.get("immediate_protocol", "Refer to local STG."),
                "refer": d.get("refer", False),
                "dosage": self._estimate_dosage(d, demographics),
                "drug_interactions": self._check_interactions(d, demographics.get("current_medications", []))
            })
        return results

    def _estimate_dosage(self, disease_entry, demographics):
        # Simple rule-based dosage suggestions (illustrative)
        age = demographics.get("age")
        weight = demographics.get("weight")  # in kg
        dos = disease_entry.get("dosage_guidance", {})
        if not dos:
            return "See STG for dosing."
        # if weight-based dosing exists:
        if "mg_per_kg" in dos and weight:
            mg = dos["mg_per_kg"] * weight
            return f"Approx {mg:.0f} mg total (weight-based {dos['mg_per_kg']} mg/kg)."
        # age-based (children/adult)
        if age and age < 12 and "pediatric" in dos:
            return dos["pediatric"]
        return dos.get("adult", "See STG for dosing.")

    def _check_interactions(self, disease_entry, current_meds: List[str]):
        # Very simple interaction alert demo based on disease_entry's drugs
        interactions = []
        disease_drugs = set([d.lower() for d in disease_entry.get("drugs", [])])
        for m in current_meds:
            if m.lower() in disease_drugs:
                interactions.append(f"Patient already on {m} — check duplication.")
        # Placeholder for more complex interaction checks
        return interactions
