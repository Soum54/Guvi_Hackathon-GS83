// frontend/app.js
const form = document.getElementById("diagForm");
const resultsEl = document.getElementById("results");

// very small phrasebook for the 4 Indian languages to English mapping (illustrative)
const phrasebook = {
  hi: {
    "bukhar": "fever",
    "khansi": "cough",
    "saans lene mein takleef": "shortness of breath",
    "daast": "diarrhea"
  },
  ta: {
    "பரிதாபம்": "fever"
  },
  te: {},
  bn: {
    "জ্বর": "fever",
    "কাশি": "cough"
  }
};

function translateIfNeeded(text, lang) {
  if (!text) return text;
  if (!phrasebook[lang]) return text;
  let t = text;
  for (let [k, v] of Object.entries(phrasebook[lang])) {
    const re = new RegExp(k, "gi");
    t = t.replace(re, v);
  }
  return t;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultsEl.innerHTML = "<div class='small'>Processing — doing offline inference...</div>";
  const patient = {
    name: document.getElementById("name").value || null,
    age: parseInt(document.getElementById("age").value || "0"),
    gender: document.getElementById("gender").value,
    weight: null,
    phone: null,
    language: document.getElementById("language").value
  };
  let symptomsRaw = document.getElementById("symptoms").value || "";
  const lang = patient.language;
  // brief phrasebook translation
  const symptoms = translateIfNeeded(symptomsRaw, lang);

  const payload = {
    patient,
    demographics: {
      age: patient.age,
      weight: null,
      current_medications: (document.getElementById("meds").value || "").split(",").map(s => s.trim()).filter(s => s)
    },
    vitals: {
      pulse: parseInt(document.getElementById("pulse").value || "0"),
      systolic: parseInt(document.getElementById("systolic").value || "0"),
      spo2: parseInt(document.getElementById("spo2").value || "0")
    },
    symptoms: symptoms
  };

  try {
    const res = await fetch("/diagnose", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data.diagnoses);
  } catch (err) {
    resultsEl.innerHTML = `<div class="card">Error: ${err.message}</div>`;
  }
});

function renderResults(diags) {
  if (!diags || diags.length==0) {
    resultsEl.innerHTML = "<div class='card'>No likely diagnoses found. Consider referral.</div>";
    return;
  }
  resultsEl.innerHTML = "";
  diags.forEach(d => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <strong>${d.name}</strong> <span class="small"> (confidence ${Math.round(d.confidence*100)}%)</span>
      <div class="small"><em>Symptoms:</em> ${d.common_symptoms.join(", ")}</div>
      <div class="small"><em>Immediate:</em> ${d.immediate_protocol}</div>
      <div class="small"><em>Dosage hint:</em> ${d.dosage}</div>
      ${d.drug_interactions.length ? `<div class="small" style="color:#b91c1c;"><strong>Possible interaction/duplication:</strong> ${d.drug_interactions.join("; ")}</div>` : ""}
      <div class="small"><strong>Referral suggested:</strong> ${d.refer ? "Yes — consider refer" : "No (manage at PHC if resources available)"}</div>
    `;
    resultsEl.appendChild(card);
  });
}
