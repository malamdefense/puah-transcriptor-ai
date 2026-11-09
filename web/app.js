"use strict";

const API = "/api";
const SPEAKER_COLORS = [
  "#2f7df6", "#2fb673", "#e6a23c", "#e35d6a", "#a06af6",
  "#19b3c9", "#d96aae", "#7fb800", "#ff7849", "#5c7cfa",
];

const state = {
  selectedFile: null,
  currentJobId: null,
  result: null,
  jobs: [],
  pollTimer: null,
  findQuery: "",
  findMarks: [],
  findIndex: -1,
  searchTimer: null,
  view: "original",   // active transcript tab: "original" | "pii"
  piiJobId: null,     // recording whose de-identified copy is cached below
  piiResult: null,
  piiError: null,
  piiLoading: false,
  lowpHint: false,    // opt-in "unvalidated" low-confidence word marks (off by default)
};

const $ = (id) => document.getElementById(id);

// ---------- Helper functions ----------
function fmtClock(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}
function fmtDuration(sec) {
  if (!sec) return "";
  return fmtClock(sec);
}
const STATUS_HE = { queued: "בתור", processing: "מעבד", done: "הושלם", error: "שגיאה" };
const STAGE_HE = {
  extracting_audio: "חילוץ אודיו",
  transcribing: "מתמלל",
  diarizing: "מזהה דוברים",
  merging: "מסדר תמלול",
  summarizing: "מפיק סיכומים",
};
function speakerColor(id, speakerIds) {
  const idx = speakerIds.indexOf(id);
  return SPEAKER_COLORS[(idx >= 0 ? idx : 0) % SPEAKER_COLORS.length];
}
const LANG_HE = { he: "עברית", en: "אנגלית" };
// Direction of a transcript line: from the detected segment language when there is one,
// otherwise the browser picks it from the first strong character (old results, search hits).
function textDir(lang) {
  return lang === "en" ? "ltr" : lang === "he" ? "rtl" : "auto";
}
// Call language for the header — only when a detected call is mixed or not Hebrew;
// plain Hebrew calls and old results (no `languages`) show nothing.
function languageLabel(res) {
  const langs = Array.isArray(res.languages) ? res.languages : [];
  if (!langs.length || (langs.length === 1 && (res.language || langs[0]) === "he")) return "";
  return "שפה: " + langs.map((l) => LANG_HE[l] || l).join(" + ");
}

// ---------- Health check ----------
// The system version in the header, after the title (from /api/health = app/__init__.py __version__).
function showVersion(v) {
  const el = $("appVersion");
  if (!el) return;
  el.textContent = v ? `גרסה ${v}` : "";
}

async function checkHealth() {
  const el = $("health");
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error();
    const h = await r.json();
    showVersion(h && h.version);
    // Connected — no need to show anything
    el.className = "health hidden";
    el.textContent = "";
  } catch {
    // Only when disconnected — show a warning
    el.textContent = "⚠ אין חיבור לשרת";
    el.className = "health bad";
  }
}

// ---------- File selection ----------
function setupUpload() {
  const dz = $("dropzone");
  const input = $("fileInput");
  $("browseBtn").addEventListener("click", () => input.click());
  dz.addEventListener("click", (e) => {
    if (e.target.id !== "browseBtn") input.click();
  });
  input.addEventListener("change", () => input.files[0] && selectFile(input.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); })
  );
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) selectFile(f);
  });

  $("uploadBtn").addEventListener("click", startUpload);
}

function selectFile(file) {
  state.selectedFile = file;
  $("selectedName").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)}MB`;
  $("selectedFile").classList.remove("hidden");
  $("uploadStatus").textContent = "";
  $("uploadStatus").className = "upload-status";
}

function startUpload() {
  if (!state.selectedFile) return;
  const fd = new FormData();
  fd.append("file", state.selectedFile);
  const ns = $("numSpeakers").value;
  if (ns) fd.append("num_speakers", ns);
  fd.append("language", "he");

  const btn = $("uploadBtn");
  const status = $("uploadStatus");
  btn.disabled = true;

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${API}/jobs`);
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      status.textContent = `מעלה… ${pct}%`;
      status.className = "upload-status";
    }
  };
  xhr.onload = () => {
    btn.disabled = false;
    if (xhr.status === 201) {
      status.textContent = "הקובץ הועלה ונכנס לתור ✓";
      status.className = "upload-status ok";
      state.selectedFile = null;
      $("selectedFile").classList.add("hidden");
      $("fileInput").value = "";
      refreshJobs();
    } else {
      let msg = "ההעלאה נכשלה";
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
      status.textContent = msg;
      status.className = "upload-status err";
    }
  };
  xhr.onerror = () => {
    btn.disabled = false;
    status.textContent = "שגיאת רשת בהעלאה";
    status.className = "upload-status err";
  };
  xhr.send(fd);
}

// ---------- Job list ----------
async function refreshJobs() {
  try {
    const r = await fetch(`${API}/jobs`);
    state.jobs = await r.json();
    renderJobs();
  } catch {
    /* Ignore — will try again in the next poll */
  }
}

function renderJobs() {
  const list = $("jobsList");
  if (!state.jobs.length) {
    list.innerHTML = `<li style="color:var(--text-dim);font-size:13px;padding:8px">אין הקלטות עדיין.</li>`;
    return;
  }
  list.innerHTML = "";
  for (const j of state.jobs) {
    const li = document.createElement("li");
    li.className = "job" + (j.id === state.currentJobId ? " active" : "");
    li.onclick = () => selectJob(j.id);

    const stageTxt = j.status === "processing" && j.stage ? STAGE_HE[j.stage] || "" : "";
    const right = j.status === "done" && j.num_speakers
      ? `${j.num_speakers} דוברים`
      : (j.status === "error" ? "נכשל" : stageTxt);

    li.innerHTML = `
      <div class="job-top">
        <span class="job-name" title="${escapeHtml(j.original_name)}">${escapeHtml(j.original_name)}</span>
        <span class="badge ${j.status}">${STATUS_HE[j.status] || j.status}</span>
      </div>
      <div class="job-meta">
        <span>${right || ""}</span>
        <span>${fmtDuration(j.duration_sec)}</span>
      </div>
      ${j.status === "processing"
        ? `<div class="progress"><i style="width:${j.progress || 0}%"></i></div>` : ""}
    `;

    const del = document.createElement("button");
    del.className = "job-del";
    del.textContent = "🗑";
    del.title = "מחיקה";
    del.onclick = (e) => { e.stopPropagation(); deleteJob(j.id); };
    li.querySelector(".job-top").appendChild(del);

    list.appendChild(li);
  }
}

async function deleteJob(id) {
  if (!confirm("למחוק את ההקלטה והתמלול לצמיתות?")) return;
  await fetch(`${API}/jobs/${id}`, { method: "DELETE" });
  if (state.currentJobId === id) {
    state.currentJobId = null;
    state.result = null;
    showEmpty();
  }
  refreshJobs();
}

// ---------- Job selection and transcription display ----------
async function selectJob(id) {
  hideDashboard();
  hideSettings();
  state.currentJobId = id;
  state.result = null; // Clear previous result to allow automatic loading on the next poll
  resetPiiView();      // every recording opens on the full transcript
  state.lowpHint = false; // ... with the unvalidated low-confidence hint off
  renderJobs();
  const job = state.jobs.find((j) => j.id === id);
  if (!job) return;

  if (job.status === "done") {
    await loadResult(id);
  } else if (job.status === "error") {
    showError(job);
  } else {
    showProcessing(job);
  }
}

function showEmpty() {
  $("emptyState").classList.remove("hidden");
  $("transcriptView").classList.add("hidden");
}

function showProcessing(job) {
  $("emptyState").classList.remove("hidden");
  $("transcriptView").classList.add("hidden");
  $("emptyState").innerHTML = `
    <div class="empty-icon">⏳</div>
    <h2>${escapeHtml(job.original_name)}</h2>
    <p>${STAGE_HE[job.stage] || "ממתין בתור"}… ${Math.round(job.progress || 0)}%</p>`;
}

function showError(job) {
  $("emptyState").classList.remove("hidden");
  $("transcriptView").classList.add("hidden");
  $("emptyState").innerHTML = `
    <div class="empty-icon">⚠️</div>
    <h2>התמלול נכשל</h2>
    <p style="max-width:560px;margin:0 auto;color:var(--err)">${escapeHtml(job.error || "שגיאה לא ידועה")}</p>`;
}

async function loadResult(id) {
  const r = await fetch(`${API}/jobs/${id}/result`);
  if (!r.ok) return;
  state.result = await r.json();
  renderTranscript();
}

function renderTranscript() {
  const res = state.result;
  $("emptyState").classList.add("hidden");
  $("transcriptView").classList.remove("hidden");

  const job = state.jobs.find((j) => j.id === res.job_id);
  $("tTitle").textContent = job ? job.original_name : "תמלול";
  const lang = languageLabel(res);
  $("tMeta").textContent =
    `${res.num_speakers} דוברים · משך ${fmtDuration(res.duration_sec)} · ${res.segments.length} קטעים` +
    (lang ? ` · ${lang}` : "");

  // Export links
  $("expWord").href = `${API}/jobs/${res.job_id}/export?format=docx`;
  $("expTxt").href = `${API}/jobs/${res.job_id}/export?format=txt`;
  $("expSrt").href = `${API}/jobs/${res.job_id}/export?format=srt`;
  $("expVtt").href = `${API}/jobs/${res.job_id}/export?format=vtt`;
  $("expJson").href = `${API}/jobs/${res.job_id}/export?format=json`;
  $("expPiiWord").href = `${API}/jobs/${res.job_id}/pii?format=docx`;
  $("expPiiJson").href = `${API}/jobs/${res.job_id}/pii?format=json`;

  // PUAH products: client summary + consultant feedback.
  // The structured "automatic summary" is still generated server-side (it feeds the
  // dashboard keywords/topics) but is intentionally not shown in the transcript view.
  renderPuahOutputs(res);

  // Player
  const player = $("player");
  player.src = `${API}/jobs/${res.job_id}/media`;

  const speakerIds = Object.keys(res.speakers);

  // Speaker legend with name editing
  const legend = $("speakerLegend");
  legend.innerHTML = "";
  for (const sid of speakerIds) {
    const chip = document.createElement("div");
    chip.className = "chip";
    const color = speakerColor(sid, speakerIds);
    chip.innerHTML = `<span class="dot" style="background:${color}"></span>`;
    const inp = document.createElement("input");
    inp.value = res.speakers[sid];
    inp.dataset.sid = sid;
    inp.addEventListener("change", () => saveSpeakerNames());
    chip.appendChild(inp);
    legend.appendChild(chip);
  }

  // Recording-quality check — from the live result, so it names speakers as the legend does
  renderAudioQuality(res);

  // Reset internal search and build segments (keeping the active tab)
  $("findInput").value = "";
  state.findQuery = "";
  updateTabs();
  if (state.view === "pii") void switchView("pii");
  else { renderPiiNotice(); renderSegments(""); }

  // Highlight the currently playing segment
  player.ontimeupdate = () => highlightSegment(player.currentTime);
}

// Build segments, with optional internal search highlighting.
// Renders either the full transcript or the de-identified (PII) copy, per the active tab.
function renderSegments(query) {
  const pii = state.view === "pii";
  const res = pii ? state.piiResult : state.result;
  const segWrap = $("segments");

  if (!res) {
    // Nothing to mark: the hint toggle goes too (PII copy loading, missing or failed)
    updateLowpToggle(false);
    // PII tab with nothing loaded yet (or unavailable) — #piiNotice above carries the explanation
    if (pii) {
      segWrap.innerHTML = state.piiLoading ? `<div class="dash-empty">טוען…</div>` : "";
      updateFindState();
    }
    return;
  }

  // Speaker names and colours always come from the live transcript, so both tabs stay in
  // sync with the editable legend above (the PII copy may hold names from before a rename).
  const names = (state.result && state.result.speakers) || res.speakers;
  const speakerIds = Object.keys(names);
  segWrap.innerHTML = "";

  // Confidence marks exist only in the full transcript (the PII copy has no words):
  // window flags always, the low-confidence word hint only when the viewer opts in.
  const threshold = pii ? null : lowpThreshold(res);
  updateLowpToggle(!pii && threshold !== null);
  const hintAt = state.lowpHint ? threshold : null;

  res.segments.forEach((seg, i) => {
    const div = document.createElement("div");
    div.className = "seg";
    const color = speakerColor(seg.speaker, speakerIds);
    div.style.borderRightColor = color;
    div.dataset.start = seg.start;
    div.dataset.end = seg.end;
    div.dataset.idx = i;
    const check = !pii && segWindowFlag(seg);
    div.innerHTML = `
      <div class="seg-head">
        <span class="seg-speaker" style="color:${color}">${escapeHtml(names[seg.speaker] || seg.speaker)}</span>
        <span class="seg-time">${fmtClock(seg.start)} – ${fmtClock(seg.end)}</span>
        ${check ? `<span class="seg-check" title="${escapeHtml(WINDOW_FLAG_TITLE[check] || WINDOW_FLAG_TITLE.gate_failed)}">לבדוק מול ההקלטה</span>` : ""}
      </div>
      <div class="seg-text" dir="${textDir(seg.language)}">${pii ? piiHtml(seg.text, query) : segmentHtml(seg, query, hintAt)}</div>`;
    div.onclick = () => { const p = $("player"); p.currentTime = seg.start; p.play(); };
    segWrap.appendChild(div);
  });

  updateFindState();
}

// ---------- Confidence marks (ASR window flags; opt-in low-confidence hint) ----------
// A word's decode window that failed a quality gate, or was decoded by random sampling, is
// marked "check against the recording". Word probabilities are NOT validated against a manual
// transcript yet, so their marks are an opt-in hint, labelled as such.
const WINDOW_FLAG_TITLE = {
  gate_failed: "המודל התקשה בקטע זה (פענוח שלא עבר את בדיקת האיכות) — כדאי לבדוק מול ההקלטה",
  sampled_hi: "קטע זה פוענח בדגימה אקראית לאחר ניסיונות שנכשלו — כדאי לבדוק מול ההקלטה",
};
const LOWP_LABEL = "רמז: מילים בביטחון נמוך (לא מאומת)";
const LOWP_TITLE =
  "סימון לפי הסתברות המודל בלבד, שטרם נבדקה מול תמלול ידני: " +
  "מילה מסומנת אינה בהכרח שגויה, ומילה שאינה מסומנת אינה בהכרח נכונה.";

// The first window flag among the segment's words, or null
function segWindowFlag(seg) {
  for (const w of seg.words || []) if (w && w.window_flag) return String(w.window_flag);
  return null;
}

// The display threshold, when the result says how confidence is shown and carries word probabilities
function lowpThreshold(res) {
  const cd = res && res.confidence_display;
  if (!cd || typeof cd.threshold !== "number") return null;
  const any = (res.segments || []).some((s) => (s.words || []).some((w) => w && typeof w.probability === "number"));
  return any ? cd.threshold : null;
}

// The toggle lives in the find bar; created on first use, hidden when it does not apply (PII tab,
// results without probabilities)
function updateLowpToggle(show) {
  let wrap = $("lowpToggleWrap");
  if (!wrap) {
    if (!show) return; // never created just to be hidden
    const bar = document.querySelector(".findbar");
    if (!bar) return;
    wrap = document.createElement("label");
    wrap.id = "lowpToggleWrap";
    wrap.className = "lowp-toggle";
    wrap.title = LOWP_TITLE;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.id = "lowpToggle";
    box.addEventListener("change", (e) => {
      state.lowpHint = e.target.checked;
      renderSegments(state.findQuery);
    });
    const text = document.createElement("span");
    text.textContent = LOWP_LABEL;
    wrap.append(box, text);
    bar.appendChild(wrap);
  }
  wrap.classList.toggle("hidden", !show);
  const box = $("lowpToggle");
  if (box) box.checked = !!state.lowpHint;
}

// Segment text with per-word marks. Each word is escaped and find-highlighted BEFORE it is
// wrapped, so the find highlighter never runs over tag text and the <mark> count is unchanged.
// The word view is used only when the words spell the stored text exactly (the merge's rule);
// otherwise the text is shown as stored.
function segmentHtml(seg, query, hintAt) {
  const words = Array.isArray(seg.words) ? seg.words : [];
  const low = (w) => hintAt !== null && typeof w.probability === "number" && w.probability < hintAt;
  if (!words.some((w) => w && (w.window_flag || low(w)))) return highlightHtml(seg.text, query);
  if (words.map((w) => String((w && w.word) ?? "")).join(" ").trim() !== String(seg.text || "").trim()) {
    return highlightHtml(seg.text, query);
  }
  return words.map((w) => {
    const inner = highlightHtml(String(w.word ?? ""), query);
    const cls = [];
    const titles = [];
    if (w.window_flag) {
      cls.push("w-check");
      titles.push(WINDOW_FLAG_TITLE[w.window_flag] || WINDOW_FLAG_TITLE.gate_failed);
    }
    if (low(w)) {
      cls.push("w-lowp");
      titles.push(`ביטחון המודל ${Math.round(w.probability * 100)}% — רמז לא מאומת`);
    }
    if (!cls.length) return inner;
    return `<span class="${cls.join(" ")}" title="${escapeHtml(titles.join(" · "))}">${inner}</span>`;
  }).join(" ");
}

function highlightSegment(t) {
  const segs = document.querySelectorAll(".seg");
  let active = null;
  segs.forEach((s) => {
    const start = parseFloat(s.dataset.start);
    const end = parseFloat(s.dataset.end);
    if (t >= start && t < end) active = s;
    s.classList.remove("playing");
  });
  if (active) active.classList.add("playing");
}

async function saveSpeakerNames() {
  const inputs = document.querySelectorAll("#speakerLegend input");
  const speakers = {};
  inputs.forEach((i) => (speakers[i.dataset.sid] = i.value));
  const r = await fetch(`${API}/jobs/${state.result.job_id}/speakers`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speakers }),
  });
  if (r.ok) {
    state.result = await r.json();
    renderTranscript();
  }
}

// ---------- PUAH products: client summary + consultant feedback ----------
// These are the only two summaries shown in the transcript view. The structured
// "automatic summary" is still produced server-side (it powers the dashboard's
// keywords/topics) but is deliberately not rendered here.
function renderPuahOutputs(res) {
  const el = $("puahSection");
  const cs = res.client_summary;
  const fb = res.feedback;
  const has = cs || fb;

  let html = `<div class="puah-actions"><h3>סיכומי השיחה</h3>
    <button id="regenSummaryBtn" class="btn ghost">${has ? "הפק מחדש" : "הפק סיכומים כעת"}</button></div>`;

  // Produced automatically by the worker right after the transcript PII step;
  // the button above also allows manual recovery if the LLM was busy or unreachable.
  if (!has) {
    html += `<div class="puah-card"><div class="sc-empty">
      הסיכומים מופקים אוטומטית בסיום התמלול לאחר הסרת פרטים מזהים (PII) מקומית.
      אם הם אינם מופיעים — ניתן ללחוץ על <b>הפק סיכומים כעת</b> למעלה, או לבדוק את יומן ה-worker.</div></div>`;
  }
  // Redaction-placeholder check of each text (server record: labels only, never text).
  // residual = tags still in the stored text (results from before the guard); sanitized = the
  // tags were removed automatically after the retries, so a detail may be missing from a
  // sentence. Repaired by a retry, or clean: nothing to show.
  const ck = res.client_summary_check || {};
  const fk = res.feedback_check || {};
  const cr = res.client_summary_reid || {};
  const fr = res.feedback_reid || {};
  const csTagged = Array.isArray(ck.residual) && ck.residual.length > 0;
  const fbTagged = Array.isArray(fk.residual) && fk.residual.length > 0;
  let csNote = "";
  if (csTagged)
    csNote = `<div class="pii-notice err"><b>הסיכום מכיל תגיות הסתרה — אין לשלוח לפונה לפני תיקון</b>
      · ${tagLabelsHtml(ck.residual)}</div>`;
  else if (ck.sanitized)
    csNote = `<div class="pii-notice warn">תגיות הסתרה הוסרו אוטומטית מהסיכום. מומלץ לקרוא אותו לפני השליחה.
      · ${tagLabelsHtml(ck.tags_found)}</div>`;
  const csReid = reidNoticeHtml(cr, true);
  // The feedback is internal: a muted note only.
  let fbNote = "";
  if (fbTagged) fbNote = `<div class="pii-notice">המשוב מכיל תגיות הסתרה · ${tagLabelsHtml(fk.residual)}</div>`;
  else if (fk.sanitized)
    fbNote = `<div class="pii-notice">תגיות הסתרה הוסרו אוטומטית מהמשוב · ${tagLabelsHtml(fk.tags_found)}</div>`;
  const fbReid = reidNoticeHtml(fr, false);

  if (cs)
    html += `<div class="puah-card client">
      <div class="pc-head"><h3>📩 סיכום ללקוח</h3>
        <button class="btn ghost copy-btn" data-copy="cs">העתק</button></div>
      ${csNote}${csReid}<div class="puah-text">${csTagged ? piiHtml(cs, "") : escapeHtml(cs)}</div></div>`;
  if (fb)
    html += `<div class="puah-card feedback">
      <div class="pc-head"><h3>📋 משוב ליועץ (פנימי)</h3>
        <button class="btn ghost copy-btn" data-copy="fb">העתק</button></div>
      ${fbNote}${fbReid}<div class="puah-text">${fbTagged ? piiHtml(fb, "") : escapeHtml(fb)}</div></div>`;

  el.innerHTML = html;
  const regenBtn = $("regenSummaryBtn");
  if (regenBtn && res && res.job_id) {
    regenBtn.addEventListener("click", () => regenerateSummaries(res.job_id));
  }
  el.querySelectorAll(".copy-btn").forEach((b) =>
    b.addEventListener("click", () => {
      const isCs = b.dataset.copy === "cs";
      // The client summary goes straight to the caller: a tag left in it needs a conscious yes.
      if (isCs && csTagged && !confirm("הסיכום מכיל תגיות הסתרה. להעתיק בכל זאת?")) return;
      const txt = isCs ? cs : fb;
      if (navigator.clipboard) navigator.clipboard.writeText(txt);
      b.textContent = "הועתק ✓";
      setTimeout(() => (b.textContent = "העתק"), 1500);
    })
  );
}

async function regenerateSummaries(jobId) {
  const btn = $("regenSummaryBtn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "מפיק סיכומים…";
  }
  try {
    const r = await fetch(`${API}/jobs/${jobId}/summarize`, { method: "POST" });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `שגיאה ${r.status}`);
    }
    const updated = await r.json();
    if (state.currentJobId === jobId) {
      state.result = updated;
      state.piiJobId = null;
      state.piiResult = null;
      renderTranscript();
    }
  } catch (err) {
    alert(err.message || "הפקת הסיכומים נכשלה");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "הפק סיכומים כעת";
    }
  }
}

// Placeholder labels of a check record as tag chips (the labels themselves contain < >).
function tagLabelsHtml(labels) {
  return (Array.isArray(labels) ? labels : [])
    .map((l) => `<span class="pii-tag">${escapeHtml(l)}</span>`)
    .join(" ");
}

// Re-identification notice: counts of restored labels only (no values).
function reidNoticeHtml(reid, forCaller) {
  const restored = (reid && typeof reid.restored === "object" && reid.restored) || {};
  const entries = Object.entries(restored).filter(([, n]) => Number(n) > 0);
  if (!entries.length) return "";
  const chips = entries
    .map(([label, n]) => `<span class="pii-tag">${escapeHtml(label)} ×${Number(n)}</span>`)
    .join(" ");
  const tail = forCaller ? " — מומלץ לוודא דיוק לפני שליחה לפונה" : "";
  return `<div class="pii-notice">שוחזרו מקומית מהתמלול (ללא חשיפה למודל ה-AI)${tail} · ${chips}</div>`;
}

// ---------- Recording-quality check ----------
// result.audio_quality comes from the worker: numbers and fixed Hebrew messages, never transcript
// text. A message names its speaker as "{speaker}"; it is replaced here by the CURRENT display name
// (the legend can rename), falling back to the speaker id. Old results have no audio_quality and
// show nothing. The box sits above the transcript tabs and is built from state.result only, so the
// PII tab (which swaps just the segments below it) leaves it as it is.
const AQ_TITLE = "בדיקת איכות הקלטה";
const AQ_CLEAN = "לא נמצאו בעיות באיכות ההקלטה.";
const AQ_FOOTNOTE = "הבדיקה אוטומטית; ההקלטה המקורית לא שונתה.";
const AQ_SEVERITY = {
  severe: { rank: 0, label: "חמור" },
  warn: { rank: 1, label: "אזהרה" },
  info: { rank: 2, label: "לידיעה" },
};
const AQ_FIX_LABEL = "טופל אוטומטית";
const AQ_ANY_SPEAKER = "אחד הדוברים";
const AQ_SPEAKER_ID = /^SPEAKER_\d+$/;
const _own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

// Speakers an entry is about: its `speaker` (flags), else the speaker keys of its `detail` (a
// speaker_gain fix lists the lifted speakers there).
function aqSpeakerIds(entry, names) {
  if (entry.speaker) return [String(entry.speaker)];
  const d = entry.detail;
  if (!d || typeof d !== "object" || Array.isArray(d)) return [];
  return Object.keys(d).filter((k) => AQ_SPEAKER_ID.test(k) || _own(names, k));
}

// The message as HTML: every piece escaped, "{speaker}" -> the bidi-isolated display name(s).
function aqMessageHtml(entry, names) {
  const ids = aqSpeakerIds(entry, names);
  const who = ids.length
    ? ids.map((id) => {
        const name = _own(names, id) ? String(names[id] == null ? "" : names[id]).trim() : "";
        return `<bdi>${escapeHtml(name || id)}</bdi>`;
      }).join(", ")
    : escapeHtml(AQ_ANY_SPEAKER);
  return String(entry.message_he).split("{speaker}").map(escapeHtml).join(who);
}

// What the box shows: null for a result without a check (show nothing); a single muted line when
// no flag is warn/severe and nothing was fixed (info notes, if any, folded behind it); otherwise the
// title, the flags (most severe first), the applied fixes and the footnote.
// Returns {className, html, title}.
function audioQualityBox(aq, names) {
  if (!aq || typeof aq !== "object" || Array.isArray(aq)) return null;
  names = names && typeof names === "object" ? names : {};
  const usable = (e) => e && typeof e === "object" && typeof e.message_he === "string" && e.message_he.trim();
  const allFlags = (Array.isArray(aq.flags) ? aq.flags : []).filter(usable);
  // The worker also lists every applied fix as an info flag of the same code (for a speaker_gain
  // fix that flag names the lifted speaker, the fix itself does not). One line per fix: the twin
  // flag is folded into it and words the fix line when it names a speaker.
  const twins = new Set();
  const fixes = (Array.isArray(aq.fixes) ? aq.fixes : []).filter(usable).map((fx) => {
    const j = typeof fx.code === "string"
      ? allFlags.findIndex((f, i) => !twins.has(i) && f.code === fx.code) : -1;
    if (j < 0) return fx;
    twins.add(j);
    return allFlags[j].speaker ? allFlags[j] : fx;
  });
  const flags = allFlags
    .filter((e, i) => !twins.has(i))
    .map((e, i) => ({
      e, i,
      sev: typeof e.severity === "string" && _own(AQ_SEVERITY, e.severity) ? e.severity : "info",
    }))
    .sort((a, b) => AQ_SEVERITY[a.sev].rank - AQ_SEVERITY[b.sev].rank || a.i - b.i);
  const worst = flags.length ? flags[0].sev : "info";

  const line = (cls, label, entry) =>
    `<li class="aq-item ${cls}"><span class="aq-sev">${escapeHtml(label)}</span>` +
    `<span class="aq-msg">${aqMessageHtml(entry, names)}</span></li>`;
  const items = flags
    .map(({ e, sev }) => line(`aq-${sev}`, AQ_SEVERITY[sev].label, e))
    .concat(fixes.map((e) => line("aq-fix", AQ_FIX_LABEL, e)))
    .join("");
  const foot = `<div class="aq-foot">${escapeHtml(AQ_FOOTNOTE)}</div>`;

  if (worst === "info" && !fixes.length) {
    // Nothing to warn about. Info-only notes (e.g. a narrow-band line) stay one click away.
    if (!flags.length)
      return {
        className: "pii-notice aq aq-clean",
        html: escapeHtml(AQ_CLEAN),
        title: `${AQ_TITLE} — ${AQ_FOOTNOTE}`,
      };
    return {
      className: "pii-notice aq aq-clean",
      html: `<details><summary>${escapeHtml(AQ_CLEAN)}</summary><ul class="aq-list">${items}</ul>${foot}</details>`,
      title: AQ_TITLE,
    };
  }

  return {
    className: "pii-notice aq" + (worst === "severe" ? " err" : worst === "warn" ? " warn" : ""),
    html: `<b>${escapeHtml(AQ_TITLE)}</b><ul class="aq-list">${items}</ul>${foot}`,
    title: "",
  };
}

// The box element: created on first use, just above the transcript tabs.
function audioQualityEl() {
  let el = $("audioQuality");
  if (!el) {
    el = document.createElement("div");
    el.id = "audioQuality";
    const tabs = $("transcriptTabs");
    if (tabs && tabs.parentNode) tabs.parentNode.insertBefore(el, tabs);
    else $("transcriptView").appendChild(el);
  }
  return el;
}

function renderAudioQuality(res) {
  const el = audioQualityEl();
  const box = res ? audioQualityBox(res.audio_quality, res.speakers) : null;
  // A rename re-renders the view: keep the folded info notes open if they were, on the same job
  const keepOpen = !!(res && el.dataset.job === String(res.job_id) && el.querySelector("details[open]"));
  el.dataset.job = res ? String(res.job_id) : "";
  el.className = box ? box.className : "hidden";
  el.innerHTML = box ? box.html : "";
  if (keepOpen) { const d = el.querySelector("details"); if (d) d.open = true; }
  if (box && box.title) el.title = box.title;
  else el.removeAttribute("title");
}

// ---------- Tabs: full transcript / de-identified (PII) transcript ----------
// Both tabs feed the same #segments container, so the find bar, the "currently playing"
// highlight and click-to-seek keep working unchanged — only the text source differs.
function setupTabs() {
  $("tabOriginal").addEventListener("click", () => switchView("original"));
  $("tabPii").addEventListener("click", () => switchView("pii"));
  $("transcriptTabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    switchView(state.view === "pii" ? "original" : "pii");
    $(state.view === "pii" ? "tabPii" : "tabOriginal").focus();
  });
}

function resetPiiView() {
  state.view = "original";
  state.piiJobId = null;
  state.piiResult = null;
  state.piiError = null;
  state.piiLoading = false;
}

function updateTabs() {
  const pii = state.view === "pii";
  $("tabOriginal").classList.toggle("active", !pii);
  $("tabPii").classList.toggle("active", pii);
  $("tabOriginal").setAttribute("aria-selected", String(!pii));
  $("tabPii").setAttribute("aria-selected", String(pii));
}

async function switchView(view) {
  state.view = view;
  updateTabs();

  const res = state.result;
  if (view === "pii" && res) {
    const jobId = res.job_id;
    await ensurePiiResult(jobId);
    // The tab or the recording may have changed while the request was in flight
    if (state.view !== "pii" || !state.result || state.result.job_id !== jobId) return;
  }
  renderPiiNotice();
  renderSegments(state.findQuery);
}

// Fetch the de-identified copy once per recording and cache it in state.
// 404 here is a normal outcome (PII removal disabled, or the job predates it).
async function ensurePiiResult(jobId) {
  if (state.piiJobId === jobId && (state.piiResult || state.piiError)) return;
  state.piiJobId = jobId;
  state.piiResult = null;
  state.piiError = null;
  state.piiLoading = true;
  renderPiiNotice();
  renderSegments(state.findQuery);
  try {
    const r = await fetch(`${API}/jobs/${jobId}/pii?format=json`);
    if (!r.ok) {
      if (r.status === 404) {
        throw new Error(
          "לא נוצרה להקלטה זו גרסה ללא פרטים מזהים " +
          "(ייתכן ששלב ה-PII דולג או נכשל). הפרטים ביומן ה-worker."
        );
      }
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `שגיאה ${r.status}`);
    }
    const data = await r.json();
    if (state.piiJobId !== jobId) return;
    state.piiResult = data;
  } catch (err) {
    if (state.piiJobId !== jobId) return;
    state.piiError = err.message || "טעינת גרסת ה-PII נכשלה";
  } finally {
    if (state.piiJobId === jobId) state.piiLoading = false;
  }
}

function renderPiiNotice() {
  const el = $("piiNotice");
  const hide = () => { el.className = "pii-notice hidden"; el.innerHTML = ""; };
  if (state.view !== "pii") return hide();

  el.className = "pii-notice";
  if (state.piiLoading) {
    el.innerHTML = `<span class="spinner"></span> טוען את גרסת ה-PII…`;
    return;
  }
  if (state.piiError) {
    el.className = "pii-notice err";
    el.innerHTML = `<b>גרסת PII אינה זמינה</b> — ${escapeHtml(state.piiError)}`;
    return;
  }
  const res = state.piiResult;
  if (!res) return hide();

  const n = countPiiTags(res.segments);
  const engines = (res.pii_engines || []).join(", ");
  const removed = n
    ? `הוסרו <b>${n}</b> פרטים מזהים והוחלפו בתגיות כגון <span class="pii-tag">&lt;שם_&gt;</span>`
    : `לא נמצאו פרטים מזהים בתמלול זה`;
  el.innerHTML =
    `<b>תמלול ללא פרטים מזהים</b> — ` +
    `זו הגרסה שנשלחת ל-Gemini להפקת הסיכומים. ` +
    `${removed}${engines ? ` · מנועי זיהוי: ${escapeHtml(engines)}` : ""}.`;
}

// ---------- Internal search (within the transcription) ----------
function setupFind() {
  $("findInput").addEventListener("input", (e) => doFind(e.target.value));
  $("findNext").addEventListener("click", () => navigateFind(1));
  $("findPrev").addEventListener("click", () => navigateFind(-1));
  $("findInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); navigateFind(e.shiftKey ? -1 : 1); }
  });
}

function doFind(query) {
  state.findQuery = (query || "").trim();
  renderSegments(state.findQuery);
}

function updateFindState() {
  state.findMarks = Array.from(document.querySelectorAll("#segments mark"));
  state.findIndex = -1;
  const count = $("findCount");
  if (!state.findQuery) { count.textContent = ""; return; }
  count.textContent = state.findMarks.length ? `${state.findMarks.length} תוצאות` : "אין תוצאות";
  if (state.findMarks.length) navigateFind(1);
}

function navigateFind(dir) {
  const marks = state.findMarks;
  if (!marks.length) return;
  if (state.findIndex >= 0 && marks[state.findIndex]) marks[state.findIndex].classList.remove("current");
  state.findIndex = (state.findIndex + dir + marks.length) % marks.length;
  const m = marks[state.findIndex];
  m.classList.add("current");
  m.scrollIntoView({ behavior: "smooth", block: "center" });
  $("findCount").textContent = `${state.findIndex + 1} / ${marks.length}`;
}

// ---------- Global search (across all transcriptions) ----------
function setupGlobalSearch() {
  const input = $("globalSearch");
  input.addEventListener("input", () => {
    clearTimeout(state.searchTimer);
    const q = input.value.trim();
    if (q.length < 2) { $("searchResults").classList.remove("show"); return; }
    state.searchTimer = setTimeout(() => runGlobalSearch(q), 300);
  });
}

async function runGlobalSearch(q) {
  const box = $("searchResults");
  try {
    const r = await fetch(`${API}/search?q=${encodeURIComponent(q)}`);
    const data = await r.json();
    box.classList.add("show");
    if (!data.count) {
      box.innerHTML = `<div class="sr-meta">לא נמצאו תוצאות עבור "${escapeHtml(q)}"</div>`;
      return;
    }
    let html = `<div class="sr-meta">${data.count} תוצאות עבור "${escapeHtml(q)}"</div>`;
    for (const m of data.matches) {
      html += `
        <div class="sr-item" data-job="${m.job_id}" data-start="${m.start}" data-q="${escapeHtml(q)}">
          <div class="sr-top">
            <span class="sr-name">${escapeHtml(m.original_name)}</span>
            <span>${escapeHtml(m.speaker)} · ${fmtClock(m.start)}</span>
          </div>
          <div class="sr-text" dir="auto">${highlightHtml(m.text, q)}</div>
        </div>`;
    }
    box.innerHTML = html;
    box.querySelectorAll(".sr-item").forEach((it) =>
      it.addEventListener("click", () =>
        openSearchResult(it.dataset.job, parseFloat(it.dataset.start), it.dataset.q)
      )
    );
  } catch {
    box.classList.add("show");
    box.innerHTML = `<div class="sr-meta">שגיאת חיפוש</div>`;
  }
}

async function openSearchResult(jobId, start, query) {
  await selectJob(jobId);
  // Short delay for rendering the transcription, then jump to time + highlight
  setTimeout(() => {
    if (state.result && state.result.job_id === jobId) {
      const player = $("player");
      player.currentTime = start;
      $("findInput").value = query;
      doFind(query);
      highlightSegment(start);
      const playing = document.querySelector(".seg.playing");
      if (playing) playing.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, 250);
}

// ---------- Highlighting terms ----------
// The redaction placeholders (<שם_>, <מספר_טלפון> …) get a visible tag style.
// The raw text is split on the placeholders first and each piece escaped/highlighted on its
// own, so a search hit *inside* a placeholder nests a <mark> in the tag instead of splitting it.
function piiHtml(text, query) {
  return String(text || "")
    .split(/(<[^<>]*>)/g)
    .map((part) =>
      /^<[^<>]*>$/.test(part)
        ? `<span class="pii-tag">${highlightHtml(part, query)}</span>`
        : highlightHtml(part, query)
    )
    .join("");
}

function countPiiTags(segments) {
  let n = 0;
  for (const seg of segments || []) n += (String(seg.text || "").match(/<[^<>]*>/g) || []).length;
  return n;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightHtml(text, query) {
  const esc = escapeHtml(text);
  if (!query) return esc;
  const terms = query.split(/\s+/).filter(Boolean).map(escapeRegex);
  if (!terms.length) return esc;
  try {
    const re = new RegExp("(" + terms.join("|") + ")", "gi");
    return esc.replace(re, "<mark>$1</mark>");
  } catch {
    return esc;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---------- Polling cycle ----------
function startPolling() {
  state.pollTimer = setInterval(async () => {
    const hadProcessing = state.jobs.some((j) => ["queued", "processing"].includes(j.status));
    await refreshJobs();
    // If the current job just finished — load the result
    const cur = state.jobs.find((j) => j.id === state.currentJobId);
    if (cur && cur.status === "done" && !state.result) await loadResult(cur.id);
    if (cur && cur.status === "processing") showProcessing(cur);
    void hadProcessing;
  }, 2500);
}

// ---------- Statistics dashboard ----------
function hideDashboard() {
  $("dashboardView").classList.add("hidden");
  $("dashBtn").classList.remove("active");
}

function toggleDashboard() {
  if ($("dashboardView").classList.contains("hidden")) {
    hideSettings();
    $("emptyState").classList.add("hidden");
    $("transcriptView").classList.add("hidden");
    $("dashboardView").classList.remove("hidden");
    $("dashBtn").classList.add("active");
    loadStats();
  } else {
    hideDashboard();
    if (state.currentJobId) selectJob(state.currentJobId);
    else showEmpty();
  }
}

async function loadStats() {
  $("kpis").innerHTML = `<div class="dash-empty">טוען…</div>`;
  try {
    const s = await (await fetch(`${API}/stats`)).json();
    renderDashboard(s);
  } catch {
    $("kpis").innerHTML = `<div class="dash-empty">שגיאה בטעינת הסטטיסטיקות</div>`;
  }
}

function _kpi(v, l, cls = "") {
  return `<div class="kpi"><div class="v ${cls}">${v}</div><div class="l">${l}</div></div>`;
}

function renderDashboard(s) {
  const bs = s.by_status || {};
  const dur = (sec) => (sec ? fmtClock(sec) : "—");
  $("kpis").innerHTML = [
    _kpi(s.total, "סך הקלטות"),
    _kpi(bs.done || 0, "הושלמו", "ok"),
    _kpi((bs.processing || 0) + (bs.queued || 0), "בעיבוד / בתור", "accent"),
    _kpi(bs.error || 0, "נכשלו", (bs.error ? "err" : "")),
    _kpi((s.audio_seconds / 3600).toFixed(1), "שעות אודיו"),
    _kpi(s.segments.toLocaleString("he-IL"), "קטעי תמלול"),
    _kpi(s.with_summary, "עם סיכום"),
    _kpi(s.with_pii, "עם גרסת PII"),
    _kpi(dur(s.avg_duration_sec), "אורך הקלטה ממוצע"),
    _kpi(dur(s.avg_processing_sec), "זמן עיבוד ממוצע"),
  ].join("");

  const ddmm = (iso) => { const p = (iso || "").split("-"); return p.length === 3 ? `${p[2]}/${p[1]}` : iso; };
  renderBars("chartPerDay", (s.per_day || []).map((d) => ({ label: ddmm(d.date), value: d.count })));
  renderBars(
    "chartSpeakers",
    Object.entries(s.speaker_distribution || {}).map(([k, v]) => ({ label: `${k} 🗣`, value: v }))
  );
  renderCloud("chartKeywords", s.top_keywords || []);
  renderCloud("chartTopics", s.top_topics || []);
}

function renderBars(id, items) {
  const el = $(id);
  if (!items.length) { el.innerHTML = `<div class="dash-empty">אין נתונים</div>`; return; }
  const max = Math.max(...items.map((i) => i.value), 1);
  el.innerHTML = items
    .map(
      (i) => `<div class="bar-col" title="${escapeHtml(i.label)}: ${i.value}">
        <span class="bar-val">${i.value}</span>
        <div class="bar" style="height:${Math.round((i.value / max) * 120)}px"></div>
        <span class="bar-lbl">${escapeHtml(i.label)}</span></div>`
    )
    .join("");
}

function renderCloud(id, items) {
  const el = $(id);
  if (!items.length) {
    el.innerHTML = `<div class="dash-empty">עדיין אין נתונים (נדרשים סיכומים)</div>`;
    return;
  }
  el.innerHTML = items
    .map(
      (k) =>
        `<span class="kw-item" data-term="${escapeHtml(k.term)}">${escapeHtml(k.term)}<span class="c">${k.count}</span></span>`
    )
    .join("");
  el.querySelectorAll(".kw-item").forEach((it) =>
    it.addEventListener("click", () => {
      toggleDashboard(); // Close the dashboard
      $("globalSearch").value = it.dataset.term;
      runGlobalSearch(it.dataset.term);
    })
  );
}

// ---------- Settings: prompts + branding ----------
const _cap = (s) => s[0].toUpperCase() + s.slice(1);

function hideSettings() {
  $("settingsView").classList.add("hidden");
  $("settingsBtn").classList.remove("active");
}

function toggleSettings() {
  if ($("settingsView").classList.contains("hidden")) {
    hideDashboard();
    $("emptyState").classList.add("hidden");
    $("transcriptView").classList.add("hidden");
    $("settingsView").classList.remove("hidden");
    $("settingsBtn").classList.add("active");
    loadPrompts();
    loadBrandingEditor();
    loadSecrets();
  } else {
    hideSettings();
    if (state.currentJobId) selectJob(state.currentJobId);
    else showEmpty();
  }
}

async function loadPrompts() {
  const el = $("promptsEditor");
  el.innerHTML = `<div class="dash-empty">טוען…</div>`;
  try {
    const d = await (await fetch(`${API}/prompts`)).json();
    renderPrompts(d.prompts);
  } catch {
    el.innerHTML = `<div class="dash-empty">שגיאה בטעינת הפרומפטים</div>`;
  }
}

function renderPrompts(prompts) {
  const el = $("promptsEditor");
  el.innerHTML = "";
  for (const [key, p] of Object.entries(prompts)) {
    const f = document.createElement("div");
    f.className = "prompt-field";
    f.innerHTML = `
      <div class="pf-head">
        <label>${escapeHtml(p.label)}</label>
        ${p.is_custom ? '<span class="pf-badge">נערך</span>' : ""}
      </div>
      <textarea data-key="${key}"></textarea>
      <div class="pf-actions">
        <button class="btn ghost pf-save" data-key="${key}">שמור</button>
        <button class="btn ghost pf-reset" data-key="${key}">שחזר ברירת מחדל</button>
        <span class="pf-saved" data-key="${key}"></span>
      </div>`;
    f.querySelector("textarea").value = p.value;
    el.appendChild(f);
  }
  el.querySelectorAll(".pf-save").forEach((b) => b.addEventListener("click", () => savePrompt(b.dataset.key)));
  el.querySelectorAll(".pf-reset").forEach((b) => b.addEventListener("click", () => resetPrompt(b.dataset.key)));
}

async function savePrompt(key) {
  const ta = document.querySelector(`#promptsEditor textarea[data-key="${key}"]`);
  const msg = document.querySelector(`#promptsEditor .pf-saved[data-key="${key}"]`);
  try {
    const r = await fetch(`${API}/prompts`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompts: { [key]: ta.value } }),
    });
    if (r.ok) { msg.textContent = "נשמר ✓"; setTimeout(() => (msg.textContent = ""), 2500); loadPrompts(); }
    else msg.textContent = "שגיאה";
  } catch { msg.textContent = "שגיאה"; }
}

async function resetPrompt(key) {
  await fetch(`${API}/prompts/reset?key=${encodeURIComponent(key)}`, { method: "POST" });
  loadPrompts();
}

// ---------- API keys (override .env) ----------

async function loadSecrets() {
  const st = $("geminiKeyStatus");
  try {
    const d = await (await fetch(`${API}/secrets`)).json();
    const s = (d.secrets || {}).gemini_api_key || {};
    if (s.source === "ui") st.innerHTML = `<b class="ok">מוגדר מהממשק</b> (${escapeHtml(s.masked || "")}) · דורס את ‎.env‎`;
    else if (s.source === "env") st.innerHTML = `מוגדר מקובץ ‎.env‎ (${escapeHtml(s.masked || "")})`;
    else st.innerHTML = `<b class="warn">לא מוגדר</b> — נדרש רק אם עוברים למודל חיצוני`;
  } catch {
    st.textContent = "שגיאה בטעינת מצב המפתח";
  }
}

async function saveSecret() {
  const input = $("geminiKeyInput");
  const st = $("geminiKeyStatus");
  const value = input.value.trim();
  if (!value) { st.textContent = "יש להזין מפתח"; return; }
  try {
    const r = await fetch(`${API}/secrets/gemini_api_key`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    });
    if (r.ok) { input.value = ""; loadSecrets(); }
    else { const d = await r.json().catch(() => ({})); st.textContent = d.detail || "שמירה נכשלה"; }
  } catch { st.textContent = "שמירה נכשלה"; }
}

async function clearSecret() {
  await fetch(`${API}/secrets/gemini_api_key`, { method: "DELETE" });
  $("geminiKeyInput").value = "";
  loadSecrets();
}

async function loadBrandingEditor() {
  try {
    const st = await (await fetch(`${API}/branding`)).json();
    for (const side of ["right", "left"]) {
      const prev = $(`logo${_cap(side)}Preview`);
      prev.innerHTML = st[side]
        ? `<img src="${API}/branding/${side}?t=${Date.now()}" alt="" />`
        : "אין לוגו";
    }
  } catch {}
}

async function uploadLogo(side, file) {
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch(`${API}/branding/${side}`, { method: "POST", body: fd });
    if (r.ok) { loadBrandingEditor(); applyBranding(); }
    else { const d = await r.json().catch(() => ({})); alert(d.detail || "העלאה נכשלה"); }
  } catch { alert("העלאה נכשלה"); }
}

async function removeLogo(side) {
  await fetch(`${API}/branding/${side}`, { method: "DELETE" });
  loadBrandingEditor();
  applyBranding();
}

// Display the logos in the title
async function applyBranding() {
  try {
    const st = await (await fetch(`${API}/branding`)).json();
    for (const side of ["right", "left"]) {
      const img = $(`logo${_cap(side)}`);
      if (st[side]) { img.src = `${API}/branding/${side}?t=${Date.now()}`; img.classList.remove("hidden"); }
      else img.classList.add("hidden");
    }
  } catch {}
}

function setupSettings() {
  $("settingsBtn").addEventListener("click", toggleSettings);
  $("promptsResetAll").addEventListener("click", async () => {
    await fetch(`${API}/prompts/reset`, { method: "POST" });
    loadPrompts();
  });
  $("geminiKeySave").addEventListener("click", saveSecret);
  $("geminiKeyClear").addEventListener("click", clearSecret);
  document.querySelectorAll("[data-logo]").forEach((b) => {
    const side = b.dataset.logo;
    if (b.dataset.act === "upload") b.addEventListener("click", () => $(`logo${_cap(side)}Input`).click());
    if (b.dataset.act === "remove") b.addEventListener("click", () => removeLogo(side));
  });
  ["right", "left"].forEach((side) =>
    $(`logo${_cap(side)}Input`).addEventListener("change", (e) => uploadLogo(side, e.target.files[0]))
  );
}

// ---------- Initialization ----------
checkHealth();
setupUpload();
setupFind();
setupTabs();
setupGlobalSearch();
setupSettings();
applyBranding();
refreshJobs();
startPolling();
$("refreshBtn").addEventListener("click", refreshJobs);
$("dashBtn").addEventListener("click", toggleDashboard);
$("dashRefresh").addEventListener("click", loadStats);
setInterval(checkHealth, 15000);
