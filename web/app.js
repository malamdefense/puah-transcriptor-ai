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
};
function speakerColor(id, speakerIds) {
  const idx = speakerIds.indexOf(id);
  return SPEAKER_COLORS[(idx >= 0 ? idx : 0) % SPEAKER_COLORS.length];
}

// ---------- Health check ----------
async function checkHealth() {
  const el = $("health");
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error();
    await r.json();
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
  $("tMeta").textContent =
    `${res.num_speakers} דוברים · משך ${fmtDuration(res.duration_sec)} · ${res.segments.length} קטעים`;

  // Export links
  $("expWord").href = `${API}/jobs/${res.job_id}/export?format=docx`;
  $("expTxt").href = `${API}/jobs/${res.job_id}/export?format=txt`;
  $("expSrt").href = `${API}/jobs/${res.job_id}/export?format=srt`;
  $("expVtt").href = `${API}/jobs/${res.job_id}/export?format=vtt`;
  $("expJson").href = `${API}/jobs/${res.job_id}/export?format=json`;
  $("expPiiWord").href = `${API}/jobs/${res.job_id}/pii?format=docx`;
  $("expPiiJson").href = `${API}/jobs/${res.job_id}/pii?format=json`;

  // Automatic summary + PUAH products (client summary, feedback to the consultant)
  renderSummary(res);
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

  // Reset internal search and build segments
  $("findInput").value = "";
  state.findQuery = "";
  renderSegments("");

  // Highlight the currently playing segment
  player.ontimeupdate = () => highlightSegment(player.currentTime);
}

// Build segments, with optional internal search highlighting
function renderSegments(query) {
  const res = state.result;
  if (!res) return;
  const speakerIds = Object.keys(res.speakers);
  const segWrap = $("segments");
  segWrap.innerHTML = "";

  res.segments.forEach((seg, i) => {
    const div = document.createElement("div");
    div.className = "seg";
    const color = speakerColor(seg.speaker, speakerIds);
    div.style.borderRightColor = color;
    div.dataset.start = seg.start;
    div.dataset.end = seg.end;
    div.dataset.idx = i;
    div.innerHTML = `
      <div class="seg-head">
        <span class="seg-speaker" style="color:${color}">${escapeHtml(res.speakers[seg.speaker] || seg.speaker)}</span>
        <span class="seg-time">${fmtClock(seg.start)} – ${fmtClock(seg.end)}</span>
      </div>
      <div class="seg-text">${highlightHtml(seg.text, query)}</div>`;
    div.onclick = () => { const p = $("player"); p.currentTime = seg.start; p.play(); };
    segWrap.appendChild(div);
  });

  updateFindState();
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

// ---------- Automatic summary ----------
function renderSummary(res) {
  const el = $("summarySection");
  const s = res.summary;

  const btn = `<button id="genSummaryBtn" class="btn ghost">${s ? "רענן סיכום" : "צור סיכום"}</button>`;

  if (!s) {
    el.innerHTML = `
      <div class="summary-card">
        <div class="sc-head"><h3>📝 סיכום אוטומטי</h3>${btn}</div>
        <div class="sc-empty">אין סיכום עדיין. לחצו "צור סיכום" (דורש מודל שפה מקומי פעיל).</div>
      </div>`;
    $("genSummaryBtn").onclick = generateSummary;
    return;
  }

  const list = (arr) =>
    arr && arr.length ? `<ul>${arr.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>` : "";
  const tags = (arr, cls) =>
    arr && arr.length
      ? `<div class="tags">${arr.map((x) => `<span class="tag ${cls}" data-term="${escapeHtml(x)}">${escapeHtml(x)}</span>`).join("")}</div>`
      : "";

  el.innerHTML = `
    <div class="summary-card">
      <div class="sc-head"><h3>📝 סיכום אוטומטי</h3>${btn}</div>
      ${s.summary ? `<div class="sc-body">${escapeHtml(s.summary)}</div>` : ""}
      ${s.key_points && s.key_points.length ? `<div class="sc-block"><h4>נקודות עיקריות</h4>${list(s.key_points)}</div>` : ""}
      ${s.action_items && s.action_items.length ? `<div class="sc-block"><h4>פעולות והמשך טיפול</h4>${list(s.action_items)}</div>` : ""}
      ${s.topics && s.topics.length ? `<div class="sc-block"><h4>נושאים</h4>${tags(s.topics, "")}</div>` : ""}
      ${s.keywords && s.keywords.length ? `<div class="sc-block"><h4>מילות מפתח</h4>${tags(s.keywords, "kw")}</div>` : ""}
    </div>`;

  $("genSummaryBtn").onclick = generateSummary;
  // Clicking on a topic/keyword -> internal search
  el.querySelectorAll(".tag").forEach((t) =>
    t.addEventListener("click", () => {
      const term = t.dataset.term;
      $("findInput").value = term;
      doFind(term);
    })
  );
}

function renderPuahOutputs(res) {
  const el = $("puahSection");
  const cs = res.client_summary;
  const fb = res.feedback;
  let html = "";
  if (cs)
    html += `<div class="puah-card client">
      <div class="pc-head"><h3>📩 סיכום ללקוח</h3>
        <button class="btn ghost copy-btn" data-copy="cs">העתק</button></div>
      <div class="puah-text">${escapeHtml(cs)}</div></div>`;
  if (fb)
    html += `<div class="puah-card feedback">
      <div class="pc-head"><h3>📋 משוב ליועץ (פנימי)</h3>
        <button class="btn ghost copy-btn" data-copy="fb">העתק</button></div>
      <div class="puah-text">${escapeHtml(fb)}</div></div>`;
  el.innerHTML = html;
  el.querySelectorAll(".copy-btn").forEach((b) =>
    b.addEventListener("click", () => {
      const txt = b.dataset.copy === "cs" ? cs : fb;
      if (navigator.clipboard) navigator.clipboard.writeText(txt);
      b.textContent = "הועתק ✓";
      setTimeout(() => (b.textContent = "העתק"), 1500);
    })
  );
}

async function generateSummary() {
  const btn = $("genSummaryBtn");
  if (!state.result) return;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> מסכם…`;
  try {
    const r = await fetch(`${API}/jobs/${state.result.job_id}/summarize`, { method: "POST" });
    if (r.ok) {
      state.result = await r.json();
      renderSummary(state.result);
      renderPuahOutputs(state.result);
    } else {
      const d = await r.json().catch(() => ({}));
      const card = $("summarySection").querySelector(".summary-card");
      if (card) {
        card.insertAdjacentHTML(
          "beforeend",
          `<div class="sc-empty" style="color:var(--err);margin-top:10px">${escapeHtml(d.detail || "יצירת הסיכום נכשלה")}</div>`
        );
      }
      btn.disabled = false;
      btn.textContent = "נסה שוב";
    }
  } catch {
    btn.disabled = false;
    btn.textContent = "נסה שוב";
  }
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
          <div class="sr-text">${highlightHtml(m.text, q)}</div>
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
setupGlobalSearch();
setupSettings();
applyBranding();
refreshJobs();
startPolling();
$("refreshBtn").addEventListener("click", refreshJobs);
$("dashBtn").addEventListener("click", toggleDashboard);
$("dashRefresh").addEventListener("click", loadStats);
setInterval(checkHealth, 15000);
