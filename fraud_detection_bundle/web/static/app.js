// MythosBank · Forensic Integrity Console
// ----------------------------------------------------------------- state
const state = {
  /** @type {{ id: string, file: File, kind: string, refOf: string|null }[]} */
  files: [],
  busy: false,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const fmtBytes = (n) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(2)} MB`;
};
const uid = () => Math.random().toString(36).slice(2, 9);

// ----------------------------------------------------------------- health probe
async function probeHealth() {
  const pill = $("#health-pill");
  const text = $("#health-text");
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(r.statusText);
    const j = await r.json();
    pill.classList.remove("ok", "online");
    if (j.online) {
      pill.classList.add("online");
      text.textContent = `live · ${j.model}`;
    } else {
      pill.classList.add("ok");
      text.textContent = `offline mode · ${j.model || "local fusion"}`;
    }
  } catch {
    pill.classList.remove("ok", "online");
    text.textContent = "backend unreachable";
  }
}

// ----------------------------------------------------------------- file picker
const dz = $("#dropzone");
const input = $("#file-input");

dz.addEventListener("click", () => input.click());
dz.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    input.click();
  }
});

["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.add("over");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.remove("over");
  })
);
dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
input.addEventListener("change", () => {
  addFiles(input.files);
  input.value = "";
});

function addFiles(fileList) {
  for (const f of fileList) {
    state.files.push({
      id: uid(),
      file: f,
      kind: guessKind(f.name),
      refOf: null,
    });
  }
  renderFileList();
  refreshSubmit();
}

function guessKind(name) {
  const ext = name.toLowerCase().split(".").pop();
  if (["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"].includes(ext))
    return "auto";
  if (["pdf"].includes(ext)) return "document";
  if (["mp4", "mov", "avi", "mkv", "webm", "m4v"].includes(ext)) return "video";
  return "auto";
}

// ----------------------------------------------------------------- file list render
function renderFileList() {
  const list = $("#file-list");
  list.innerHTML = "";
  for (const item of state.files) {
    const tpl = $("#tpl-file-row").content.firstElementChild.cloneNode(true);
    tpl.dataset.id = item.id;
    $(".file-name", tpl).textContent = item.file.name;
    $(".file-detail", tpl).textContent = `${item.file.type || "—"} · ${fmtBytes(item.file.size)}`;

    const kindSel = $(".kind-select", tpl);
    kindSel.value = item.kind;
    kindSel.addEventListener("change", () => {
      item.kind = kindSel.value;
      renderFileList(); // re-render so ref selectors appear/hide
    });

    // Reference dropdown for signatures
    const refSel = $(".ref-select", tpl);
    if (item.kind === "signature") {
      refSel.hidden = false;
      refSel.innerHTML = `<option value="">— questioned (no reference) —</option>`;
      for (const candidate of state.files) {
        if (candidate.id === item.id) continue;
        if (candidate.kind !== "signature" && candidate.kind !== "image" && candidate.kind !== "auto")
          continue;
        const opt = document.createElement("option");
        opt.value = candidate.file.name;
        opt.textContent = `↪ reference: ${candidate.file.name}`;
        if (item.refOf === candidate.file.name) opt.selected = true;
        refSel.appendChild(opt);
      }
      refSel.addEventListener("change", () => {
        item.refOf = refSel.value || null;
      });
    }

    $(".remove", tpl).addEventListener("click", () => {
      state.files = state.files.filter((x) => x.id !== item.id);
      // Clear refOf entries that pointed at the removed file
      for (const x of state.files)
        if (x.refOf === item.file.name) x.refOf = null;
      renderFileList();
      refreshSubmit();
    });

    list.appendChild(tpl);
  }
}

function refreshSubmit() {
  $("#submit-btn").disabled = state.busy || state.files.length === 0;
}

// ----------------------------------------------------------------- submit
$("#submit-btn").addEventListener("click", async () => {
  if (state.busy || state.files.length === 0) return;
  state.busy = true;
  refreshSubmit();
  $("#submit-btn").classList.add("loading");
  $("#error").hidden = true;
  $("#report").hidden = true;
  $("#placeholder").style.display = "flex";
  $("#placeholder").innerHTML = `
    <div class="ph-shield"><svg viewBox="0 0 64 64"><path d="M32 4 L56 14 V32 C56 46 44 56 32 60 C20 56 8 46 8 32 V14 Z"
                fill="none" stroke="currentColor" stroke-width="2"/></svg></div>
    <h2>Running forensic pipeline…</h2>
    <p>ELA · JPEG-Q · DCT-Benford · CFA · pHash copy-move · PRNU · Lighting</p>`;

  const form = new FormData();
  for (const item of state.files) {
    form.append("files", item.file, item.file.name);
    form.append("kinds", item.kind);
    form.append("labels", item.kind === "signature" ? "signature" : item.kind);
    form.append("references", item.refOf || "");
  }
  const ctx = $("#context").value.trim();
  if (ctx) form.append("context", ctx);

  try {
    const r = await fetch("/api/analyze", { method: "POST", body: form });
    if (!r.ok) {
      const err = await safeJSON(r);
      throw new Error(err?.detail || `Server returned ${r.status}`);
    }
    const report = await r.json();
    renderReport(report);
  } catch (e) {
    $("#error").hidden = false;
    $("#error").textContent = `Analysis failed: ${e.message}`;
    $("#placeholder").style.display = "flex";
    $("#placeholder").innerHTML = `
      <div class="ph-shield"><svg viewBox="0 0 64 64"><path d="M32 4 L56 14 V32 C56 46 44 56 32 60 C20 56 8 46 8 32 V14 Z"
                  fill="none" stroke="currentColor" stroke-width="2"/></svg></div>
      <h2>No report</h2>
      <p>Adjust your inputs and try again.</p>`;
  } finally {
    state.busy = false;
    $("#submit-btn").classList.remove("loading");
    refreshSubmit();
  }
});

async function safeJSON(r) {
  try { return await r.json(); } catch { return null; }
}

// ----------------------------------------------------------------- render report
function renderReport(report) {
  $("#placeholder").style.display = "none";
  const root = $("#report");
  root.hidden = false;

  // verdict
  const verdict = report.verdict || "inconclusive";
  const risk = report.risk || "minimal";
  const score = Number(report.score || 0);
  const verdictEl = $(".verdict");
  verdictEl.classList.remove("high", "medium", "low", "minimal");
  verdictEl.classList.add(risk);

  $("#v-verdict").textContent = verdict;
  $("#v-summary").textContent = report.summary || "";
  const riskPill = $("#v-risk");
  riskPill.textContent = risk;
  riskPill.classList.remove("high", "medium", "low", "minimal");
  riskPill.classList.add(risk);

  animateScore($("#v-score"), score);
  animateRing($("#v-score-ring-fg"), score);

  // input cards
  const cards = $("#v-cards");
  cards.innerHTML = "";
  const chain = report.chain_of_evidence || [];
  for (const entry of chain) {
    cards.appendChild(renderCard(entry));
  }
  if (chain.length === 0) {
    cards.innerHTML = `<p class="placeholder" style="min-height:0">No chain-of-evidence entries returned.</p>`;
  }

  // recommendations
  const ul = $("#v-recs");
  ul.innerHTML = "";
  for (const rec of report.recommendations || []) {
    const li = document.createElement("li");
    li.textContent = rec;
    ul.appendChild(li);
  }
  if (!ul.children.length) {
    ul.innerHTML = `<li>No additional actions recommended.</li>`;
  }

  // provenance
  const dl = $("#v-provenance");
  dl.innerHTML = "";
  addRow(dl, "Generated", report.generated_at || "—");
  addRow(dl, "Model", report.model || "offline");
  if (report.reproducibility_hash) {
    const dt = document.createElement("dt");
    dt.textContent = "Reproducibility";
    const dd = document.createElement("dd");
    dd.classList.add("copy");
    dd.textContent = report.reproducibility_hash;
    dd.title = "Click to copy";
    dd.addEventListener("click", () => {
      navigator.clipboard.writeText(report.reproducibility_hash).then(() => {
        dd.classList.add("copied");
        setTimeout(() => dd.classList.remove("copied"), 1400);
      });
    });
    dl.appendChild(dt); dl.appendChild(dd);
  }
  if (report.algorithm_versions) {
    const versions = Object.entries(report.algorithm_versions)
      .map(([k, v]) => `${k}@${v}`).join(" · ");
    addRow(dl, "Algorithms", versions);
  }

  $("#v-raw").textContent = JSON.stringify(report, null, 2);

  // download
  const dl_btn = $("#download-btn");
  dl_btn.onclick = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fraud_report_${(report.reproducibility_hash || Date.now()).toString().slice(0, 12)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  root.scrollIntoView({ behavior: "smooth", block: "start" });
}

function addRow(dl, key, val) {
  const dt = document.createElement("dt");
  dt.textContent = key;
  const dd = document.createElement("dd");
  dd.textContent = val;
  dl.appendChild(dt);
  dl.appendChild(dd);
}

function renderCard(entry) {
  const tpl = $("#tpl-input-card").content.firstElementChild.cloneNode(true);
  $(".card-title", tpl).textContent =
    `${entry.label || "input"}  ·  ${entry.kind || ""}`;

  const score = Number(
    entry.overall_score ?? entry.forgery_score ?? 0
  );
  const risk = entry.risk || riskFromScore(score);
  const badge = $(".card-badge", tpl);
  badge.textContent = `${risk} · ${score.toFixed(3)}`;
  badge.classList.add(risk);

  const meta = [];
  if (entry.input) meta.push(entry.input.split("/").pop());
  if (entry.size_bytes) meta.push(fmtBytes(entry.size_bytes));
  if (entry.sha256) meta.push(`sha256: ${entry.sha256.slice(0, 16)}…`);
  if (entry.page_count) meta.push(`pages: ${entry.page_count}`);
  if (entry.duplicate_frames !== undefined)
    meta.push(`dup-frames: ${entry.duplicate_frames}`);
  if (entry.temporal_score !== undefined)
    meta.push(`temporal: ${entry.temporal_score}`);
  $(".card-meta", tpl).textContent = meta.join("  ·  ");

  const bars = $(".card-bars", tpl);
  if (Array.isArray(entry.detectors)) {
    for (const d of entry.detectors) {
      bars.appendChild(renderBar(d.name, d.score, d.confidence, d.version));
    }
  }
  return tpl;
}

function renderBar(name, score, confidence, version) {
  const wrap = document.createElement("div");
  const r = riskFromScore(score);
  wrap.className = `bar ${r}`;
  wrap.innerHTML = `
    <span class="bar-name" title="${name}@${version || "?"}">${name}</span>
    <div class="bar-track"><div class="bar-fill"></div></div>
    <span class="bar-value">${score.toFixed(2)}<span class="bar-conf"> ·c${confidence?.toFixed(2) ?? "?"}</span></span>
  `;
  // Animate fill in next frame
  requestAnimationFrame(() => {
    wrap.querySelector(".bar-fill").style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
  });
  return wrap;
}

function riskFromScore(s) {
  if (s >= 0.4434) return "high";
  if (s >= 0.2412) return "medium";
  if (s >= 0.1912) return "low";
  return "minimal";
}

function animateScore(el, target) {
  const from = 0;
  const dur = 1100;
  const t0 = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
    const v = from + (target - from) * eased;
    el.textContent = v.toFixed(3);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function animateRing(el, score) {
  const c = 2 * Math.PI * 52; // r=52 -> ~326.7
  // Reset first so the transition replays.
  el.style.transition = "none";
  el.style.strokeDashoffset = c;
  // Force layout
  void el.getBoundingClientRect();
  el.style.transition = "";
  const offset = c - Math.max(0, Math.min(1, score)) * c;
  el.style.strokeDashoffset = offset;
}

// ----------------------------------------------------------------- boot
probeHealth();
setInterval(probeHealth, 30000);
