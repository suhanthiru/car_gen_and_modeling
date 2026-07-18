// Capture client. Uploads to the server that served this page, so the same
// build works on the laptop's LAN IP today and a Tailscale/cloud host later
// with no change.
const API = "";

const $ = (id) => document.getElementById(id);
const nameIn = $("name");
const statusEl = $("status");
const progEl = $("prog");

nameIn.value = localStorage.getItem("cargen.lastName") || "";
nameIn.addEventListener("input", () =>
  localStorage.setItem("cargen.lastName", nameIn.value)
);

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + kind;
}

function pickFile(inputId) {
  if (!nameIn.value.trim()) {
    setStatus("Give the vehicle a name first.", "err");
    nameIn.focus();
    return;
  }
  $(inputId).click();
}

$("photoBtn").onclick = () => pickFile("photoIn");
$("videoBtn").onclick = () => pickFile("videoIn");
$("photoIn").onchange = (e) => upload(e.target.files[0], "photo");
$("videoIn").onchange = (e) => upload(e.target.files[0], "video");

// XHR rather than fetch: upload progress is the whole point of the bar, and
// fetch still can't report it.
function upload(file, kind) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("name", nameIn.value.trim());
  form.append("device", $("device").value);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", API + "/observations");
  setStatus(`Uploading ${kind}…`);
  progEl.style.width = "0%";

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = (e.loaded / e.total) * 100;
    progEl.style.width = pct + "%";
    if (pct >= 100) setStatus("Reconstructing… (this runs on the server)");
  };
  xhr.onload = () => {
    progEl.style.width = "0%";
    if (xhr.status >= 400) {
      let detail = xhr.statusText;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch {}
      setStatus("Failed: " + detail, "err");
      return;
    }
    const res = JSON.parse(xhr.responseText);
    const pct = (res.result.observed_fraction * 100).toFixed(1);
    setStatus(
      `Done — ${res.vehicle.name}: ${res.result.splats} splats, ${pct}% observed` +
        (res.result.frames_fused > 1 ? ` (${res.result.frames_fused} frames fused)` : ""),
      "ok"
    );
    showView(res.vehicle.folder); // drop the fresh model into the embedded viewer
    refresh();
  };
  xhr.onerror = () => setStatus("Network error — is the server still running?", "err");
  xhr.send(form);
}

async function refresh() {
  await Promise.all([refreshVehicles(), refreshPending()]);
}

// Show a vehicle in the embedded viewer (same page, no navigation). The iframe
// IS the full viewer, so the Before/Photoreal toggle and live refresh come free.
function showView(folder) {
  const section = $("viewSection");
  const view = $("view");
  const src = `/viewer/?v=${encodeURIComponent(folder)}`;
  if (view.getAttribute("src") !== src) view.src = src;
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function refreshVehicles() {
  const list = $("vehicles");
  try {
    const { vehicles } = await (await fetch(API + "/vehicles")).json();
    if (!vehicles.length) {
      list.innerHTML = '<li class="empty">Nothing scanned yet.</li>';
      return;
    }
    list.innerHTML = vehicles
      .map((v) => {
        const pct = (v.observed_fraction * 100).toFixed(0);
        return `<li>
          <div>
            <div class="name">${escapeHtml(v.name)}</div>
            <div class="meta">${v.splats} splats · ${pct}% observed · ${v.observations} obs</div>
          </div>
          <button class="btn mini" data-view="${escapeHtml(v.folder)}">View 3D</button>
        </li>`;
      })
      .join("");
  } catch {
    list.innerHTML = '<li class="empty">Server unreachable.</li>';
  }
}

$("vehicles").addEventListener("click", (e) => {
  const folder = e.target.dataset.view;
  if (folder) showView(folder);
});

async function refreshPending() {
  const box = $("pending");
  try {
    const { pending } = await (await fetch(API + "/merges/pending")).json();
    if (!pending.length) {
      box.innerHTML = "";
      return;
    }
    box.innerHTML =
      "<ul>" +
      pending
        .map(
          (p) => `<li class="pend">
            <div>
              <div class="name">Same vehicle?</div>
              <div class="meta">${escapeHtml(p.data.duplicate)} ≈ ${escapeHtml(
            p.data.primary
          )} (${p.data.score.toFixed(3)})</div>
            </div>
            <span style="display:flex;gap:6px">
              <button class="btn mini ok" data-approve="${p.event_id}">Merge</button>
              <button class="btn mini no" data-reject="${p.event_id}">No</button>
            </span>
          </li>`
        )
        .join("") +
      "</ul>";
  } catch {
    box.innerHTML = "";
  }
}

$("pending").addEventListener("click", async (e) => {
  const approve = e.target.dataset.approve;
  const reject = e.target.dataset.reject;
  if (!approve && !reject) return;
  e.target.disabled = true;
  const id = approve || reject;
  await fetch(`${API}/merges/${id}/${approve ? "approve" : "reject"}`, { method: "POST" });
  setStatus(approve ? "Merged." : "Kept separate.", "ok");
  refresh();
});

const autoMerge = $("autoMerge");
autoMerge.onchange = async () => {
  const form = new FormData();
  form.append("enabled", autoMerge.checked ? "true" : "false");
  await fetch(API + "/settings/auto-merge", { method: "POST", body: form });
  syncHint(autoMerge.checked);
};

function syncHint(on) {
  $("amHint").textContent = on
    ? "On — duplicates merge automatically."
    : "Off — duplicates are flagged for your approval.";
}

async function loadSettings() {
  try {
    const s = await (await fetch(API + "/settings/auto-merge")).json();
    autoMerge.checked = s.auto_merge;
    syncHint(s.auto_merge);
  } catch {}
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

loadSettings();
refresh();
setInterval(refresh, 5000);
