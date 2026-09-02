import { api, escapeHtml, formatDate, shortId, statusClass } from "../../api.js";
import { errorCard, loading, sectionHeading, status } from "../../components/ui.js";

export async function renderRuns(container) {
  container.innerHTML = loading("Loading run ledger…");
  try {
    const [runs, snapshots, models] = await Promise.all([api("/api/runs"), api("/api/snapshots"), api("/api/models")]);
    container.innerHTML = shell(snapshots, models);
    const list = container.querySelector("#run-list");
    const detail = container.querySelector("#run-detail");
    const form = container.querySelector("#run-form");
    let selectedId = null;
    let stream = null;

    const showDetail = async (runId) => {
      selectedId = runId;
      list.querySelectorAll("tr").forEach((row) => row.classList.toggle("run-selected", row.dataset.runId === runId));
      try {
        const current = await api(`/api/runs/${encodeURIComponent(runId)}`);
        detail.innerHTML = runDetail(current);
        detail.querySelector("#cancel-run")?.addEventListener("click", async () => {
          await api(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
          await refreshList();
          await showDetail(runId);
        });
        stream?.close();
        if (["queued", "claimed", "running"].includes(current.status)) {
          stream = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events?after=${Math.max(-1, (current.events || []).at(-1)?.seq ?? -1)}`);
          stream.onmessage = async () => { await refreshList(); await showDetail(runId); };
          stream.onerror = () => stream.close();
        }
      } catch (error) { detail.innerHTML = errorCard(error); }
    };

    const refreshList = async () => {
      const currentRuns = await api("/api/runs");
      list.innerHTML = currentRuns.map((run) => `<tr class="run-row${run.id === selectedId ? " run-selected" : ""}" data-run-id="${escapeHtml(run.id)}"><td>${escapeHtml(run.kind)}</td><td class="mono">${escapeHtml(shortId(run.id))}</td><td>${status(run.status)}</td><td>${escapeHtml(run.currentStep || "—")}</td><td>${formatDate(run.createdAt)}</td></tr>`).join("") || `<tr><td colspan="5" class="muted">No runs submitted.</td></tr>`;
      list.querySelectorAll("[data-run-id]").forEach((row) => row.addEventListener("click", () => showDetail(row.dataset.runId)));
    };

    const updateType = () => {
      const kind = form.querySelector("#run-kind").value;
      form.querySelectorAll("[data-kind]").forEach((field) => { field.hidden = field.dataset.kind !== kind; });
    };
    form.querySelector("#run-kind").addEventListener("change", updateType);
    updateType();
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const kind = form.querySelector("#run-kind").value;
      const spec = kind === "infer"
        ? { kind, modelId: form.querySelector("#run-model").value, snapshotId: form.querySelector("#run-snapshot").value }
        : { kind, snapshotId: form.querySelector("#simulate-snapshot").value, simulationConfig: "default" };
      const submit = form.querySelector("button[type=submit]");
      submit.disabled = true;
      try {
        const created = await api("/api/runs", { method: "POST", body: JSON.stringify({ spec }) });
        await refreshList();
        await showDetail(created.id);
      } catch (error) {
        form.querySelector("#run-form-error").textContent = error.message;
        form.querySelector("#run-form-error").hidden = false;
      } finally { submit.disabled = false; }
    });
    await refreshList();
    if (runs[0]) await showDetail(runs[0].id);
  } catch (error) { container.innerHTML = errorCard(error); }
}

function shell(snapshots, models) {
  const snapshotOptions = snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId || shortId(snapshot.id))} · ${escapeHtml(snapshot.serviceDate)}</option>`).join("");
  const modelOptions = models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.version)} · ${escapeHtml(shortId(model.id))}</option>`).join("");
  return `<div class="page-intro"><div><p class="eyebrow">Operational ledger</p><h2>Runs have a contract.</h2><p>Submit a known run specification. The worker translates it to an allow-listed Rust argv array, records JSONL events, and leaves human logs as diagnostics only.</p></div><span class="read-only-badge"><span class="live-dot"></span> No arbitrary shell commands</span></div><div class="run-layout"><section class="card form-card"><h3>Submit a run</h3><p>Choose a typed operation. Inputs are selected from indexed artifacts.</p><form id="run-form" class="form-stack"><div class="field"><label for="run-kind">Operation</label><select class="select" id="run-kind"><option value="infer">Infer criticality</option><option value="simulate-criticality">Simulate criticality</option></select></div><div class="field" data-kind="infer"><label for="run-model">Model</label><select class="select" id="run-model">${modelOptions || `<option value="">No model indexed</option>`}</select></div><div class="field" data-kind="infer"><label for="run-snapshot">Snapshot</label><select class="select" id="run-snapshot">${snapshotOptions || `<option value="">No snapshot indexed</option>`}</select></div><div class="field" data-kind="simulate-criticality"><label for="simulate-snapshot">Snapshot</label><select class="select" id="simulate-snapshot">${snapshotOptions || `<option value="">No snapshot indexed</option>`}</select></div><p id="run-form-error" class="danger" hidden></p><div class="form-actions"><button class="button button-primary" type="submit">Queue run</button></div></form></section><section class="card section-card" style="margin-top:0">${sectionHeading("Run history", "Claimed atomically by one Studio worker.", "")}<div class="table-wrap"><table class="data-table"><thead><tr><th>Kind</th><th>Run</th><th>Status</th><th>Step</th><th>Created</th></tr></thead><tbody id="run-list"></tbody></table></div></section></div><section class="card section-card" id="run-detail" style="min-height:180px">${sectionHeading("Selected run", "Replayable structured events appear here.", "")}<div class="empty"><p>Select a run from the ledger.</p></div></section>`;
}

function runDetail(run) {
  const events = (run.events || []).map((event) => `<div class="event-line"><span>#${escapeHtml(event.seq)}</span><strong>${escapeHtml(event.type)}</strong><span>${escapeHtml(event.step || event.message || JSON.stringify(event))}</span></div>`).join("") || `<div class="muted">No events yet.</div>`;
  const canCancel = ["queued", "claimed", "running"].includes(run.status);
  return `${sectionHeading("Selected run", `${escapeHtml(run.kind)} · ${escapeHtml(formatDate(run.createdAt))}`, canCancel ? `<button class="button button-quiet" id="cancel-run" type="button">Request cancellation</button>` : "")}<div class="card-pad"><div style="display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:12px"><span>Run <strong class="mono">${escapeHtml(shortId(run.id))}</strong></span><span>${status(run.status)}</span><span>Step <strong>${escapeHtml(run.currentStep || "—")}</strong></span><span>Progress <strong>${escapeHtml(`${run.progress?.completed || 0} / ${run.progress?.total || 0} ${run.progress?.unit || ""}`)}</strong></span></div><div class="run-events">${events}</div>${run.error ? `<p class="danger" style="font-size:11px">${escapeHtml(run.error.code)}: ${escapeHtml(run.error.message)}</p>` : ""}</div>`;
}
