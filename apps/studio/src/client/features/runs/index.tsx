import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { api, formatDate, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { Button, Card, DataTable, EmptyState, ErrorCard, Field, inputClassName, LiveBadge, LoadingState, Mono, PageIntro, SectionHeading, StatusBadge, cn } from "../../components/ui.tsx";

const TERMINAL_STATES = new Set(["succeeded", "failed", "cancelled", "orphaned"]);
const ACTIVE_STATES = new Set(["queued", "claimed", "starting", "running", "checkpointing", "interrupted"]);

export function RunsView() {
  const [runs, setRuns] = useState<any[] | null>(null);
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selectedRun, setSelectedRun] = useState<any>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<unknown>(null);
  const [error, setError] = useState<unknown>(null);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");
  const streamRef = useRef<EventSource | null>(null);

  async function loadList() {
    const nextRuns = await api("/api/runs");
    setRuns(nextRuns);
    return nextRuns;
  }

  async function loadDetail(runId: string) {
    if (!runId) return null;
    setLoadingDetail(true);
    setDetailError(null);
    try {
      const current = await api(runPath(runId));
      setSelectedRun(current);
      return current;
    } catch (value) {
      setDetailError(value);
      return null;
    } finally {
      setLoadingDetail(false);
    }
  }

  useEffect(() => {
    let active = true;
    Promise.all([api("/api/runs"), api("/api/snapshots"), api("/api/models"), api("/api/datasets")])
      .then(([nextRuns, nextSnapshots, nextModels, nextDatasets]) => {
        if (!active) return;
        setRuns(nextRuns);
        setSnapshots(nextSnapshots);
        setModels(nextModels);
        setDatasets(nextDatasets);
        setSelectedId(nextRuns[0]?.id || "");
      })
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setSelectedRun(null);
      return;
    }
    let active = true;
    void loadDetail(selectedId).then((current) => {
      if (!active || !current) return;
      const state = current.status || current.observedState;
      if (!ACTIVE_STATES.has(state) || TERMINAL_STATES.has(state)) return;
      const source = new EventSource(`${runPath(selectedId, "/events")}?after=${Math.max(-1, (current.events || []).at(-1)?.seq ?? -1)}`);
      streamRef.current = source;
      source.onmessage = () => {
        void Promise.all([loadList(), api(runPath(selectedId))]).then(([, updated]) => {
          if (active) setSelectedRun(updated);
        }).catch(() => undefined);
      };
      source.onerror = () => source.close();
    });
    return () => {
      active = false;
      streamRef.current?.close();
      streamRef.current = null;
    };
  }, [selectedId]);

  if (error) return <ErrorCard error={error} />;
  if (!runs) return <LoadingState message="Loading run ledger…" />;

  async function submitRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const kind = String(form.get("kind") || "infer");
    const spec = kind === "infer"
      ? { kind, modelId: String(form.get("modelId") || ""), snapshotId: String(form.get("snapshotId") || "") }
      : kind === "train"
        ? trainingSpec(form)
        : { kind, snapshotId: String(form.get("simulationSnapshotId") || ""), simulationConfig: "default" };
    setSubmitting(true);
    setFormError("");
    try {
      const created = await api("/api/runs", { method: "POST", body: JSON.stringify({ spec }) });
      await loadList();
      setSelectedId(created.id);
    } catch (value) {
      setFormError(value instanceof Error ? value.message : String(value));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <PageIntro
        kicker="Operational ledger"
        title="Runs have a contract."
        description="Submit a known run specification. The worker translates it to an allow-listed Rust argv array, records JSONL events, and leaves human logs as diagnostics only."
        action={<LiveBadge>No arbitrary shell commands</LiveBadge>}
      />
      <div className="grid grid-cols-1 gap-4 2xl:grid-cols-[minmax(280px,350px)_minmax(0,1fr)]">
        <RunForm snapshots={snapshots} models={models} datasets={datasets} onSubmit={submitRun} submitting={submitting} formError={formError} />
        <Card className="overflow-hidden">
          <SectionHeading title="Run history" copy="Claimed atomically by one Studio worker." />
          <DataTable headers={["Kind", "Run", "Status", "Phase / progress", "Created"]} hasRows={runs.length > 0} emptyTitle="No runs submitted" emptyMessage="Queue a typed operation to see it in the ledger.">
            {runs.map((run) => {
              const progress = run.progress?.total > 0
                ? `${run.progress.completed || 0}/${run.progress.total} ${run.progress.unit || ""}`
                : run.phase || run.currentStep || "—";
              const selected = run.id === selectedId;
              return (
                <tr
                  key={run.id}
                  className={cn("cursor-pointer", selected && "bg-[#17312b] outline outline-1 -outline-offset-1 outline-[#3f7665]")}
                  onClick={() => setSelectedId(run.id)}
                  onKeyDown={(event: KeyboardEvent<HTMLTableRowElement>) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedId(run.id); } }}
                  tabIndex={0}
                  role="button"
                >
                  <td>{run.kind}</td>
                  <td><Mono>{shortId(run.id)}</Mono></td>
                  <td><StatusBadge value={run.status} /></td>
                  <td>{progress}</td>
                  <td>{formatDate(run.createdAt)}</td>
                </tr>
              );
            })}
          </DataTable>
        </Card>
      </div>
      <Card className="mt-4 min-h-[180px] overflow-hidden">
        <SectionHeading
          title="Selected run"
          copy="Replayable structured events appear here."
          action={selectedRun ? <RunActions run={selectedRun} onAction={(action) => performRunAction(action, selectedRun.id)} /> : null}
        />
        {loadingDetail ? <LoadingState message="Loading selected run…" /> : detailError ? <ErrorCard error={detailError} /> : selectedRun ? <RunDetail run={selectedRun} onAction={(action) => performRunAction(action, selectedRun.id)} /> : <EmptyState message="Select a run from the ledger." />}
      </Card>
    </>
  );

  async function performRunAction(action: string, runId: string) {
    try {
      await api(runPath(runId, `/${action}`), {
        method: "POST",
        ...(action === "resume" || action === "fork" ? { body: "{}" } : {})
      });
      await loadList();
      await loadDetail(runId);
    } catch (value) {
      setDetailError(value);
    }
  }
}

function RunForm({ snapshots, models, datasets, onSubmit, submitting, formError }: { snapshots: any[]; models: any[]; datasets: any[]; onSubmit: (event: FormEvent<HTMLFormElement>) => void; submitting: boolean; formError: string }) {
  const [kind, setKind] = useState("infer");
  const snapshotOptions = snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.sourceName || snapshot.networkId || shortId(snapshot.id)} · {snapshot.serviceDate}</option>);
  const modelOptions = models.map((model) => <option key={model.id} value={model.id}>{model.version} · {shortId(model.id)}</option>);
  const datasetOptions = datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.id} · {dataset.featureSchema || "dataset"}</option>);
  return (
    <Card className="p-5">
      <h3 className="m-0 text-[15px] font-semibold text-copy">Submit a run</h3>
      <p className="mb-4 mt-1 text-xs leading-relaxed text-muted">Choose a typed operation. Inputs are selected from indexed artifacts.</p>
      <form className="grid gap-3" onSubmit={onSubmit}>
        <Field label="Operation" htmlFor="run-kind">
          <select id="run-kind" name="kind" className={inputClassName} value={kind} onChange={(event) => setKind(event.target.value)}>
            <option value="infer">Infer criticality</option>
            <option value="simulate-criticality">Simulate criticality</option>
            <option value="train">Train model</option>
          </select>
        </Field>
        {kind === "infer" ? <>
          <Field label="Model" htmlFor="run-model"><select id="run-model" name="modelId" className={inputClassName}>{modelOptions.length ? modelOptions : <option value="">No model indexed</option>}</select></Field>
          <Field label="Snapshot" htmlFor="run-snapshot"><select id="run-snapshot" name="snapshotId" className={inputClassName}>{snapshotOptions.length ? snapshotOptions : <option value="">No snapshot indexed</option>}</select></Field>
        </> : null}
        {kind === "simulate-criticality" ? <Field label="Snapshot" htmlFor="simulate-snapshot"><select id="simulate-snapshot" name="simulationSnapshotId" className={inputClassName}>{snapshotOptions.length ? snapshotOptions : <option value="">No snapshot indexed</option>}</select></Field> : null}
        {kind === "train" ? <TrainingFields datasetOptions={datasetOptions} /> : null}
        <p className="min-h-4 text-xs text-coral" aria-live="polite">{formError}</p>
        <div className="flex justify-end"><Button variant="primary" type="submit" disabled={submitting}>{submitting ? "Queueing…" : "Queue run"}</Button></div>
      </form>
    </Card>
  );
}

function TrainingFields({ datasetOptions }: { datasetOptions: ReactNode[] }) {
  return (
    <div className="grid gap-3">
      <Field label="Dataset" htmlFor="train-dataset"><select id="train-dataset" name="datasetId" className={inputClassName}>{datasetOptions.length ? datasetOptions : <option value="">No dataset indexed</option>}</select></Field>
      <Field label="Model config" htmlFor="train-model-config"><input id="train-model-config" name="modelConfig" className={inputClassName} defaultValue="configs/models/multitask-v1.yaml" /></Field>
      <Field label="Seed" htmlFor="train-seed"><input id="train-seed" name="seed" className={inputClassName} type="number" min="0" defaultValue="7" /></Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Checkpoint steps" htmlFor="train-checkpoint-steps"><input id="train-checkpoint-steps" name="checkpointEverySteps" className={inputClassName} type="number" min="1" placeholder="500" /></Field>
        <Field label="Checkpoint seconds" htmlFor="train-checkpoint-seconds"><input id="train-checkpoint-seconds" name="checkpointEverySeconds" className={inputClassName} type="number" min="1" placeholder="900" /></Field>
        <Field label="Max attempt seconds" htmlFor="train-max-attempt"><input id="train-max-attempt" name="maxAttemptSeconds" className={inputClassName} type="number" min="1" placeholder="Optional" /></Field>
        <Field label="Checkpoint grace seconds" htmlFor="train-checkpoint-grace"><input id="train-checkpoint-grace" name="checkpointGraceSeconds" className={inputClassName} type="number" min="1" placeholder="300" /></Field>
      </div>
      <Field label="Allowed days (optional)" htmlFor="train-window-days"><input id="train-window-days" name="allowedDays" className={inputClassName} placeholder="monday,tuesday" /></Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Window start" htmlFor="train-window-start"><input id="train-window-start" name="windowStart" className={inputClassName} type="time" /></Field>
        <Field label="Window end" htmlFor="train-window-end"><input id="train-window-end" name="windowEnd" className={inputClassName} type="time" /></Field>
      </div>
      <Field label="Window timezone" htmlFor="train-window-timezone"><input id="train-window-timezone" name="windowTimezone" className={inputClassName} placeholder="Europe/Rome" /></Field>
    </div>
  );
}

function RunActions({ run, onAction }: { run: any; onAction: (action: string) => Promise<void> }) {
  const statusValue = run.status || run.observedState;
  const terminal = TERMINAL_STATES.has(statusValue);
  const actions = [
    !terminal && run.desiredState !== "paused" && run.desiredState !== "cancelled" ? ["pause", "Pause after checkpoint", "quiet"] : null,
    !terminal && (run.desiredState === "paused" || ["paused", "interrupted"].includes(statusValue)) ? ["resume", "Resume", "primary"] : null,
    !terminal && Array.isArray(run.checkpoints) && run.checkpoints.length > 0 ? ["fork", "Fork latest checkpoint", "quiet"] : null,
    !terminal && run.desiredState !== "cancelled" ? ["cancel", "Cancel permanently", "danger"] : null
  ].filter(Boolean) as Array<[string, string, "quiet" | "primary" | "danger"]>;
  return <div className="flex flex-wrap justify-end gap-2">{actions.map(([action, label, variant]) => <ActionButton key={action} label={label} variant={variant} onAction={() => onAction(action)} />)}</div>;
}

function ActionButton({ label, variant, onAction }: { label: string; variant: "quiet" | "primary" | "danger"; onAction: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  return <Button variant={variant} type="button" disabled={busy} onClick={async () => { setBusy(true); try { await onAction(); } finally { setBusy(false); } }}>{busy ? "Working…" : label}</Button>;
}

function RunDetail({ run, onAction }: { run: any; onAction: (action: string) => Promise<void> }) {
  const statusValue = run.status || run.observedState;
  const estimate = run.estimate || {};
  const schedule = run.schedule || {};
  const activeWindow = schedule.allowedWindows?.length ? `${schedule.allowedWindows.length} allowed window${schedule.allowedWindows.length === 1 ? "" : "s"}` : "unrestricted";
  const events = (run.events || []).slice().reverse();
  const attempts = run.attempts || [];
  const checkpoints = (run.checkpoints || []).slice().reverse();
  return (
    <div className="p-5">
      <div className="mb-4 flex flex-wrap items-center gap-4 text-[11px] text-muted">
        <span>Run <Mono className="text-copy">{shortId(run.id)}</Mono></span>
        <StatusBadge value={statusValue} />
        <span>Desired <strong className="ml-1 font-medium text-copy">{run.desiredState || "running"}</strong></span>
        <span>Phase <strong className="ml-1 font-medium text-copy">{run.phase || run.currentStep || "—"}</strong></span>
        <span>Step <strong className="ml-1 font-medium text-copy">{run.globalStep || 0}</strong></span>
      </div>
      <div className="mb-5 grid grid-cols-2 gap-2 lg:grid-cols-4">
        <RunMetric label="Progress" value={`${run.progress?.completed || 0} / ${run.progress?.total || 0} ${run.progress?.unit || ""}`} />
        <RunMetric label="Throughput" value={estimate.throughput > 0 ? `${Number(estimate.throughput).toFixed(2)} / sec` : "—"} />
        <RunMetric label="ETA" value={estimate.etaSeconds === null || estimate.etaSeconds === undefined ? "—" : formatDuration(estimate.etaSeconds)} />
        <RunMetric label="Compute" value={formatDuration(estimate.measuredComputeSeconds || run.totalComputeSeconds || 0)} />
      </div>
      <div className="mb-5 flex flex-wrap gap-x-5 gap-y-2 text-[11px] text-muted">
        <span>Schedule <strong className="ml-1 font-medium text-copy">{activeWindow}</strong></span>
        <span>Next window <strong className="ml-1 font-medium text-copy">{run.resumeNotBefore ? formatDate(run.resumeNotBefore) : "when eligible"}</strong></span>
        <span>Attempts <strong className="ml-1 font-medium text-copy">{run.attempts?.length || 0}</strong></span>
        <span>Latest checkpoint <strong className="ml-1 font-medium text-copy">{run.latestCheckpointId || "—"}</strong></span>
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <section>
          <h4 className="mb-2.5 mt-0 text-xs font-semibold text-copy">Attempt timeline</h4>
          <div className="grid gap-2">
            {attempts.length ? attempts.map((attempt) => <div className="flex gap-2 border-b border-[#203031] py-2" key={attempt.ordinal}><span className="mt-1 size-1.5 shrink-0 rounded-full bg-mint-strong" /><div><strong className="block text-[11px] text-copy">Attempt {attempt.ordinal}</strong><span className="mt-0.5 block text-[10px] text-muted">{attempt.status} · {attempt.exitReason || "active"}</span><small className="mt-0.5 block text-[10px] text-faint">{formatDate(attempt.startedAt)}{attempt.finishedAt ? ` → ${formatDate(attempt.finishedAt)}` : ""}</small></div></div>) : <p className="text-xs text-muted">No execution attempts yet.</p>}
          </div>
        </section>
        <section>
          <h4 className="mb-2.5 mt-0 text-xs font-semibold text-copy">Committed checkpoints</h4>
          <div className="grid gap-2">
            {checkpoints.length ? checkpoints.map((checkpoint) => <div className="grid grid-cols-[65px_85px_minmax(0,1fr)] gap-2 border-b border-[#203031] py-2 text-[10px] text-muted" key={checkpoint.id || checkpoint.globalStep}><span className="font-mono text-mint">step {checkpoint.globalStep}</span><span>{checkpoint.phase}</span><span className="truncate font-mono">{checkpoint.localPath}</span></div>) : <p className="text-xs text-muted">No committed checkpoints yet.</p>}
          </div>
        </section>
      </div>
      <h4 className="mb-2.5 mt-5 text-xs font-semibold text-copy">Structured events</h4>
      <div className="max-h-[230px] overflow-auto rounded-lg border border-line bg-ink p-3 font-mono text-[10px]">
        {events.length ? events.map((event) => <div className="grid grid-cols-[32px_130px_minmax(0,1fr)] gap-2 border-b border-[#1b292a] py-1.5 text-muted last:border-b-0" key={`${event.seq}-${event.type}`}><span>#{event.seq}</span><strong className="font-semibold text-mint">{event.type}</strong><span>{event.step ?? event.globalStep ?? event.message ?? event.reason ?? event.path ?? JSON.stringify(event)}</span></div>) : <span className="text-muted">No events yet.</span>}
      </div>
      {run.error ? <p className="text-[11px] text-coral">{run.error.code}: {run.error.message}</p> : null}
    </div>
  );
}

function RunMetric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg border border-line bg-panel-soft p-3"><span className="block text-[10px] text-faint">{label}</span><strong className="mt-1.5 block text-[15px] text-mint">{value}</strong></div>;
}

function trainingSpec(form: FormData) {
  const runtime: Record<string, unknown> = {
    checkpointEverySteps: optionalNumber(form.get("checkpointEverySteps")),
    checkpointEverySeconds: optionalNumber(form.get("checkpointEverySeconds")),
    maxAttemptSeconds: optionalNumber(form.get("maxAttemptSeconds")),
    checkpointGraceSeconds: optionalNumber(form.get("checkpointGraceSeconds"))
  };
  const days = String(form.get("allowedDays") || "").trim();
  const start = String(form.get("windowStart") || "");
  const end = String(form.get("windowEnd") || "");
  const timezone = String(form.get("windowTimezone") || "").trim();
  if (days || start || end || timezone) {
    runtime.allowedWindows = [{
      days: (days || "monday,tuesday,wednesday,thursday,friday").split(",").map((day) => day.trim()).filter(Boolean),
      start: start || "22:00",
      end: end || "08:00",
      timezone: timezone || "UTC"
    }];
  }
  return {
    kind: "train",
    datasetId: String(form.get("datasetId") || ""),
    modelConfig: String(form.get("modelConfig") || "configs/models/multitask-v1.yaml"),
    seed: Number(form.get("seed") || 7),
    runtime: Object.fromEntries(Object.entries(runtime).filter(([, value]) => value !== undefined))
  };
}

function optionalNumber(value: FormDataEntryValue | null) {
  if (value === "" || value === null || value === undefined) return undefined;
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : undefined;
}

function runPath(runId: string, suffix = "") {
  return `/api/control/runs/${encodeURIComponent(runId)}${suffix}`;
}

function formatDuration(seconds: unknown) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value}s`;
  const minutes = Math.floor(value / 60);
  if (minutes < 60) return `${minutes}m ${value % 60}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}
