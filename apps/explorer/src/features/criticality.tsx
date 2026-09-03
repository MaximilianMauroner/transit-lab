import { useEffect, useState } from "react";
import { api, formatCount, shortId } from "../../../../packages/api-client/src/index.ts";
import { Card, DataTable, ErrorCard, Field, inputClassName, LoadingState, PageIntro, SectionHeading } from "../../../../packages/ui/src/react.tsx";

export function CriticalityView() {
  const [snapshots, setSnapshots] = useState<any[] | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [resultError, setResultError] = useState<unknown>(null);
  useEffect(() => {
    let active = true;
    api("/api/public/snapshots").then((value) => { if (!active) return; setSnapshots(value); setSelectedId(value[0]?.id || ""); }).catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (!selectedId) return;
    let active = true;
    setData(null);
    setResultError(null);
    api(`/api/public/criticality?snapshotId=${encodeURIComponent(selectedId)}`).then((value) => active && setData(value)).catch((value) => active && setResultError(value));
    return () => { active = false; };
  }, [selectedId]);
  if (error) return <ErrorCard error={error} />;
  if (!snapshots) return <LoadingState message="Loading published snapshots…" />;
  if (!snapshots.length) return <ErrorCard error={new Error("No published snapshots are available.")} />;
  const predictions = data?.predictions || [];
  return <><PageIntro kicker="Published model output" title="Line criticality" description="Rust-produced scores and percentiles, ranked for inspection." action={<Field label="Snapshot" htmlFor="criticality-snapshot"><select id="criticality-snapshot" className={inputClassName} value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>{snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.sourceName || snapshot.networkId || shortId(snapshot.id)} · {snapshot.serviceDate}</option>)}</select></Field>} /><Card className="overflow-hidden"><SectionHeading title="Criticality ranking" copy={resultError ? "Unable to load this publication." : `${formatCount(predictions.length)} published line predictions`} action={data ? <span className="font-mono text-[11px] text-faint">{shortId(data.modelId)}</span> : null} />{resultError ? <p className="px-5 py-8 text-sm text-coral">{resultError instanceof Error ? resultError.message : String(resultError)}</p> : !data ? <LoadingState message="Loading Rust-produced criticality…" /> : <DataTable headers={["Line", data.metricNames?.[0] || "Primary score", "Percentile", "Structural uniqueness", "Uncertainty"]} hasRows={predictions.length > 0} emptyTitle="No predictions" emptyMessage="This publication does not include line predictions.">{predictions.map((prediction, index) => <tr key={`${prediction.line || prediction.lineName}-${index}`}><td>{prediction.lineName || prediction.line}</td><td className="font-medium text-mint">{Number(prediction.metrics?.[0] || 0).toFixed(3)}</td><td>{Number(prediction.metricPercentiles?.[0] || 0).toFixed(3)}</td><td>{Number(prediction.structuralUniqueness || 0).toFixed(3)}</td><td>{Number(prediction.uncertainty || 0).toFixed(3)}</td></tr>)}</DataTable>}</Card></>;
}
