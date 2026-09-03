import { useEffect, useRef, useState } from "react";
import { api, formatCount, shortId } from "../../../../packages/api-client/src/index.ts";
import { GraphRenderer, validateNetwork } from "../../../../packages/visualizations/src/index.ts";
import { Card, EmptyState, ErrorCard, Field, inputClassName, LoadingState, PageIntro, SectionHeading } from "../../../../packages/ui/src/react.tsx";

export function NetworkView() {
  const [snapshots, setSnapshots] = useState<any[] | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [network, setNetwork] = useState<any>(null);
  const [loadingNetwork, setLoadingNetwork] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [networkError, setNetworkError] = useState("");
  const [rendererError, setRendererError] = useState("");
  const [inspected, setInspected] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<GraphRenderer | null>(null);
  useEffect(() => { let active = true; api("/api/public/snapshots").then((value) => { if (!active) return; setSnapshots(value); setSelectedId(value[0]?.id || ""); }).catch((value) => active && setError(value)); return () => { active = false; }; }, []);
  useEffect(() => { if (!snapshots || !canvasRef.current) return; try { const renderer = new GraphRenderer(canvasRef.current, { onHover: (station) => station !== null && setInspected(station), onSelect: (station) => station !== null && setInspected(station) }); rendererRef.current = renderer; setRendererError(""); return () => { rendererRef.current = null; }; } catch (value) { setRendererError(value instanceof Error ? value.message : String(value)); return undefined; } }, [snapshots]);
  useEffect(() => { if (!selectedId) return; let active = true; setLoadingNetwork(true); setNetworkError(""); setNetwork(null); setInspected(null); api(`/api/public/snapshots/${encodeURIComponent(selectedId)}/network`).then((value) => { if (!active) return; const raw = validateNetwork(value); const next = { ...raw, lines: (raw.lines || []).map((line, index) => ({ ...line, index: Number(line.index ?? index) })) }; setNetwork(next); }).catch((value) => active && setNetworkError(value instanceof Error ? value.message : String(value))).finally(() => active && setLoadingNetwork(false)); return () => { active = false; }; }, [selectedId]);
  useEffect(() => { if (!rendererRef.current || !network) return; rendererRef.current.setNetwork(network); }, [network]);
  if (error) return <ErrorCard error={error} />;
  if (!snapshots) return <LoadingState message="Loading published snapshots…" />;
  if (!snapshots.length) return <ErrorCard error={new Error("No published snapshots are available.")} />;
  const snapshot = snapshots.find((item) => item.id === selectedId);
  return <><PageIntro kicker="Published network" title={snapshot?.sourceName || "Network snapshot"} description={`${snapshot?.serviceDate || "Unknown date"} · ${shortId(selectedId)} · ${formatCount(network?.stations?.length || 0)} stations. Read-only station, line, pattern, and relation inspection.`} action={<Field label="Snapshot" htmlFor="snapshot-select"><select id="snapshot-select" className={inputClassName} value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>{snapshots.map((item) => <option key={item.id} value={item.id}>{item.sourceName || item.networkId || shortId(item.id)} · {item.serviceDate}</option>)}</select></Field>} /><div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(250px,1fr)]"><div className="relative min-h-[360px] overflow-hidden rounded-xl border border-line bg-[#080e10] xl:min-h-[520px]"><canvas ref={canvasRef} className="block h-[360px] w-full xl:h-[520px]" aria-label="Published transit network" />{(loadingNetwork || networkError || rendererError) ? <div className="absolute inset-0 grid place-items-center bg-[#0b1114]/80 px-6 text-center text-xs text-muted">{loadingNetwork ? "Loading network…" : networkError || `3D renderer unavailable: ${rendererError}`}</div> : null}</div><Card className="overflow-hidden"><SectionHeading title="Inspector" copy="Select a station in the published network." /><div className="p-[18px]">{inspected !== null && network ? <StationDetails network={network} index={inspected} /> : <EmptyState message="Select a station in the published network." className="px-0 py-8" />}</div></Card></div></>;
}

function StationDetails({ network, index }: { network: any; index: number }) {
  const station = network?.stations?.[index];
  if (!station) return <EmptyState title="Station not found" className="px-0 py-8" />;
  return <><p className="mb-1.5 text-[10px] font-extrabold uppercase tracking-[0.14em] text-mint">Station {index}</p><h3 className="mb-4 mt-0 text-base font-semibold text-copy">{station.name || station.canonical_id || `Station ${index}`}</h3><dl className="grid gap-2.5">{[["Coordinates", `${Number(station.latitude).toFixed(4)}, ${Number(station.longitude).toFixed(4)}`], ["Lines", station.line_count ?? "—"], ["Daily departures", station.daily_departures ?? "—"]].map(([label, value]) => <div key={label} className="flex justify-between gap-3 border-b border-[#203031] pb-2.5"><dt className="text-[10px] text-faint">{label}</dt><dd className="m-0 text-right text-[11px] text-copy">{String(value)}</dd></div>)}</dl></>;
}
