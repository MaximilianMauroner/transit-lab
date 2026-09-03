import { useEffect, useMemo, useRef, useState } from "react";
import { api, formatCount, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { GraphRenderer, modeColor, transitMode, validateNetwork } from "../../../../../../packages/visualizations/src/index.ts";
import { Button, Card, EmptyState, ErrorCard, Field, inputClassName, LoadingState, PageIntro, SectionHeading, cn } from "../../components/ui.tsx";

export function NetworkView() {
  const [snapshots, setSnapshots] = useState<any[] | null>(null);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState("");
  const [network, setNetwork] = useState<any>(null);
  const [loadingNetwork, setLoadingNetwork] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [networkError, setNetworkError] = useState("");
  const [rendererError, setRendererError] = useState("");
  const [search, setSearch] = useState("");
  const [heightMode, setHeightMode] = useState("connectivity");
  const [showStations, setShowStations] = useState(true);
  const [showTransfers, setShowTransfers] = useState(true);
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [inspected, setInspected] = useState<{ type: "line" | "station"; index: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<GraphRenderer | null>(null);

  useEffect(() => {
    let active = true;
    api("/api/snapshots")
      .then((value) => {
        if (!active) return;
        setSnapshots(value);
        setSelectedSnapshotId(value[0]?.id || "");
      })
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!snapshots || !canvasRef.current) return;
    try {
      const renderer = new GraphRenderer(canvasRef.current, {
        onHover: (station) => {
          if (station !== null) setInspected({ type: "station", index: station });
        },
        onSelect: (station) => {
          if (station !== null) setInspected({ type: "station", index: station });
        }
      });
      rendererRef.current = renderer;
      setRendererError("");
      return () => { rendererRef.current = null; };
    } catch (value) {
      setRendererError(value instanceof Error ? value.message : String(value));
      return undefined;
    }
  }, [snapshots]);

  useEffect(() => {
    if (!selectedSnapshotId) return;
    let active = true;
    setLoadingNetwork(true);
    setNetworkError("");
    setNetwork(null);
    setSelectedLine(null);
    setInspected(null);
    api(`/api/snapshots/${encodeURIComponent(selectedSnapshotId)}/network`)
      .then((value) => {
        if (!active) return;
        const raw = validateNetwork(value);
        const normalized = {
          ...raw,
          lines: (raw.lines || []).map((line, index) => ({ ...line, index: Number(line.index ?? index) }))
        };
        setNetwork(normalized);
      })
      .catch((value) => active && setNetworkError(value instanceof Error ? value.message : String(value)))
      .finally(() => active && setLoadingNetwork(false));
    return () => { active = false; };
  }, [selectedSnapshotId]);

  const visibleLines = useMemo(() => {
    const term = search.trim().toLowerCase();
    return (network?.lines || []).filter((line, position) => {
      const index = Number(line.index ?? position);
      const name = String(line.display_name || line.canonical_id || index).toLowerCase();
      return !term || name.includes(term) || String(index).includes(term);
    });
  }, [network, search]);

  useEffect(() => {
    if (!rendererRef.current || !network) return;
    rendererRef.current.setNetwork(network);
    rendererRef.current.setHeightMode(heightMode);
    rendererRef.current.setShowStations(showStations);
    rendererRef.current.setShowTransfers(showTransfers);
  }, [network]);

  useEffect(() => {
    rendererRef.current?.setVisibleLines(visibleLines.map((line, position) => Number(line.index ?? position)));
  }, [visibleLines]);

  useEffect(() => { rendererRef.current?.setHeightMode(heightMode); }, [heightMode]);
  useEffect(() => { rendererRef.current?.setShowStations(showStations); }, [showStations]);
  useEffect(() => { rendererRef.current?.setShowTransfers(showTransfers); }, [showTransfers]);
  useEffect(() => { rendererRef.current?.setSelectedLine(selectedLine); }, [selectedLine]);

  if (error) return <ErrorCard error={error} />;
  if (!snapshots) return <LoadingState message="Loading indexed snapshots…" />;
  if (!snapshots.length) {
    return <ErrorCard error={new Error("No compiled snapshot is indexed yet. Produce a Rust snapshot, then refresh the index.")} />;
  }

  const snapshot = snapshots.find((item) => item.id === selectedSnapshotId);
  const inspector = inspected?.type === "line"
    ? <LineDetails network={network} index={inspected.index} />
    : <StationDetails network={network} index={inspected?.index ?? null} />;

  return (
    <>
      <PageIntro
        kicker="Compiled network"
        title={snapshot?.sourceName || snapshot?.networkId || "Network"}
        description={`${snapshot?.serviceDate || "Unknown date"} · ${shortId(selectedSnapshotId)} · ${formatCount(network?.stations?.length || 0)} stations. Choose an indexed service-day snapshot to inspect its stations, lines, and relations.`}
        action={<Button variant="quiet" onClick={() => navigate("data")}>View lineage →</Button>}
      />
      <Card className="overflow-hidden">
        <div className="grid grid-cols-2 gap-2 border-b border-line p-4 sm:grid-cols-4 sm:px-5">
          <NetworkSummary label="Stations" value={formatCount(network?.stations?.length || 0)} />
          <NetworkSummary label="Lines" value={formatCount(network?.lines?.length || 0)} />
          <NetworkSummary label="Transit edges" value={formatCount(network?.transit_edges?.length || 0)} />
          <NetworkSummary label="Interchanges" value={formatCount(network?.interchanges?.length || 0)} />
        </div>
      </Card>
      <div className="mt-4 grid min-h-[625px] grid-cols-1 gap-4 xl:grid-cols-[255px_minmax(0,1fr)_250px]">
        <Card className="overflow-hidden">
          <SectionHeading title="Scene" copy="Filter the visible graph." />
          <div className="grid gap-3.5 p-[18px]">
            <Field label="Snapshot" htmlFor="network-snapshot">
              <select id="network-snapshot" className={inputClassName} value={selectedSnapshotId} onChange={(event) => setSelectedSnapshotId(event.target.value)}>
                {snapshots.map((item) => <option key={item.id} value={item.id}>{item.sourceName || item.networkId || shortId(item.id)} · {item.serviceDate}</option>)}
              </select>
            </Field>
            <Field label="Find a line" htmlFor="network-search">
              <input id="network-search" className={inputClassName} type="search" placeholder="Name or index" value={search} onChange={(event) => setSearch(event.target.value)} />
            </Field>
            <Field label="Height shows" htmlFor="network-height">
              <select id="network-height" className={inputClassName} value={heightMode} onChange={(event) => setHeightMode(event.target.value)}>
                <option value="connectivity">Network role</option>
                <option value="departures">Daily departures</option>
                <option value="service">Service span</option>
              </select>
            </Field>
            <Toggle label="Show stations" checked={showStations} onChange={setShowStations} />
            <Toggle label="Show transfers" checked={showTransfers} onChange={setShowTransfers} />
            <div className="flex gap-2">
              <Button variant="quiet" className="flex-1 px-2 text-[11px]" onClick={() => rendererRef.current?.fit()}>Reset view</Button>
              <Button variant="quiet" className="flex-1 px-2 text-[11px]" onClick={() => rendererRef.current?.topView()}>Top view</Button>
            </div>
          </div>
          <SectionHeading title="Lines" copy="Click to isolate one." />
          <div className="max-h-[492px] overflow-y-auto p-1.5">
            {visibleLines.length ? visibleLines.map((line, position) => {
              const index = Number(line.index ?? position);
              const isSelected = selectedLine === index;
              return (
                <button
                  key={index}
                  className={cn("flex w-full items-center gap-2 rounded-lg border border-transparent px-2.5 py-2 text-left text-[11px] text-muted transition-colors hover:bg-[#182425] hover:text-copy", isSelected && "border-[#3f7665] bg-[#1a3730] text-mint")}
                  type="button"
                  onClick={() => { setSelectedLine(index); setInspected({ type: "line", index }); }}
                >
                  <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: modeColor(line.mode) }} aria-hidden="true" />
                  <span className="truncate">{line.display_name || line.canonical_id || `Line ${index}`}</span>
                  <span className="ml-auto shrink-0 text-[10px] text-faint">{transitMode(line.mode).label}</span>
                </button>
              );
            }) : <EmptyState message="No lines match this search." className="px-2 py-8 text-xs" />}
          </div>
        </Card>
        <div className="relative min-h-[530px] overflow-hidden rounded-xl border border-line bg-[#080e10] xl:min-h-[625px]">
          <canvas ref={canvasRef} className="block h-[530px] w-full xl:h-[625px]" aria-label="Interactive 3D compiled network" />
          {(loadingNetwork || networkError || rendererError) && (
            <div className="absolute inset-0 grid place-items-center bg-[#080e10]/80 px-6 text-center text-xs text-muted">
              {loadingNetwork ? "Loading network…" : networkError || `3D renderer unavailable: ${rendererError}`}
            </div>
          )}
          <div className="pointer-events-none absolute bottom-3.5 left-[18px] text-[10px] text-faint">Drag to orbit · scroll to zoom · click a station to inspect</div>
        </div>
        <Card className="overflow-hidden">
          <SectionHeading title="Inspector" copy="Manifest-backed detail" />
          <div className="p-[18px]">{inspected && network ? inspector : <EmptyState title="Select a line or station" message="Use the scene to inspect the indexed snapshot." className="px-0 py-8" />}</div>
        </Card>
      </div>
    </>
  );
}

function NetworkSummary({ label, value }: { label: string; value: string }) {
  return <div><span className="block text-[10px] text-faint">{label}</span><strong className="mt-1 block text-[15px] text-copy">{value}</strong></div>;
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="flex items-center justify-between text-xs text-muted">
      <span>{label}</span>
      <input className="accent-mint-strong" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function LineDetails({ network, index }: { network: any; index: number }) {
  const line = network?.lines?.find((candidate, position) => Number(candidate.index ?? position) === Number(index));
  if (!line) return <EmptyState title="Line not found" className="px-0 py-8" />;
  const features: Array<[string, unknown]> = [
    ["Mode", transitMode(line.mode).label],
    ["Canonical ID", line.canonical_id || "—"],
    ["Stations", line.station_count ?? line.stationCount ?? "—"],
    ["Patterns", line.pattern_count ?? line.patternCount ?? "—"],
    ["Daily trips", line.daily_trip_count ?? line.dailyTripCount ?? "—"]
  ];
  return <DetailContent kicker={`Line ${index}`} title={line.display_name || line.canonical_id || `Line ${index}`} features={features} />;
}

function StationDetails({ network, index }: { network: any; index: number | null }) {
  if (index === null) return null;
  const station = network?.stations?.[index];
  if (!station) return <EmptyState title="Station not found" className="px-0 py-8" />;
  const features: Array<[string, unknown]> = [
    ["Coordinates", `${Number(station.latitude).toFixed(4)}, ${Number(station.longitude).toFixed(4)}`],
    ["Lines", station.line_count ?? "—"],
    ["Patterns", station.pattern_count ?? "—"],
    ["Daily departures", station.daily_departures ?? "—"],
    ["Terminal", station.terminal ? "Yes" : "No"]
  ];
  return <DetailContent kicker={`Station ${index}`} title={station.name || station.canonical_id || `Station ${index}`} features={features} />;
}

function DetailContent({ kicker, title, features }: { kicker: string; title: string; features: Array<[string, unknown]> }) {
  return (
    <>
      <p className="mb-1.5 text-[10px] font-extrabold uppercase tracking-[0.14em] text-mint">{kicker}</p>
      <h3 className="mb-4 mt-0 text-base font-semibold text-copy">{title}</h3>
      <dl className="grid gap-2.5">
        {features.map(([label, value]) => <div key={label} className="flex justify-between gap-3 border-b border-[#203031] pb-2.5"><dt className="text-[10px] text-faint">{label}</dt><dd className="m-0 text-right text-[11px] text-copy">{String(value)}</dd></div>)}
      </dl>
    </>
  );
}
