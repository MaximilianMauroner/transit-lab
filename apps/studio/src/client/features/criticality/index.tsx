import { useEffect, useMemo, useState } from "react";
import { api, formatCount, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { Button, Card, DataTable, ErrorCard, Field, LoadingState, PageIntro, SectionHeading, StatCard, inputClassName } from "../../components/ui.tsx";
import {
  formatMetricValue,
  metricLabel,
  metricValue,
  normalizePredictionFile,
  primaryMetricName
} from "./normalize.ts";

const sortOptions = [
  ["accessibility_auc_loss", "Accessibility loss"],
  ["unreachable_share", "Unreachable share"],
  ["mean_delay_reachable_seconds", "Mean delay"],
  ["structural_uniqueness", "Structural uniqueness"],
  ["line", "Line"]
];

export function CriticalityView() {
  const [inferences, setInferences] = useState<any[] | null>(null);
  const [dataset, setDataset] = useState<any>(null);
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("accessibility_auc_loss");
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [datasetError, setDatasetError] = useState("");

  useEffect(() => {
    let active = true;
    api("/api/inferences")
      .then((value) => {
        if (!active) return;
        setInferences(value);
        setSelectedId(value[0]?.id || "");
      })
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let active = true;
    setLoadingDataset(true);
    setDatasetError("");
    setDataset(null);
    setSearch("");
    api(`/api/criticality?inferenceId=${encodeURIComponent(selectedId)}`)
      .then((value) => {
        if (!active) return;
        const normalized = normalizePredictionFile(value);
        setDataset(normalized);
        setSort(primaryMetricName(normalized.metricNames) || "line");
      })
      .catch((value) => active && setDatasetError(value instanceof Error ? value.message : String(value)))
      .finally(() => active && setLoadingDataset(false));
    return () => { active = false; };
  }, [selectedId]);

  const visibleRows = useMemo(() => {
    if (!dataset) return [];
    const query = search.trim().toLowerCase();
    const rows = dataset.predictions.filter((prediction) =>
      !query || prediction.label.toLowerCase().includes(query) || prediction.lineId.toLowerCase().includes(query)
    );
    rows.sort((left, right) => {
      if (sort === "line") return left.lineId.localeCompare(right.lineId, undefined, { numeric: true });
      const leftValue = sort === "structural_uniqueness" ? left.structuralUniqueness : metricValue(left, sort) ?? 0;
      const rightValue = sort === "structural_uniqueness" ? right.structuralUniqueness : metricValue(right, sort) ?? 0;
      return rightValue - leftValue || left.label.localeCompare(right.label);
    });
    return rows;
  }, [dataset, search, sort]);

  if (error) return <ErrorCard error={error} />;
  if (!inferences) return <LoadingState message="Loading inference sets…" />;
  if (!inferences.length) {
    return <ErrorCard error={new Error("No versioned inference result is indexed yet. Run infer criticality from the Runs view.")} />;
  }

  const selectedInference = inferences.find((item) => item.id === selectedId);
  const primary = primaryMetricName(dataset?.metricNames || []);

  return (
    <>
      <PageIntro
        kicker="Inference inspection"
        title="Where a disruption matters."
        description="Predictions, metric percentiles, and structural scores come from Rust inference artifacts. Studio only filters, sorts, and formats the result."
        action={<Button variant="primary" onClick={() => navigate("runs")}>Submit a run</Button>}
      />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <StatCard label="Selected model / snapshot" value={<span className="text-[17px]">{selectedInference ? `${shortId(selectedInference.modelId)} · snapshot ${shortId(selectedInference.snapshotId)}` : "—"}</span>} note="Versioned lineage" />
        <StatCard label="Predicted lines" value={formatCount(dataset?.predictions.length || 0)} note="Rows in the Rust result" />
        <StatCard label="Metrics" value={formatCount(dataset?.metricNames.length || 0)} note="Named output dimensions" />
      </div>
      <Card className="mt-4 overflow-hidden">
        <div className="flex flex-wrap items-end gap-3 border-b border-line px-5 py-4">
          <Field label="Inference result" htmlFor="criticality-inference" className="min-w-[190px]">
            <select id="criticality-inference" className={inputClassName} value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
              {inferences.map((inference) => <option key={inference.id} value={inference.id}>{shortId(inference.snapshotId)} · {shortId(inference.modelId)}</option>)}
            </select>
          </Field>
          <Field label="Search lines" htmlFor="criticality-search" className="min-w-[190px] flex-1">
            <input id="criticality-search" className={inputClassName} type="search" placeholder="Line number or name" value={search} onChange={(event) => setSearch(event.target.value)} />
          </Field>
          <Field label="Sort by" htmlFor="criticality-sort" className="min-w-[180px]">
            <select id="criticality-sort" className={inputClassName} value={sort} onChange={(event) => setSort(event.target.value)}>
              {sortOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </Field>
        </div>
        <SectionHeading title="Ranked predictions" copy="The primary metric is selected by its name, never by a UI-side recalculation." />
        <p className="mx-5 mt-3 mb-0 text-[11px] text-muted">
          {loadingDataset ? "Loading Rust-produced predictions…" : datasetError || `${formatCount(visibleRows.length)} of ${formatCount(dataset?.predictions.length || 0)} lines · sorted by ${metricLabel(sort)}`}
        </p>
        {loadingDataset ? <LoadingState message="Loading Rust-produced predictions…" /> : datasetError ? <p className="px-5 py-8 text-sm text-coral">{datasetError}</p> : (
          <DataTable headers={["Rank", "Line", "Accessibility loss", "Unreachable", "Mean delay", "Structural uniqueness"]} hasRows={visibleRows.length > 0} emptyTitle="No lines match" emptyMessage="Try a different search term.">
            {visibleRows.map((prediction, index) => (
              <tr key={`${prediction.lineId}-${prediction.index}`}>
                <td><span className="font-mono text-[11px] text-faint">{String(index + 1).padStart(2, "0")}</span></td>
                <td>{prediction.label} <span className="font-mono text-[11px] text-faint">#{prediction.lineId}</span></td>
                <td className="font-medium text-mint">{formatMetricValue(primary, metricValue(prediction, primary))}</td>
                <td>{formatMetricValue("unreachable_share", metricValue(prediction, "unreachable_share"))}</td>
                <td>{formatMetricValue("mean_delay_reachable_seconds", metricValue(prediction, "mean_delay_reachable_seconds"))}</td>
                <td>{formatMetricValue("structural_uniqueness", prediction.structuralUniqueness)}</td>
              </tr>
            ))}
          </DataTable>
        )}
      </Card>
    </>
  );
}
