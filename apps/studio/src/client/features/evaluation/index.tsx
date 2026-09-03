import { useEffect, useState } from "react";
import { api, formatCount } from "../../api.ts";
import { Card, DataTable, ErrorCard, LiveBadge, LoadingState, PageIntro, SectionHeading } from "../../components/ui.tsx";

export function EvaluationView() {
  const [rows, setRows] = useState<any[] | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let active = true;
    api("/api/evaluations")
      .then((value) => active && setRows(value))
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  if (error) return <ErrorCard error={error} />;
  if (!rows) return <LoadingState />;

  return (
    <>
      <PageIntro
        kicker="Quality & evaluation"
        title="Measure before you trust."
        description="Evaluation values are recorded by Rust runs and displayed with their model, dataset, facet, and split lineage."
        action={<LiveBadge>{formatCount(rows.length)} recorded points</LiveBadge>}
      />
      <Card className="overflow-hidden">
        <SectionHeading title="Evaluation points" copy="No client-side metric computation." />
        <DataTable headers={["Facet", "Metric", "Value", "Model", "Dataset", "Split"]} hasRows={rows.length > 0} emptyTitle="No evaluation points" emptyMessage="Run an evaluation command and publish its versioned result artifact.">
          {rows.map((row, index) => (
            <tr key={`${row.metricName}-${row.modelId}-${index}`}>
              <td>{row.facet}</td>
              <td>{row.metricName}</td>
              <td className="font-mono text-mint">{Number(row.value).toFixed(4)}</td>
              <td>{row.modelVersion || row.modelId || "—"}</td>
              <td>{row.datasetId || "—"}</td>
              <td>{row.split || "—"}</td>
            </tr>
          ))}
        </DataTable>
      </Card>
    </>
  );
}
