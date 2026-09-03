import { useEffect, useState } from "react";
import { api, formatCount, shortId } from "../../../../packages/api-client/src/index.ts";
import { Card, DataTable, ErrorCard, LiveBadge, LoadingState, PageIntro, SectionHeading } from "../../../../packages/ui/src/react.tsx";

export function EmbeddingsView() {
  return <AnalysisView path="/api/public/embeddings" title="Embeddings" description="Published embedding and projection artifacts." />;
}

export function EvaluationView() {
  return <AnalysisView path="/api/public/evaluations" title="Evaluation" description="Published evaluation summaries and quality signals." />;
}

function AnalysisView({ path, title, description }: { path: string; title: string; description: string }) {
  const [records, setRecords] = useState<any[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    let active = true;
    api(path).then((value) => active && setRecords(value)).catch((value) => active && setError(value));
    return () => { active = false; };
  }, [path]);
  if (error) return <ErrorCard error={error} />;
  if (!records) return <LoadingState message="Loading published analysis…" />;
  if (!records.length) return <ErrorCard error={new Error(`No published ${title.toLowerCase()} results are available.`)} />;
  return <><PageIntro kicker="Published analysis" title={title} description={description} action={<LiveBadge>{formatCount(records.length)} published records</LiveBadge>} /><Card className="overflow-hidden"><SectionHeading title={title} copy={`${formatCount(records.length)} published records`} /><DataTable headers={["Result", "Reference", "Facet / split", "Value"]}>{records.map((record, index) => <tr key={`${record.id || record.modelId}-${index}`}><td>{record.kind || record.metricName || "Result"}</td><td>{shortId(record.id || record.modelId || "")}</td><td>{record.facet || record.split || record.status || "ready"}</td><td className="font-medium text-mint">{record.value === undefined ? "Published" : Number(record.value).toFixed(4)}</td></tr>)}</DataTable></Card></>;
}
