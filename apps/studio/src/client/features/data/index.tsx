import { useEffect, useState } from "react";
import { api, formatCount, formatDate, shortId } from "../../api.ts";
import { Button, Card, DataTable, ErrorCard, LoadingState, PageIntro, SectionHeading, StatCard, StatusBadge, Mono } from "../../components/ui.tsx";

export function DataView() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = () => {
    setError(null);
    return Promise.all([
      api("/api/snapshots"),
      api("/api/artifacts?limit=250"),
      api("/api/datasets"),
      api("/api/models")
    ]).then(([snapshots, artifacts, datasets, models]) => setData({ snapshots, artifacts, datasets, models }));
  };

  useEffect(() => {
    let active = true;
    Promise.all([
      api("/api/snapshots"),
      api("/api/artifacts?limit=250"),
      api("/api/datasets"),
      api("/api/models")
    ])
      .then(([snapshots, artifacts, datasets, models]) => active && setData({ snapshots, artifacts, datasets, models }))
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  async function refresh() {
    setRefreshing(true);
    try {
      await api("/api/inventory/refresh", { method: "POST" });
      await load();
    } catch (value) {
      setError(value);
    } finally {
      setRefreshing(false);
    }
  }

  if (error) return <ErrorCard error={error} />;
  if (!data) return <LoadingState />;
  const { snapshots, artifacts, datasets, models } = data;

  return (
    <>
      <PageIntro
        kicker="Manifests & lineage"
        title="Data has an address."
        description="Every indexed item points back to a versioned Rust output. Directory names are presentation only; relationships come from manifest inputs."
        action={<Button variant="quiet" onClick={refresh} disabled={refreshing}>{refreshing ? "Refreshing…" : "Refresh from disk"}</Button>}
      />
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <StatCard label="Snapshots" value={formatCount(snapshots.length)} />
        <StatCard label="Artifacts" value={formatCount(artifacts.length)} />
        <StatCard label="Datasets" value={formatCount(datasets.length)} />
        <StatCard label="Models" value={formatCount(models.length)} />
      </div>
      <Card className="mt-4 overflow-hidden">
        <SectionHeading title="Snapshots" copy="Compiled network snapshots available to the Studio." />
        <DataTable headers={["Source", "Snapshot", "Date", "Entities", "Status"]} hasRows={snapshots.length > 0} emptyTitle="No snapshots" emptyMessage="Produce a Rust snapshot, then refresh the index.">
          {snapshots.map((snapshot) => (
            <tr key={snapshot.id}>
              <td>{snapshot.sourceName || snapshot.networkId || "Unknown"}</td>
              <td><Mono>{shortId(snapshot.id)}</Mono></td>
              <td>{snapshot.serviceDate}</td>
              <td>{formatCount(snapshot.counts?.stations || 0)} stations · {formatCount(snapshot.counts?.lines || 0)} lines</td>
              <td><StatusBadge value={snapshot.status} /></td>
            </tr>
          ))}
        </DataTable>
      </Card>
      <Card className="mt-4 overflow-hidden">
        <SectionHeading title="Artifacts" copy="Explicit v1 manifests and their provenance fields." />
        <DataTable headers={["Kind", "Artifact", "Schema", "SHA-256", "Created"]} hasRows={artifacts.length > 0} emptyTitle="No explicit artifact manifests" emptyMessage="Rust output manifests will appear here after the next run.">
          {artifacts.map((artifact) => (
            <tr key={artifact.id}>
              <td>{artifact.kind}</td>
              <td><Mono>{shortId(artifact.id)}</Mono></td>
              <td>{artifact.schemaVersion ?? "legacy"}</td>
              <td><Mono className="text-faint">{shortId(artifact.sha256, 10, 8)}</Mono></td>
              <td>{formatDate(artifact.createdAt)}</td>
            </tr>
          ))}
        </DataTable>
      </Card>
    </>
  );
}
