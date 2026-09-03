import { useEffect, useState, type MouseEvent } from "react";
import { api, formatCount, formatDate, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { Card, DataTable, ErrorCard, LoadingState, PageIntro, StatCard, StatusBadge, Mono, SectionHeading, Button } from "../../components/ui.tsx";

export function OverviewView() {
  const [overview, setOverview] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let active = true;
    api("/api/overview")
      .then((value) => active && setOverview(value))
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  if (error) return <ErrorCard error={error} />;
  if (!overview) return <LoadingState />;

  const counts = overview.counts || {};
  const snapshots = overview.snapshots || [];
  const recentRuns = overview.recentRuns || [];

  return (
    <>
      <PageIntro
        kicker="Repository workspace"
        title="Keep the computation traceable."
        description="Rust owns computation and immutable outputs. Studio indexes the manifests, follows run events, and makes the resulting network and model lineage inspectable."
        action={<Button variant="primary" onClick={() => navigate("network")}>Explore network</Button>}
      />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Snapshots" value={formatCount(counts.snapshots)} note="compiled service-day views" href="/network" onClick={handleLink("network")} />
        <StatCard label="Runs" value={formatCount(counts.runs)} note="queued and completed jobs" href="/runs" onClick={handleLink("runs")} />
        <StatCard label="Models" value={formatCount(counts.models)} note="versioned checkpoints" href="/data" onClick={handleLink("data")} />
        <StatCard label="Artifacts" value={formatCount(counts.explicitArtifacts ?? counts.snapshots + counts.models)} note="indexed immutable outputs" href="/data" onClick={handleLink("data")} />
      </div>
      <div className="mt-4 grid grid-cols-1 gap-4 2xl:grid-cols-2">
        <Card>
          <SectionHeading title="Recent snapshots" copy="Imported from snapshot manifests." action={<a className="text-[11px] text-mint hover:text-copy" href="/data" onClick={handleLink("data")}>All data →</a>} />
          <DataTable headers={["Snapshot", "Source", "Service date", "Status"]} hasRows={snapshots.length > 0} emptyTitle="No snapshots" emptyMessage="Refresh the inventory after producing a Rust snapshot.">
            {snapshots.map((snapshot) => (
              <tr key={snapshot.id}>
                <td><a className="font-mono text-[11px] text-mint hover:underline" href="/network" onClick={handleLink("network")}>{shortId(snapshot.id)}</a></td>
                <td>{snapshot.sourceName || snapshot.networkId || "Unknown"}</td>
                <td>{snapshot.serviceDate || "—"}</td>
                <td><StatusBadge value={snapshot.status} /></td>
              </tr>
            ))}
          </DataTable>
        </Card>
        <Card>
          <SectionHeading title="Recent runs" copy="Events are replayable from the run ledger." action={<a className="text-[11px] text-mint hover:text-copy" href="/runs" onClick={handleLink("runs")}>Open runs →</a>} />
          <DataTable headers={["Kind", "Run", "Status", "Created"]} hasRows={recentRuns.length > 0} emptyTitle="No runs" emptyMessage="Submit a typed operation from the Runs view.">
            {recentRuns.map((run) => (
              <tr key={run.id}>
                <td>{run.kind}</td>
                <td><Mono>{shortId(run.id)}</Mono></td>
                <td><StatusBadge value={run.status} /></td>
                <td>{formatDate(run.createdAt)}</td>
              </tr>
            ))}
          </DataTable>
        </Card>
      </div>
    </>
  );
}

function handleLink(route: string) {
  return (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    navigate(route);
  };
}
