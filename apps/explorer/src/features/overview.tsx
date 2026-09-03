import { useEffect, useState, type MouseEvent } from "react";
import { api, formatCount, formatDate, shortId } from "../../../../packages/api-client/src/index.ts";
import { navigate } from "../router.ts";
import { Card, DataTable, EmptyState, ErrorCard, LiveBadge, LoadingState, Mono, PageIntro, SectionHeading, StatCard } from "../../../../packages/ui/src/react.tsx";

export function OverviewView() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let active = true;
    api("/api/public/catalog")
      .then(async (catalog) => {
        const overview = await api("/api/public/overview").catch(() => ({ snapshots: [], counts: {} }));
        if (active) setData({ catalog, overview });
      })
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  if (error) return <ErrorCard error={error} />;
  if (!data) return <LoadingState message="Loading published catalog…" />;
  const publications = data.catalog.publications || [];
  if (!publications.length) return <Card><EmptyState title="No published results" message="A Studio owner must publish a result bundle before it appears here." /></Card>;

  const overview = data.overview;
  const snapshots = overview.snapshots || [];
  return (
    <>
      <PageIntro kicker="Transit Lab Explorer" title="Published transit intelligence." description="Explore immutable network, model, similarity, and evaluation results selected by the research team. Private runs and unfinished artifacts stay in Studio." action={<LiveBadge>Read-only publication</LiveBadge>} />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard label="Publications" value={formatCount(publications.length)} />
        <StatCard label="Snapshots" value={formatCount(overview.counts?.snapshots || 0)} />
        <StatCard label="Models" value={formatCount(overview.counts?.models || 0)} />
      </div>
      <Card className="mt-4 overflow-hidden">
        <SectionHeading title="Publication bundles" copy="Every item below is an intentional published view of Rust-produced artifacts." />
        <div className="grid grid-cols-1 gap-3 p-5 md:grid-cols-2 xl:grid-cols-3">
          {publications.map((publication) => <PublicationCard key={publication.id} publication={publication} />)}
        </div>
      </Card>
      {snapshots.length ? <Card className="mt-4 overflow-hidden"><SectionHeading title="Published snapshots" copy="Service-day network views available to explore." /><DataTable headers={["Snapshot", "Source", "Service date", "Status"]}>{snapshots.map((snapshot) => <tr key={snapshot.id}><td><a className="font-mono text-[11px] text-mint hover:underline" href="/network" onClick={linkClick("network")}>{shortId(snapshot.id)}</a></td><td>{snapshot.sourceName || snapshot.networkId || "Unknown"}</td><td>{snapshot.serviceDate || "—"}</td><td>{snapshot.status || "ready"}</td></tr>)}</DataTable></Card> : null}
    </>
  );
}

function PublicationCard({ publication }: { publication: any }) {
  return <article className="rounded-lg border border-line bg-panel-raised p-4"><p className="mb-1.5 text-[10px] font-extrabold uppercase tracking-[0.14em] text-mint">{publication.slug || shortId(publication.id)}</p><h3 className="m-0 text-base font-semibold text-copy">{publication.title}</h3><p className="my-2 text-sm text-muted">{formatCount(publication.snapshotIds?.length || 0)} snapshots · {formatCount(publication.modelIds?.length || 0)} models</p><small className="text-xs text-faint">Updated {formatDate(publication.updatedAt)}</small></article>;
}

function linkClick(route: "network") {
  return (event: MouseEvent<HTMLAnchorElement>) => { if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return; event.preventDefault(); navigate(route); };
}
