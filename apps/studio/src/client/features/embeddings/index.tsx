import { useEffect, useState } from "react";
import { api, formatCount, shortId } from "../../api.ts";
import { Card, DataTable, ErrorCard, LiveBadge, LoadingState, Mono, PageIntro, SectionHeading } from "../../components/ui.tsx";

export function EmbeddingsView() {
  const [artifacts, setArtifacts] = useState<any[] | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let active = true;
    api("/api/embeddings")
      .then((value) => active && setArtifacts(value))
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  if (error) return <ErrorCard error={error} />;
  if (!artifacts) return <LoadingState />;

  return (
    <>
      <PageIntro
        kicker="Representation space"
        title="Embeddings with lineage."
        description="Projection files and embedding artifacts are displayable here once Rust emits their manifests. The browser does not generate or alter vectors."
        action={<LiveBadge>{formatCount(artifacts.length)} indexed artifacts</LiveBadge>}
      />
      <Card className="overflow-hidden">
        <SectionHeading title="Embedding artifacts" copy="Projection, facet, and base representation outputs." />
        <DataTable headers={["Kind", "Artifact", "Schema", "Files", "Status"]} hasRows={artifacts.length > 0} emptyTitle="No embedding artifacts" emptyMessage="Produce a Rust embedding or projection artifact, then refresh the index.">
          {artifacts.map((artifact) => (
            <tr key={artifact.id}>
              <td>{artifact.kind}</td>
              <td><Mono>{shortId(artifact.id)}</Mono></td>
              <td>{artifact.schemaVersion ?? "legacy"}</td>
              <td>{formatCount(artifact.files?.length || 0)}</td>
              <td>{artifact.status}</td>
            </tr>
          ))}
        </DataTable>
      </Card>
    </>
  );
}
