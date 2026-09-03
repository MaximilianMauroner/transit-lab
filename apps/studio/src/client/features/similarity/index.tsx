import { useEffect, useState, type FormEvent } from "react";
import { api, formatCount, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { Button, Card, DataTable, EmptyState, ErrorCard, Field, inputClassName, LoadingState, Mono, PageIntro, SectionHeading } from "../../components/ui.tsx";

const facetNames = ["role", "service", "geometry", "resilience"];

export function SimilarityView() {
  const [snapshots, setSnapshots] = useState<any[] | null>(null);
  const [querySnapshot, setQuerySnapshot] = useState("");
  const [candidateSnapshot, setCandidateSnapshot] = useState("");
  const [queryLine, setQueryLine] = useState("0");
  const [lines, setLines] = useState<any[]>([]);
  const [profile, setProfile] = useState("general");
  const [result, setResult] = useState<any>(null);
  const [loadingLines, setLoadingLines] = useState(false);
  const [loadingResult, setLoadingResult] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [resultError, setResultError] = useState<unknown>(null);

  useEffect(() => {
    let active = true;
    api("/api/snapshots")
      .then((value) => {
        if (!active) return;
        setSnapshots(value);
        setQuerySnapshot(value[0]?.id || "");
        setCandidateSnapshot(value[0]?.id || "");
      })
      .catch((value) => active && setError(value));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!querySnapshot) return;
    let active = true;
    setLoadingLines(true);
    api(`/api/snapshots/${encodeURIComponent(querySnapshot)}/network`)
      .then((network) => {
        if (!active) return;
        const nextLines = network.lines || [];
        setLines(nextLines);
        setQueryLine(nextLines.length ? "0" : "");
      })
      .catch((value) => active && setResultError(value))
      .finally(() => active && setLoadingLines(false));
    return () => { active = false; };
  }, [querySnapshot]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoadingResult(true);
    setResultError(null);
    try {
      const params = new URLSearchParams({ querySnapshotId: querySnapshot, candidateSnapshotId: candidateSnapshot, queryLineIndex: queryLine, profile });
      setResult(await api(`/api/similarity?${params}`));
    } catch (value) {
      setResultError(value);
    } finally {
      setLoadingResult(false);
    }
  }

  if (error) return <ErrorCard error={error} />;
  if (!snapshots) return <LoadingState message="Loading snapshots…" />;
  if (!snapshots.length) return <ErrorCard error={new Error("Similarity needs at least one indexed snapshot.")} />;

  const options = snapshots.map((snapshot) => (
    <option key={snapshot.id} value={snapshot.id}>{snapshot.sourceName || snapshot.networkId || shortId(snapshot.id)} · {snapshot.serviceDate}</option>
  ));

  return (
    <>
      <PageIntro
        kicker="Cross-snapshot retrieval"
        title="Compare by meaning."
        description="Similarity facets and measured comparisons are computed by Rust. Studio displays a stored result artifact and never reconstructs embeddings in JavaScript."
        action={<Button variant="quiet" onClick={() => navigate("embeddings")}>Inspect embeddings →</Button>}
      />
      <Card className="overflow-hidden">
        <form className="flex flex-wrap items-end gap-3 border-b border-line p-4 sm:px-5" onSubmit={submit}>
          <Field label="Query snapshot" htmlFor="similarity-query-snapshot" className="min-w-[180px]">
            <select id="similarity-query-snapshot" className={inputClassName} value={querySnapshot} onChange={(event) => setQuerySnapshot(event.target.value)}>{options}</select>
          </Field>
          <Field label="Query line" htmlFor="similarity-query-line" className="min-w-[180px]">
            <select id="similarity-query-line" className={inputClassName} value={queryLine} disabled={loadingLines || !lines.length} onChange={(event) => setQueryLine(event.target.value)}>
              {loadingLines ? <option>Loading lines…</option> : lines.map((line, index) => <option key={index} value={index}>{line.display_name || line.canonical_id || `Line ${index}`}</option>)}
            </select>
          </Field>
          <Field label="Candidate snapshot" htmlFor="similarity-candidate-snapshot" className="min-w-[180px]">
            <select id="similarity-candidate-snapshot" className={inputClassName} value={candidateSnapshot} onChange={(event) => setCandidateSnapshot(event.target.value)}>{options}</select>
          </Field>
          <Field label="Facet profile" htmlFor="similarity-profile" className="min-w-[160px]">
            <select id="similarity-profile" className={inputClassName} value={profile} onChange={(event) => setProfile(event.target.value)}>
              <option value="general">General</option>
              <option value="role">Network role</option>
              <option value="service">Service</option>
              <option value="geometry">Geometry</option>
              <option value="resilience">Resilience</option>
            </select>
          </Field>
          <Button variant="primary" type="submit" disabled={loadingResult || loadingLines || !queryLine}>{loadingResult ? "Loading…" : "Load result"}</Button>
        </form>
      </Card>
      <Card className="mt-4 overflow-hidden">
        <SectionHeading title="Similarity result" copy="Select a Rust-produced result artifact to inspect facet scores." action={result ? <Mono className="text-faint">{shortId(result.artifactId || "")}</Mono> : null} />
        {resultError ? <p className="px-5 py-8 text-sm text-coral">{resultError instanceof Error ? resultError.message : String(resultError)}</p> : loadingResult ? <LoadingState message="Reading Rust-produced similarity results…" /> : result ? <SimilarityResult result={result} /> : <EmptyState title="No result selected" message="Run the Rust similar-lines command and refresh the index." />}
      </Card>
    </>
  );
}

function SimilarityResult({ result }: { result: any }) {
  const matches = result.matches || result.results || [];
  return (
    <DataTable headers={["Candidate", "Score", "Role", "Service", "Geometry", "Resilience", "Mode"]} hasRows={matches.length > 0} emptyTitle="No matching lines" emptyMessage="The Rust result contains no candidates for this query.">
      {matches.map((match, index) => (
        <tr key={`${match.lineInstanceId || match.lineName || index}`}>
          <td>{match.displayName || match.lineName || match.lineInstanceId || "Line"}</td>
          <td className="font-medium text-mint">{Number(match.similarity ?? 0).toFixed(3)}</td>
          {facetNames.map((name) => <td key={name}>{Number(match.facetScores?.[name] ?? match.facets?.[name] ?? 0).toFixed(3)}</td>)}
          <td>{match.comparison?.mode || "—"}</td>
        </tr>
      ))}
    </DataTable>
  );
}
