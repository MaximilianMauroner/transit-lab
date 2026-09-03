# Transit Lab Studio

Transit Lab Studio is the repository-local control plane for Transit Lab. It
indexes versioned Rust artifacts, accepts typed run specifications, records
structured JSONL events, and exposes the network and criticality explorers in
one application.

The Studio does not calculate model-critical values. It reads predictions,
percentiles, metrics, and embeddings emitted by Rust and only sorts, filters,
formats, and visualizes them.

The shipped routes are `/`, `/data`, `/runs`, `/network`, `/criticality`,
`/similarity`, `/embeddings`, and `/evaluation`. Dataset and model lineage are
available from `/data`.

The current run form submits simulation and inference operations. Dataset and
evaluation views display indexed Rust outputs. The worker rejects run kinds
that do not have an allow-listed Rust command.

## Run locally

From the repository root:

```bash
TRANSIT_LAB_ROOT="$PWD" bun run apps/studio/src/server/index.ts
```

In another terminal, start the control API and allow-listed Rust worker:

```bash
TRANSIT_LAB_ROOT="$PWD" bun run apps/api/src/index.ts
TRANSIT_LAB_ROOT="$PWD" bun run apps/worker/src/index.ts
```

Studio defaults to `http://localhost:3000` and proxies `/api/control` to the
control API. Configure repository and artifact locations with
`TRANSIT_LAB_ROOT`, `TRANSIT_LAB_DATA_DIR`, `TRANSIT_LAB_BINARY`, and
`TRANSIT_LAB_DB`. Run `bun run db:push` to push the control-store schema
explicitly. Startup repeats the same idempotent push for local databases.

The browser client is a React application bundled by Bun. Tailwind CSS is
compiled alongside the client into `public/dist` when Studio is built or
started for the first time.
