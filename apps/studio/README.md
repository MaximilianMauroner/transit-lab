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
evaluation views remain read-only until the Rust CLI exposes versioned
`build-dataset` and `evaluate` commands; the worker rejects those unsupported
kinds explicitly.

## Run locally

From the repository root:

```bash
TRANSIT_LAB_ROOT="$PWD" bun run apps/studio/src/server/index.js
```

In a second terminal, start the allow-listed Rust worker:

```bash
TRANSIT_LAB_ROOT="$PWD" bun run apps/studio/src/worker/index.js
```

The server defaults to `http://localhost:3000`. Configure the repository and
artifact locations with `TRANSIT_LAB_ROOT`, `TRANSIT_LAB_DATA_DIR`,
`TRANSIT_LAB_BINARY`, and `TRANSIT_LAB_DB`.

Legacy `display/` and `web-wrapper/` remain available while `/network` and
`/criticality` reach parity. They are not additional Studio data sources.
