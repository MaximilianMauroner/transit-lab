# Transit Lab

Transit Lab is a Rust-first GTFS research workspace with a local-first Build
and Explore application. It compiles service-day snapshots into canonical
station, line, pattern, and timetable data; simulates line disruptions; and
stores the provenance needed to compare results safely.

The current control plane is deliberately small and runnable on one machine:

```text
Browser → Bun API → SQLite metadata + filesystem artifacts
                       ↓
                    Bun worker → Rust CLI
```

The API serves the browser, and the separate worker claims queued runs from
the same SQLite database. Every run, event, and artifact is versioned by a
stable fingerprint. PostgreSQL, S3-compatible storage, and remote workers can
be added behind these boundaries later.

## Current status

The dependency-free Rust reference path is operational for GTFS compilation,
ordered graph construction, timetable-aware line-removal labels, separate
line similarity facets, criticality inference, and cross-snapshot identity
pairing. The Bun API, worker, artifact manifests, dataset manifests, SSE run
events, line/network explorers, and provenance views are implemented.

Dataset and evaluation artifacts are currently indexed and displayed as
read-only results. The worker does not submit `build-dataset` or `evaluate`
runs until the Rust CLI exposes stable commands and output contracts for them;
it rejects those kinds explicitly rather than inventing a fallback.

The optional LibTorch graph model has a differentiable forward path and a
small masked reconstruction training helper, but the end-to-end GPU
multi-task CLI, held-out cross-city retrieval benchmark, and validated
criticality generalization are still pending. Reference-model outputs are
useful for pipeline verification; they are not evidence of production model
quality.

## Quick start

Install Rust 1.75 or newer and Bun, then run the checks and generate the
deterministic demo corpus:

```bash
cargo test --workspace
bun test apps/studio packages display web-wrapper
cargo run -p transit-cli -- demo --output data/demo
```

Start the active API/web application from the repository root. It serves the
browser interface at <http://localhost:3000>; run the worker in a second
terminal:

```bash
bun run api
bun run worker
```

`bun run web` is an alias for the API/web server. The older Studio server is
still available with `bun run studio` while the two implementations converge.

The worker never accepts arbitrary shell commands; it maps validated run specs
to an allow-listed Rust argument array.

Useful configuration variables are:

```text
PORT                    API port, default 3000
TRANSIT_LAB_ROOT        repository root, default current repository
TRANSIT_LAB_DATA_ROOT   artifact/data directory, default data
TRANSIT_LAB_DB          SQLite file, default data/transit-lab.sqlite
TRANSIT_LAB_WORKER_ID   optional worker identity
```

Health and overview endpoints are available at `/health`, `/api/health`, and
`/api/overview`. The overview reports indexed artifacts and run status rather
than a manually entered progress percentage. Set `TRANSIT_LAB_DATA_ROOT` and
`TRANSIT_LAB_DB` together when keeping metadata and artifacts outside the
repository.

## Rust pipeline

For a real feed:

```bash
cargo run -p transit-cli -- validate --input path/to/gtfs.zip
cargo run -p transit-cli -- compile --input path/to/gtfs.zip \
  --service-date 2026-09-09 --output data/snapshots/example
cargo run -p transit-cli -- graph build \
  --snapshot data/snapshots/example --output data/graphs/example
cargo run -p transit-cli -- labels line-removal \
  --snapshot data/snapshots/example --output data/labels/example.jsonl
```

The compiler preserves GTFS times beyond 24:00, selects active service for a
date, canonicalizes physical stations and passenger-facing lines, retains
ordered route patterns, and materializes compact graph arrays. Label origins
use seeded geographic spread, interchange coverage, and uniform sampling;
the sampling policy is fingerprinted beside the labels. `verify top-lines`
simulates only the selected predicted lines.

The CLI can compare separate facet spaces:

```bash
cargo run -p transit-cli -- similar-lines \
  --query-graph data/graphs/vienna --query-line U2 \
  --candidate-graph data/graphs/berlin \
  --profile network-role --top-k 10
```

Available profiles are `network-role`, `service`, `geometry`, `resilience`,
`general`, and explicit weighted profiles. Measured feature differences are
returned with the scores so a neural or reference similarity value is not
presented as an explanation.

## Transit Lab Studio

The persistent navigation is split into the routes currently shipped by the
Studio:

```text
BUILD     Overview · Data & lineage · Runs
EXPLORE   Network · Criticality · Similarity · Embeddings · Evaluation
```

Datasets and model lineage are panels in `Data & lineage`, rather than
separate routes. All visible results carry network, snapshot, service
date/profile, model, and facet context where applicable. Studio indexes
existing files under `data/` on startup, including feed metadata, compiled
snapshots, graphs, labels, models, predictions, artifact manifests, and
dataset manifests.

Run tracking uses SQLite transactions for atomic claims and server-sent events
for reconnectable progress. Rust stdout/stderr are retained as diagnostics;
machine events are written as versioned JSONL and validated before entering
the run ledger. Artifact and dataset manifests are immutable and retain input
fingerprints instead of deleting superseded outputs.

The older [`display`](display) and [`web-wrapper`](web-wrapper) viewers remain
temporarily while their network and criticality capabilities reach parity in
Studio. They are not additional Studio data sources.

## Data contracts and provenance

The versioned boundary contracts are:

```text
schemas/run-event.v1.json
schemas/artifact-manifest.v1.json
schemas/dataset-manifest.v1.json
schemas/inference-result.v1.json
```

Snapshot, dataset, model, and inference fingerprints include their upstream
inputs and configuration. Raw GTFS IDs and operator strings remain lookup
metadata; normalized numeric features are used for model inputs to reduce
feed-producer and city-name leakage.

Generated data is ignored by Git. Do not commit downloaded feeds, SQLite
databases, checkpoints, or other files under the ignored `data/` paths.

## Optional LibTorch backend

LibTorch is not bundled. On a compatible Linux/NVIDIA machine, compile the
feature-gated code with:

```bash
LIBTORCH=/opt/libtorch cargo build -p transit-model --features tch-backend
```

This enables the differentiable relational model modules. It does not yet
make the default CLI a complete GPU multi-task training/evaluation workflow;
that remains an explicit next implementation stage.
