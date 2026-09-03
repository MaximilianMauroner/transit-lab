# Transit Lab

Transit Lab turns raw public transport data into a trained model that learns
how transit networks behave. It can compare cities and lines, estimate which
lines are structurally important, and publish those results for exploration.

The research question is whether a model can learn reusable structure from
many cities' GTFS networks. The Studio, API, worker, and Explorer support that
work. They do not replace it.

The shortest description is:

> Take many cities' GTFS networks, turn them into comparable graphs, generate
> labels by simulating network disruptions, train a graph model to learn
> reusable representations of transit lines and networks, then use that model
> to compare lines and predict network importance in cities it has not seen
> before.

## The research pipeline

```text
GTFS ZIP
      |
      v
1. Raw transit data
      |
      v
2. Ingest and validate
      |
      v
3. Build a canonical transit snapshot
      |
      v
4. Convert the snapshot into a graph
      |
      v
5. Generate targets and training examples
      |
      v
6. Build a versioned dataset
      |
      v
7. Configure an experiment
      |
      v
8. Train
      |
      v
9. Evaluate
      |
      v
10. Run inference
      |
      v
Results for Studio and Explorer
```

### 1. Raw transit data

The input is a standard GTFS ZIP containing stops, routes, trips, stop times,
calendars, transfers, and related tables. A feed fetch creates an immutable
copy of the source and records metadata such as its download time and SHA-256
checksum.

### 2. Ingest and validate

The GTFS parser checks the feed before any model work begins. It catches broken
references, malformed times, inconsistent stop times, and invalid service
calendars. Training on a broken feed makes data errors look like model errors,
so validation is a hard boundary in the pipeline.

```bash
cargo run -p transit-cli -- validate --input path/to/gtfs.zip
```

### 3. Build a canonical transit snapshot

GTFS identifiers and grouping rules differ between operators. Transit Lab
compiles each feed into one service-day snapshot with stable internal shapes:

```text
Vienna / 2026-09-02
    |
    +-- stations
    +-- lines
    +-- route patterns
    +-- trips and stop times
    +-- transfers
    +-- source and merge evidence
```

The snapshot records the service date, source metadata, transfer policy, line
grouping policy, validation report, and an ID derived from those inputs. Raw
GTFS IDs remain lookup metadata. Model features use normalized measurements so
the model does not learn a city name or an operator's identifier by accident.

### 4. Convert the snapshot into a graph

The compiled network becomes a `GraphTensor` suitable for relational learning.
It contains station and line entities plus relations for:

- which stations a line serves;
- station-to-station transit segments;
- passenger transfers and physical interchanges;
- ordered route patterns and their trips.

The graph also stores numeric and temporal features. A line representation can
include station and pattern counts, route length, branching, service span,
headways, trip counts, transfer coverage, geometry, and service across
15-minute bins. Transit edges carry distance, travel time, route position, and
time-dependent service information.

```bash
cargo run -p transit-cli -- compile --input path/to/gtfs.zip \
  --service-date 2026-09-02 --output data/snapshots/vienna
cargo run -p transit-cli -- graph build \
  --snapshot data/snapshots/vienna --output data/graphs/vienna
```

### 5. Generate targets and training examples

The model can learn from several kinds of signals:

- Masked reconstruction hides parts of a graph and asks the model to recover
  the missing features.
- Cross-snapshot line identities pair the same real line across service dates
  or feed revisions. The representations should stay close even when the
  timetable changes.
- Similarity examples use positive pairs and triplets to teach separate
  network-role, service, geometry, and resilience facets.
- Criticality labels come from counterfactual simulation. The router measures
  what changes when one line is disabled.

```text
Intact network
      |
      +-- disable one line
      |
      +-- recompute timetable-aware journeys
      |
      +-- measure the damage
```

A line-impact label records values such as accessibility loss, unreachable
share, mean and p95 delay for destinations that remain reachable, extra
transfers, and the share of stations that lose all service. These are computed
targets, not hand-written importance scores.

```bash
cargo run -p transit-cli -- labels line-removal \
  --snapshot data/snapshots/vienna --output data/labels/vienna.jsonl
```

### 6. Build a versioned dataset

Graphs, labels, feature schemas, snapshot IDs, objectives, and train/validation/
test split rules belong to a dataset definition. The dataset manifest gives
that definition an ID and content fingerprint.

```text
Dataset europe-v3
    +-- city and service-day snapshots
    +-- graph tensors
    +-- disruption labels
    +-- positive pairs and triplets
    +-- split definition
    +-- input fingerprints
```

This is what makes two experiments comparable. If the inputs or label policy
change, the dataset identity changes too.

### 7. Configure an experiment

An experiment binds a dataset to a model configuration, runtime settings, and
a seed. Conceptually:

```text
Dataset: europe-v3
Model: hidden dimension 256, 4 graph layers
Training: learning rate 0.0005, 100 epochs
Objectives: reconstruction 1.0, similarity 0.4, criticality 0.8
Seed: 42
```

The worker resolves the configuration into an immutable run file. A
reproducible model run should identify the code revision, dataset fingerprint,
resolved configuration, and seed.

### 8. Train

The Studio controls a run, but the Rust training engine owns the model values,
losses, optimizer steps, checkpoints, and metric calculations.

```text
Studio
  |
  +-- submit typed experiment spec
  v
Control API
  |
  +-- queue run
  v
Worker
  |
  +-- launch an allow-listed Rust CLI command
  v
Rust training engine
  |
  +-- forward pass, loss, backward pass, checkpoint
  v
Versioned model artifacts and structured run events
```

The worker streams machine-readable progress to the Studio. A run can expose
the current phase, epoch, learning rate, reconstruction loss, criticality
metrics, and checkpoint locations while it is running.

The reference training command is:

```bash
cargo run -p transit-cli -- train multitask \
  --graph data/graphs/vienna \
  --labels data/labels/vienna.jsonl \
  --config configs/models/multitask-v1.yaml \
  --output data/models/vienna.json
```

Training attempts can checkpoint and exit cooperatively. The logical run keeps
its identity while later worker attempts resume the newest committed
checkpoint:

```bash
cargo run -p transit-cli -- train multitask \
  --graph data/graphs/vienna \
  --labels data/labels/vienna.jsonl \
  --config configs/models/multitask-v1.yaml \
  --output data/models/vienna.json \
  --checkpoint-dir data/runs/vienna-training/checkpoints \
  --control-file data/runs/vienna-training/control.json \
  --checkpoint-every-steps 500 \
  --checkpoint-every-seconds 900 \
  --resume latest
```

Pause requests are honored after a complete optimizer step. The trainer saves
an atomic checkpoint directory and exits, releasing its process and device;
the worker can later start a new attempt from that directory. A resumable
checkpoint includes model weights, optimizer moments, scheduler/scaler state,
the cursor, sampler state, RNG state, metrics, and dataset/configuration
fingerprints. The flat model export is a separate inference artifact: native
LibTorch exports use `model.json` plus a sibling `.weights.ot` file and do not
claim to be resumable by themselves.

### 9. Evaluate

Training loss is not enough. Evaluation asks whether the learned features work
on data the model did not train on. A useful split holds out a complete city
or transit system:

```text
Train: Vienna, Berlin, Prague, Munich
Test:  Hamburg
```

Evaluation can measure criticality rank correlation, top-k ranking quality,
similar-line retrieval, cross-city retrieval, and reconstruction error. The
held-out system is the test of whether the model learned network structure
rather than memorized city-specific details.

### 10. Run inference

Inference applies a trained model to a new graph:

```text
New GTFS feed
      |
      v
Canonical snapshot and graph
      |
      v
Trained model
      |
      +-- line and network embeddings
      +-- criticality predictions
      +-- similar-line results
```

This is intended to cost much less than running every disruption simulation.
Exact top-line verification remains available when a prediction needs to be
checked against the router.

```bash
cargo run -p transit-cli -- infer criticality \
  --graph data/graphs/vienna \
  --model data/models/model.json \
  --output data/predictions/vienna.json

cargo run -p transit-cli -- similar-lines \
  --query-graph data/graphs/vienna --query-line U2 \
  --candidate-graph data/graphs/berlin \
  --profile network-role --top-k 10
```

## Studio and Explorer

The research pipeline produces the values. The applications make those values
usable.

```text
                         Research pipeline
GTFS -> snapshots -> graphs -> labels -> datasets -> models -> results
                                                               |
                                                               v
                                                        Studio and Explorer
```

Studio is the private control plane. It configures experiments, queues runs,
tracks structured events, indexes artifacts, and provides network,
criticality, similarity, embedding, and evaluation views.

Explorer is the public read-only surface. It reads immutable publication
bundles and presents maps, rankings, line comparisons, and embedding views.
It mostly displays pipeline outputs rather than calculating new model values.

The local architecture is:

```text
Explorer  -> Public API  -> immutable publication bundles
Studio    -> Control API  -> SQLite control store and repository data
Worker    -> Control API  -> allow-listed Rust CLI commands
Rust      -> model, routing, labels, training, inference
```

Rust owns transit semantics, routing, model calculations, and metrics. Bun
applications index, sort, filter, format, and display the resulting artifacts.
The worker never accepts arbitrary shell commands.

## Current status

The dependency-free Rust reference path is operational for GTFS fetching and
validation, snapshot compilation, ordered graph construction, timetable-aware
line-removal labels, separate similarity facets, criticality inference, and
cross-snapshot identity pairing. Studio, Explorer, the control API, worker,
artifact manifests, dataset manifests, server-sent run events, and provenance
views are implemented.

Dataset, training, evaluation, benchmark, inference, and embedding artifacts
are indexed and displayed as read-only results. The worker submits these
commands only after their typed inputs and immutable output contracts pass
validation. Reference-model outputs verify the pipeline; they are not evidence
of production model quality or cross-city generalisation.

The optional LibTorch backend provides the differentiable relational model,
resumable multi-task training, native model export, and CPU inference. The
standard `tch::nn::Optimizer` wrapper remains weights-only because `tch` 0.18
does not expose its private C++ state dictionary. The resumable session uses an
explicit Rust-owned Adam/AdamW implementation so optimizer moments are part of
the committed checkpoint contract.

## Quick start

Install Rust 1.75 or newer and Bun. Then initialize the local control store,
run the checks, and generate the deterministic demo corpus:

```bash
bun run db:push
cargo test --workspace
bun test
cargo run -p transit-cli -- demo --output data/demo
```

Start the control API, Studio, worker, and public Explorer in separate
terminals:

```bash
bun run api       # http://localhost:3100
bun run studio    # http://localhost:3000
bun run worker
bun run explorer  # http://localhost:3200
```

Useful configuration variables are:

```text
CONTROL_API_PORT        Control API port, default 3100
STUDIO_PORT             Studio port, default 3000
EXPLORER_PORT           Explorer port, default 3200
TRANSIT_LAB_CONTROL_API_URL
                        API URL used by Studio and Explorer proxies
TRANSIT_LAB_ROOT        repository root, default current repository
TRANSIT_LAB_DATA_DIR    artifact/data directory, default data
TRANSIT_LAB_DATA_ROOT   legacy alias for TRANSIT_LAB_DATA_DIR
TRANSIT_LAB_DB          SQLite file, default data/transit-lab.sqlite
TRANSIT_LAB_BINARY      optional transit CLI executable
TRANSIT_LAB_WORKER_ID   optional worker identity
TRANSIT_LAB_PUBLIC_ORIGIN
                        public API CORS origin, default *
```

The control API uses `/api/control/...` for private Studio operations and
`/api/public/...` for published read-only data. Unscoped `/api/...` aliases
are disabled unless `TRANSIT_LAB_ALLOW_LEGACY_API=1` is set during the
transition.

`bun run db:push` and application startup call the same idempotent
`pushDatabaseSchema` function. There is no migration directory or migration
ledger. Existing local databases receive missing compatible columns when the
schema is pushed.

## Data contracts and provenance

The versioned boundary contracts live in [`schemas`](schemas):

```text
schemas/run-event.v1.json
schemas/run-event.v2.json
schemas/experiment-spec.v1.json
schemas/artifact-manifest.v1.json
schemas/dataset-manifest.v1.json
schemas/training-checkpoint.v1.json
schemas/training-control.v1.json
schemas/benchmark-result.v1.json
schemas/evaluation-result.v1.json
schemas/inference-result.v1.json
schemas/publication-manifest.v1.json
```

Snapshot, dataset, model, and inference fingerprints include their upstream
inputs and configuration. Artifact and dataset manifests remain immutable and
retain input fingerprints instead of deleting superseded outputs.

Generated data stays at repository level so Rust, the worker, API, tests, and
both clients can refer to the same artifacts. Do not put it under an app
folder. Downloaded feeds, SQLite databases, checkpoints, and other local data
are ignored by Git.

## Optional LibTorch backend

LibTorch is not bundled. On a compatible machine, compile the feature-gated
code with:

```bash
LIBTORCH=/opt/libtorch cargo build -p transit-model --features tch-backend
# Or build and run the complete optional CLI backend:
LIBTORCH=/opt/libtorch cargo run -p transit-cli --features tch-backend -- train multitask --help
```

This enables the differentiable relational model modules. The default build
remains dependency-free and uses the reference backend; LibTorch builds add
the native multi-task training and inference path. Verify the installed
LibTorch version against the pinned `tch` dependency before running a real
training job.
