# Transit Lab

Transit Lab is a Rust-first GTFS graph learning pipeline. It compiles a
service-day snapshot into canonical station, line, pattern, and timetable data;
runs timetable-aware line-removal simulations; and exposes a masked relational
graph autoencoder interface for representation learning.

The repository is intentionally staged around a small, testable vertical slice:

1. Parse a GTFS directory or ZIP.
2. Select active service for one date and preserve GTFS times beyond 24:00.
3. Canonicalize physical stations, passenger-facing lines, and trip patterns.
4. Materialize compact graph arrays and a JSON manifest.
5. Run a RAPTOR-style one-to-all router with line masks.
6. Generate aggregate single-line disruption labels in parallel.
7. Produce a shared base line embedding plus separate general, network-role,
   service, geometry, and resilience facets.
8. Retrieve comparable lines across snapshots with measured comparison fields.
9. Run the reference Rust model path, or enable the optional LibTorch backend.

## Quick start

Install Rust 1.75 or newer, then run:

```bash
cargo test --workspace
cargo run -p transit-cli -- demo --output data/demo
```

The demo command builds a deterministic synthetic network, generates line
removal labels, and writes a graph plus label manifest under `data/demo`.

For a real feed:

```bash
cargo run -p transit-cli -- validate --input path/to/gtfs.zip
cargo run -p transit-cli -- compile --input path/to/gtfs.zip \
  --service-date 2026-09-09 --output data/snapshots/example
cargo run -p transit-cli -- graph build \
  --snapshot data/snapshots/example --output data/graphs/example
cargo run -p transit-cli -- labels line-removal \
  --snapshot data/snapshots/example --output data/labels/example.jsonl

# Compare a line against another compiled graph
cargo run -p transit-cli -- similar-lines \
  --query-graph data/graphs/vienna \
  --query-line U2 \
  --candidate-graph data/graphs/berlin \
  --profile network-role --top-k 10
```

### Multi-task training

Compile one graph per feed or snapshot, then pass all graphs to the shared
training command. Labels are optional for representation pretraining and line
retrieval; supply one JSONL file per graph to train the disruption-impact head.
The label files follow the graph order.

```bash
cargo run -p transit-cli -- train multitask \
  --graph data/graphs/vienna \
  --graph data/graphs/vbb \
  --labels data/labels/vienna.jsonl \
  --labels data/labels/vbb.jsonl \
  --config configs/models/multitask-v1.yaml \
  --output data/models/multitask-v1.json

cargo run -p transit-cli -- infer criticality \
  --graph data/graphs/vbb \
  --model data/models/multitask-v1.json \
  --output data/runs/vbb-predictions.json

cargo run -p transit-cli -- similar-lines \
  --query-graph data/graphs/vienna \
  --query-line U2 \
  --candidate-graph data/graphs/vbb \
  --encoder data/models/multitask-v1.json \
  --profile network-role --top-k 10
```

The command is also available as `train representation`. To retrieve by a
different meaning, use `service`, `geometry`, `resilience`, or `general`. A
weighted profile takes role, service, geometry, and resilience weights in that
order:

```bash
cargo run -p transit-cli -- similar-lines \
  --query-graph data/graphs/vienna --query-line U2 \
  --candidate-graph data/graphs/vbb \
  --encoder data/models/multitask-v1.json \
  --profile weighted:0.5,0.2,0.2,0.1
```

If individual `--*-weight` flags are used, omitted facets receive weight zero.
The output includes every facet score and measured comparison fields, so the
neural score is not the explanation.

For the checked-in feed registry, `fetch` stores an immutable ZIP and
`source.json` under `data/raw/<feed>/<sha256>/`. Compile the ZIP inside that
directory so the source metadata is carried into the snapshot:

```bash
cargo run -p transit-cli -- fetch vienna --output data/raw
cargo run -p transit-cli -- compile \
  --input data/raw/vienna/<sha256>/gtfs.zip \
  --service-date 2026-09-09 \
  --output data/snapshots/vienna
cargo run -p transit-cli -- graph build \
  --snapshot data/snapshots/vienna --output data/graphs/vienna
cargo run -p transit-cli -- labels line-removal \
  --snapshot data/snapshots/vienna --output data/labels/vienna.jsonl
```

Label generation is exact timetable simulation and can be the expensive step.
Reduce `--origins` and `--departure-times` for a smoke run, or use several
dates and feeds for snapshot-consistency training.

The responsive web wrapper is under [`web-wrapper`](web-wrapper). Start it
with `bun run dev`, then load the JSON emitted by `infer criticality` or use a
prediction API URL. It displays ranked impact, all disruption metrics,
structural uniqueness, and line names while ignoring the optional embedding
payload.

The interactive 3D graph display is under [`display`](display). Start it with
`bun run dev` from that directory to inspect compiled snapshots, isolate lines,
orbit the network, and compare station and service metrics across feeds. It
prefers a local Vienna snapshot when one is present, then falls back to
`data/demo/snapshot/network.json`. You can also choose a local snapshot folder
containing `network.json`.

The `tch-backend` feature is optional because LibTorch is not bundled in the
repository. Enable it on a Linux/NVIDIA machine with a compatible LibTorch
installation:

```bash
LIBTORCH=/opt/libtorch cargo build -p transit-model --features tch-backend
```

The default reference model is dependency-free and deterministic. It is useful
for smoke tests, structural scores, and data-pipeline validation; it is not a
replacement for GPU training.

## Data contracts

Snapshot manifests record source hashes, selected service date, scope, station
merge policy, line grouping policy, compiler version, and validation details.
Compiled snapshots are immutable by convention: changing any of those inputs
creates a new snapshot directory and ID.

Raw IDs and operator strings remain in lookup/manifest data only. Model feature
arrays use normalized numeric values and mode indicators to reduce feed-producer
and city-name leakage.

Graph manifests currently use schema `station-line-relational-v2`. In addition
to the station, line, timetable, and relation arrays, each graph stores ordered
canonical route patterns as CSR-like arrays:

```text
pattern_offsets.u32          pattern_stops.u32
pattern_lines.u32            pattern_directions.u32
pattern_trip_counts.u32      pattern_stop_features.f32
pattern_segment_features.f32
```

`pattern_offsets[p]..pattern_offsets[p + 1]` indexes the ordered stops and
stop features for pattern `p`; segment rows follow the same order. This is the
sequence input used by the pattern encoder and preserves direction, branch,
pickup, and drop-off information that a station-line bipartite graph alone
would lose. Graphs supplied to one multitask run must have compatible feature
schemas. The `lookup_*.json` files retain display metadata for explanations but
are not model inputs.

The default `reference-cpu-multitask` backend is deterministic and dependency
free. It is intended to validate the graph, training, checkpoint, and retrieval
workflow on a CPU. The optional LibTorch backend remains the path for a fully
trainable neural encoder and GPU-scale runs.
