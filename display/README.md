# Transit Lab display

`display` is a small, dependency-free WebGL viewer for compiled Transit Lab
snapshots. It is meant for looking across feeds and understanding the shape of
each network before digging into model output.

## Run it

From this directory:

```bash
bun run dev
```

Open <http://localhost:3001>. If a demo snapshot exists at `data/demo`, the
viewer prefers a compiled Vienna snapshot and otherwise opens the demo.
Generate the demo from the repository root with:

```bash
cargo run -p transit-cli -- demo --output data/demo
```

If the Vienna raw feed is already present, compile it with:

```bash
cargo run -p transit-cli -- compile \
  --input data/raw/vienna/*/gtfs.zip \
  --service-date 2026-09-07 \
  --output data/snapshots/vienna
```

The viewer can also open a local folder containing a compiled
`snapshot/network.json`. Nothing from a local folder is uploaded. A snapshot
can contain a country, city, or regional feed. Its source name, geographical
scope, service date, station coordinates, route geometry, and graph counts are
shown in the app.

Use `?snapshot=/data/snapshots/vienna/network.json` to select a
specific snapshot by URL. An explicit URL takes priority over the Vienna
default.

## What the view shows

- Real station coordinates projected into a rotatable 3D scene.
- Route links colored by line, with route visibility checkboxes.
- Station height based on network role, daily departures, or service span.
- Station hover details and click-to-inspect station or line metrics.
- Local snapshot discovery under `data/**/snapshot/network.json`.

The binary graph arrays are not needed for this first display. The snapshot
JSON keeps the viewer readable and preserves the metadata people need when
comparing countries.

```bash
bun test
```
