# Transit Lab web wrapper

This is a small, responsive Bun web app for exploring line criticality
predictions. It has no runtime dependencies. The browser can read a local JSON
file or fetch a JSON response from an API URL.

## Run it

From this directory:

```bash
bun run dev
```

Open <http://localhost:3000>, then choose a generated prediction file. The
Rust demo writes one at `../data/demo/predictions.json` when run from this
directory's parent.

Other commands:

```bash
bun test
bun run check
PORT=4173 bun run server.js
```

The API option expects the endpoint to return JSON with the shape below. A
cross-origin endpoint must allow requests from the wrapper's origin.

```json
{
  "snapshot_id": "snapshot-hash",
  "metric_names": [
    "accessibility_auc_loss",
    "unreachable_share",
    "mean_delay_reachable_seconds",
    "p95_delay_reachable_seconds",
    "mean_extra_transfers",
    "stations_losing_all_service_share"
  ],
  "predictions": [
    {
      "line": 2,
      "metrics": [0.12, 0.08, 95.5, 180.0, 0.4, 0.06],
      "structural_uniqueness": 0.71
    }
  ]
}
```

Metric values ending in `_share` or `_loss` use fractions from `0` to `1` and
appear as percentages in the UI. Delay metrics use seconds. An optional
`line_name` on each prediction, or a top-level `line_names` object keyed by
line ID, gives the list a passenger-facing label. Without one, the UI shows
`Line <id>`.

The `?api=` query parameter can prefill and load an API URL, for example:

```text
http://localhost:3000/?api=http%3A%2F%2Flocalhost%3A3001%2Fpredictions
```

## What the UI does

- Summarizes the loaded snapshot and its highest predicted accessibility loss.
- Ranks lines by accessibility loss by default.
- Searches by line ID or name, sorts by another metric, and filters to positive
  or above-average impact.
- Opens a tap-friendly detail view with every predicted metric and structural
  uniqueness.
- Reports loading, invalid-input, empty-run, and no-match states.
