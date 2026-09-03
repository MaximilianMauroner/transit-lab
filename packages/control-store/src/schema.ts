import type { Database } from "bun:sqlite";

/**
 * The control store uses schema push instead of a migration history. The
 * schema is declarative and every statement is safe to run against an empty
 * or already-populated local database.
 */
export const CONTROL_STORE_SCHEMA_VERSION = 5;

export const CONTROL_STORE_SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS networks (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  display_name TEXT NOT NULL,
  geographical_scope TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feed_revisions (
  id TEXT PRIMARY KEY,
  network_id TEXT NOT NULL REFERENCES networks(id),
  source_url TEXT NOT NULL DEFAULT '',
  landing_page TEXT NOT NULL DEFAULT '',
  downloaded_at TEXT,
  valid_from TEXT,
  valid_to TEXT,
  sha256 TEXT NOT NULL,
  byte_count INTEGER NOT NULL DEFAULT 0,
  licence TEXT,
  geographical_scope TEXT NOT NULL DEFAULT '',
  local_path TEXT NOT NULL,
  validation_status TEXT NOT NULL DEFAULT 'unknown',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE(network_id, sha256)
);

CREATE TABLE IF NOT EXISTS snapshots (
  id TEXT PRIMARY KEY,
  network_id TEXT NOT NULL REFERENCES networks(id),
  feed_revision_id TEXT REFERENCES feed_revisions(id),
  service_date TEXT NOT NULL,
  service_profile TEXT NOT NULL DEFAULT 'selected-day',
  status TEXT NOT NULL DEFAULT 'ready',
  fingerprint TEXT NOT NULL UNIQUE,
  compiler_version TEXT NOT NULL DEFAULT '',
  compiler_commit TEXT NOT NULL DEFAULT '',
  source_name TEXT NOT NULL DEFAULT '',
  geographical_scope TEXT NOT NULL DEFAULT '',
  manifest_path TEXT NOT NULL,
  network_path TEXT NOT NULL,
  graph_path TEXT,
  counts_json TEXT NOT NULL DEFAULT '{}',
  validation_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_lines (
  id TEXT PRIMARY KEY,
  network_id TEXT NOT NULL REFERENCES networks(id),
  canonical_name TEXT NOT NULL,
  mode TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS line_instances (
  id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  canonical_line_id TEXT REFERENCES canonical_lines(id),
  line_index INTEGER NOT NULL,
  canonical_id TEXT NOT NULL,
  display_name TEXT NOT NULL,
  agency_key TEXT NOT NULL DEFAULT '',
  mode INTEGER NOT NULL DEFAULT 0,
  feature_json TEXT NOT NULL DEFAULT '{}',
  geometry_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(snapshot_id, line_index)
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  fingerprint TEXT NOT NULL UNIQUE,
  uri TEXT NOT NULL,
  local_path TEXT,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  sha256 TEXT,
  schema_version INTEGER,
  producing_run_id TEXT,
  git_commit TEXT NOT NULL DEFAULT '',
  configuration_json TEXT NOT NULL DEFAULT '{}',
  files_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'ready',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  superseded_by TEXT REFERENCES artifacts(id)
);

CREATE TABLE IF NOT EXISTS artifact_dependencies (
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  depends_on_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  relation TEXT NOT NULL DEFAULT 'input',
  PRIMARY KEY (artifact_id, depends_on_artifact_id, relation)
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id),
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  desired_state TEXT NOT NULL DEFAULT 'running',
  observed_state TEXT NOT NULL DEFAULT 'queued',
  spec_json TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  config_fingerprint TEXT NOT NULL DEFAULT '',
  resolved_config_path TEXT NOT NULL DEFAULT '',
  snapshot_id TEXT REFERENCES snapshots(id),
  dataset_id TEXT,
  model_id TEXT,
  progress_completed INTEGER NOT NULL DEFAULT 0,
  progress_total INTEGER NOT NULL DEFAULT 0,
  progress_unit TEXT NOT NULL DEFAULT '',
  current_step TEXT NOT NULL DEFAULT '',
  worker_id TEXT,
  git_commit TEXT NOT NULL DEFAULT '',
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  latest_checkpoint_id TEXT,
  current_attempt_id TEXT,
  parent_run_id TEXT REFERENCES runs(id),
  resume_checkpoint_id TEXT,
  checkpoint_root TEXT NOT NULL DEFAULT '',
  control_file_path TEXT NOT NULL DEFAULT '',
  phase TEXT NOT NULL DEFAULT '',
  global_step INTEGER NOT NULL DEFAULT 0,
  resume_not_before TEXT,
  total_compute_seconds REAL NOT NULL DEFAULT 0,
  paused_seconds REAL NOT NULL DEFAULT 0,
  paused_since TEXT,
  schedule_json TEXT NOT NULL DEFAULT '{}',
  last_heartbeat_at TEXT,
  error_code TEXT,
  error_message TEXT,
  started_at TEXT,
  finished_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id),
  step TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  input_fingerprint TEXT,
  output_fingerprint TEXT,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(run_id, step)
);

CREATE TABLE IF NOT EXISTS run_attempts (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  ordinal INTEGER NOT NULL,
  worker_id TEXT,
  resume_checkpoint_id TEXT,
  status TEXT NOT NULL,
  exit_reason TEXT,
  hostname TEXT NOT NULL DEFAULT '',
  device_json TEXT NOT NULL DEFAULT '{}',
  started_at TEXT NOT NULL,
  finished_at TEXT,
  last_heartbeat_at TEXT,
  compute_seconds REAL NOT NULL DEFAULT 0,
  UNIQUE(run_id, ordinal)
);

CREATE TABLE IF NOT EXISTS training_checkpoints (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  attempt_id TEXT REFERENCES run_attempts(id),
  phase TEXT NOT NULL,
  global_step INTEGER NOT NULL,
  local_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  config_fingerprint TEXT NOT NULL,
  dataset_fingerprint TEXT NOT NULL,
  git_commit TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE(run_id, global_step)
);

CREATE TABLE IF NOT EXISTS run_events (
  run_id TEXT NOT NULL REFERENCES runs(id),
  seq INTEGER NOT NULL,
  event_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(run_id, seq)
);

CREATE TABLE IF NOT EXISTS run_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id),
  stream TEXT NOT NULL,
  line TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY,
  hostname TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'idle',
  current_run_id TEXT,
  last_heartbeat_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'ready',
  manifest_path TEXT NOT NULL,
  feature_schema TEXT NOT NULL DEFAULT '',
  snapshot_ids_json TEXT NOT NULL DEFAULT '[]',
  split_json TEXT NOT NULL DEFAULT '{}',
  objective_counts_json TEXT NOT NULL DEFAULT '{}',
  quality_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  fingerprint TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'ready',
  architecture_json TEXT NOT NULL DEFAULT '{}',
  dataset_id TEXT REFERENCES datasets(id),
  training_run_id TEXT REFERENCES runs(id),
  checkpoint_artifact_id TEXT REFERENCES artifacts(id),
  embedding_dimensions_json TEXT NOT NULL DEFAULT '{}',
  supported_heads_json TEXT NOT NULL DEFAULT '[]',
  evaluation_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_aliases (
  alias TEXT PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES model_versions(id),
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_sets (
  id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL UNIQUE,
  model_id TEXT NOT NULL REFERENCES model_versions(id),
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  status TEXT NOT NULL DEFAULT 'ready',
  embeddings_artifact_id TEXT REFERENCES artifacts(id),
  criticality_artifact_id TEXT REFERENCES artifacts(id),
  projection_artifact_id TEXT REFERENCES artifacts(id),
  config_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS criticality_predictions (
  inference_id TEXT NOT NULL REFERENCES inference_sets(id),
  line_instance_id TEXT NOT NULL REFERENCES line_instances(id),
  primary_score REAL,
  uncertainty REAL,
  values_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  PRIMARY KEY(inference_id, line_instance_id)
);

CREATE TABLE IF NOT EXISTS criticality_labels (
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
  line_index INTEGER NOT NULL,
  values_json TEXT NOT NULL DEFAULT '{}',
  source_artifact_id TEXT REFERENCES artifacts(id),
  created_at TEXT NOT NULL,
  PRIMARY KEY(snapshot_id, line_index)
);

CREATE TABLE IF NOT EXISTS metric_points (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(id),
  model_id TEXT REFERENCES model_versions(id),
  dataset_id TEXT REFERENCES datasets(id),
  evaluation_id TEXT,
  name TEXT NOT NULL,
  value REAL NOT NULL,
  split TEXT,
  network_id TEXT REFERENCES networks(id),
  dimensions_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
  id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL UNIQUE,
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  dataset_id TEXT NOT NULL REFERENCES datasets(id),
  model_id TEXT REFERENCES model_versions(id),
  split TEXT NOT NULL,
  top_k INTEGER NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
  id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL UNIQUE,
  artifact_id TEXT NOT NULL REFERENCES artifacts(id),
  run_id TEXT REFERENCES runs(id),
  benchmark TEXT NOT NULL,
  workload TEXT NOT NULL,
  snapshot_id TEXT,
  graph_id TEXT,
  thread_count INTEGER,
  warmup_units INTEGER NOT NULL DEFAULT 0,
  measured_units INTEGER NOT NULL DEFAULT 0,
  estimated_work_units INTEGER,
  median_milliseconds REAL,
  p95_milliseconds REAL,
  throughput REAL NOT NULL,
  throughput_unit TEXT NOT NULL,
  peak_resident_memory_bytes INTEGER,
  graph_counts_json TEXT NOT NULL DEFAULT '{}',
  runtime_json TEXT NOT NULL DEFAULT '{}',
  thread_configuration_json TEXT NOT NULL DEFAULT '{}',
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_checks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  actual_value REAL,
  threshold_value REAL,
  details_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE(target_type, target_id, name)
);

CREATE TABLE IF NOT EXISTS similarity_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id TEXT REFERENCES model_versions(id),
  dataset_id TEXT REFERENCES datasets(id),
  facet TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  value REAL NOT NULL,
  split TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  anchor_line_instance_id TEXT NOT NULL REFERENCES line_instances(id),
  candidate_a_line_instance_id TEXT NOT NULL REFERENCES line_instances(id),
  candidate_b_line_instance_id TEXT NOT NULL REFERENCES line_instances(id),
  facet TEXT NOT NULL,
  choice TEXT NOT NULL,
  confidence TEXT,
  notes TEXT NOT NULL DEFAULT '',
  annotator TEXT NOT NULL DEFAULT 'local',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_views (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS publications (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'published',
  manifest_path TEXT NOT NULL DEFAULT '',
  snapshot_ids_json TEXT NOT NULL DEFAULT '[]',
  model_ids_json TEXT NOT NULL DEFAULT '[]',
  artifact_ids_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
`;

type ColumnDefinition = {
  table: string;
  name: string;
  definition: string;
};

const COMPATIBILITY_COLUMNS: ColumnDefinition[] = [
  // These columns are present in the shipped schema, but keeping them here
  // also lets a damaged/partial local runs table be repaired by a push.
  { table: "runs", name: "snapshot_id", definition: "TEXT" },
  { table: "runs", name: "dataset_id", definition: "TEXT" },
  { table: "runs", name: "model_id", definition: "TEXT" },
  { table: "runs", name: "progress_completed", definition: "INTEGER NOT NULL DEFAULT 0" },
  { table: "runs", name: "progress_total", definition: "INTEGER NOT NULL DEFAULT 0" },
  { table: "runs", name: "progress_unit", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "current_step", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "worker_id", definition: "TEXT" },
  { table: "runs", name: "git_commit", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "cancel_requested", definition: "INTEGER NOT NULL DEFAULT 0" },
  { table: "runs", name: "error_code", definition: "TEXT" },
  { table: "runs", name: "error_message", definition: "TEXT" },
  { table: "runs", name: "started_at", definition: "TEXT" },
  { table: "runs", name: "finished_at", definition: "TEXT" },
  { table: "runs", name: "created_at", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "updated_at", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "config_fingerprint", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "resolved_config_path", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "desired_state", definition: "TEXT NOT NULL DEFAULT 'running'" },
  { table: "runs", name: "observed_state", definition: "TEXT NOT NULL DEFAULT 'queued'" },
  { table: "runs", name: "latest_checkpoint_id", definition: "TEXT" },
  { table: "runs", name: "current_attempt_id", definition: "TEXT" },
  { table: "runs", name: "parent_run_id", definition: "TEXT" },
  { table: "runs", name: "resume_checkpoint_id", definition: "TEXT" },
  { table: "runs", name: "checkpoint_root", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "control_file_path", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "phase", definition: "TEXT NOT NULL DEFAULT ''" },
  { table: "runs", name: "global_step", definition: "INTEGER NOT NULL DEFAULT 0" },
  { table: "runs", name: "resume_not_before", definition: "TEXT" },
  { table: "runs", name: "total_compute_seconds", definition: "REAL NOT NULL DEFAULT 0" },
  { table: "runs", name: "paused_seconds", definition: "REAL NOT NULL DEFAULT 0" },
  { table: "runs", name: "paused_since", definition: "TEXT" },
  { table: "runs", name: "schedule_json", definition: "TEXT NOT NULL DEFAULT '{}'" },
  { table: "runs", name: "last_heartbeat_at", definition: "TEXT" },
  { table: "run_attempts", name: "compute_seconds", definition: "REAL NOT NULL DEFAULT 0" },
  { table: "model_versions", name: "dataset_id", definition: "TEXT REFERENCES datasets(id)" },
  { table: "model_versions", name: "training_run_id", definition: "TEXT REFERENCES runs(id)" },
  { table: "model_versions", name: "checkpoint_artifact_id", definition: "TEXT REFERENCES artifacts(id)" },
  { table: "model_versions", name: "embedding_dimensions_json", definition: "TEXT NOT NULL DEFAULT '{}'" },
  { table: "model_versions", name: "supported_heads_json", definition: "TEXT NOT NULL DEFAULT '[]'" },
  { table: "model_versions", name: "evaluation_json", definition: "TEXT NOT NULL DEFAULT '{}'" },
  { table: "metric_points", name: "evaluation_id", definition: "TEXT" }
];

function tableColumns(db: Database, table: string) {
  return new Set(db.query(`PRAGMA table_info(${table})`).all().map((column: { name: string }) => column.name));
}

/** Apply the current declarative schema and add columns missing from older local databases. */
export function pushDatabaseSchema(db: Database) {
  db.exec("PRAGMA foreign_keys = ON;");
  db.exec(CONTROL_STORE_SCHEMA_SQL);
  for (const column of COMPATIBILITY_COLUMNS) {
    if (!tableColumns(db, column.table).has(column.name)) {
      db.exec(`ALTER TABLE ${column.table} ADD COLUMN ${column.name} ${column.definition}`);
    }
  }
  // Existing databases used `status` as the only lifecycle field. Backfill
  // the new observed state once, without rewriting an explicitly populated
  // value from a newer schema.
  db.exec(`UPDATE runs SET observed_state = status
    WHERE observed_state = 'queued' AND status <> 'queued'`);
  db.exec(`UPDATE runs SET desired_state = CASE WHEN cancel_requested = 1 THEN 'cancelled' ELSE 'running' END
    WHERE desired_state = 'running' OR desired_state IS NULL`);
  db.exec("CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status, created_at);");
  db.exec("CREATE INDEX IF NOT EXISTS runs_fingerprint_idx ON runs(fingerprint);");
  db.exec("CREATE INDEX IF NOT EXISTS runs_observed_state_idx ON runs(observed_state, created_at);");
  db.exec("CREATE INDEX IF NOT EXISTS run_attempts_run_idx ON run_attempts(run_id, ordinal);");
  db.exec("CREATE INDEX IF NOT EXISTS training_checkpoints_run_step_idx ON training_checkpoints(run_id, global_step);");
  db.exec("CREATE INDEX IF NOT EXISTS metric_points_evaluation_idx ON metric_points(evaluation_id, created_at);");
  db.exec("CREATE INDEX IF NOT EXISTS evaluation_results_dataset_model_idx ON evaluation_results(dataset_id, model_id, created_at);");
  db.exec("CREATE INDEX IF NOT EXISTS benchmarks_workload_snapshot_idx ON benchmarks(workload, snapshot_id, created_at);");
  db.exec("CREATE INDEX IF NOT EXISTS benchmarks_graph_workload_idx ON benchmarks(graph_id, workload, created_at);");
  return { version: CONTROL_STORE_SCHEMA_VERSION };
}
