PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);

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
  spec_json TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
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
  error_code TEXT,
  error_message TEXT,
  started_at TEXT,
  finished_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS runs_fingerprint_idx ON runs(fingerprint);

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
  name TEXT NOT NULL,
  value REAL NOT NULL,
  split TEXT,
  network_id TEXT REFERENCES networks(id),
  dimensions_json TEXT NOT NULL DEFAULT '{}',
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
