//! Dataset manifests joining graph tensors and simulator-generated labels.

use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use transit_domain::{hex_digest, sha256_bytes, CompiledNetwork};
use transit_graph::GraphTensor;
use transit_labels::{load_jsonl, save_jsonl, LineImpactLabel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetManifest {
    #[serde(rename = "schemaVersion", default = "dataset_schema_version")]
    pub schema_version: u32,
    #[serde(rename = "datasetId", default)]
    pub dataset_id: String,
    #[serde(default)]
    pub fingerprint: String,
    #[serde(rename = "featureSchema", default)]
    pub feature_schema: String,
    #[serde(rename = "snapshotIds", default)]
    pub snapshot_ids: Vec<String>,
    #[serde(default)]
    pub split: serde_json::Value,
    #[serde(default)]
    pub objectives: serde_json::Value,
    #[serde(rename = "createdAt", default)]
    pub created_at: String,
    #[serde(rename = "producingRunId", default)]
    pub producing_run_id: Option<String>,
    #[serde(rename = "inputArtifacts", default)]
    pub input_artifacts: Vec<serde_json::Value>,
    #[serde(rename = "snapshotId", alias = "snapshot_id", default)]
    pub snapshot_id: String,
    #[serde(rename = "graphSchemaVersion", alias = "graph_schema_version", default)]
    pub graph_schema_version: String,
    #[serde(rename = "labelCount", alias = "label_count", default)]
    pub label_count: usize,
    #[serde(rename = "labelFile", alias = "label_file", default)]
    pub label_file: String,
    #[serde(rename = "graphDirectory", alias = "graph_directory", default)]
    pub graph_directory: String,
    #[serde(default)]
    pub entries: Vec<DatasetEntry>,
    #[serde(rename = "examplesFile", alias = "examples_file", default)]
    pub examples_file: Option<String>,
    #[serde(rename = "exampleCount", alias = "example_count", default)]
    pub example_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetEntry {
    #[serde(rename = "snapshotId")]
    pub snapshot_id: String,
    #[serde(rename = "networkSystemId", default)]
    pub network_system_id: String,
    #[serde(rename = "graphDirectory")]
    pub graph_directory: String,
    #[serde(rename = "labelFile")]
    pub label_file: String,
    #[serde(rename = "labelCount")]
    pub label_count: usize,
    #[serde(default)]
    pub split: String,
}

/// An already materialized graph and its immutable simulator labels. The
/// output paths are part of the dataset identity, so a caller can copy the
/// graph and label files before writing the manifest.
pub struct DatasetPart<'a> {
    pub graph: &'a GraphTensor,
    pub labels: &'a [LineImpactLabel],
    pub graph_directory: String,
    pub label_file: String,
    pub split: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetExample {
    #[serde(rename = "snapshotId")]
    pub snapshot_id: String,
    #[serde(rename = "lineIndex")]
    pub line_index: usize,
    pub split: String,
    #[serde(rename = "lineIdentity", skip_serializing_if = "Option::is_none")]
    pub line_identity: Option<String>,
    pub targets: BTreeMap<String, f32>,
}

fn dataset_schema_version() -> u32 {
    1
}

#[derive(Clone, Debug)]
pub struct LoadedDataset {
    pub manifest: DatasetManifest,
    pub graph: GraphTensor,
    pub labels: Vec<LineImpactLabel>,
    pub network_system_id: String,
    pub split: String,
}

pub fn save_dataset(
    network: &CompiledNetwork,
    graph: &GraphTensor,
    labels: &[LineImpactLabel],
    directory: &Path,
) -> Result<DatasetManifest> {
    fs::create_dir_all(directory).with_context(|| format!("creating {}", directory.display()))?;
    let graph_directory = directory.join("graph");
    graph.save(&graph_directory, network)?;
    let label_path = directory.join("labels.jsonl");
    save_jsonl(&label_path, labels)?;
    let fingerprint_input = serde_json::json!({
        "snapshotId": network.snapshot_id,
        "graphSchema": graph.manifest.schema_version,
        "labels": labels,
    });
    let fingerprint = hex_digest(&sha256_bytes(&serde_json::to_vec(&fingerprint_input)?));
    let manifest = DatasetManifest {
        schema_version: dataset_schema_version(),
        dataset_id: format!("dataset-{}", &fingerprint[..24]),
        fingerprint,
        feature_schema: graph.manifest.schema_version.clone(),
        snapshot_ids: vec![network.snapshot_id.clone()],
        split: serde_json::json!({"strategy": "snapshot"}),
        objectives: serde_json::json!({"lineImpactLabels": labels.len()}),
        created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        producing_run_id: std::env::var("TRANSIT_RUN_ID").ok(),
        input_artifacts: Vec::new(),
        snapshot_id: network.snapshot_id.clone(),
        graph_schema_version: graph.manifest.schema_version.clone(),
        label_count: labels.len(),
        label_file: "labels.jsonl".into(),
        graph_directory: "graph".into(),
        entries: vec![DatasetEntry {
            snapshot_id: network.snapshot_id.clone(),
            network_system_id: graph.manifest.network_system_id.clone(),
            graph_directory: "graph".into(),
            label_file: "labels.jsonl".into(),
            label_count: labels.len(),
            split: "train".into(),
        }],
        examples_file: None,
        example_count: labels.len(),
    };
    write_immutable_json(&directory.join("dataset-manifest.json"), &manifest)?;
    Ok(manifest)
}

/// Build the common manifest for a multi-city dataset. File copying and
/// example materialization remain in the CLI so this crate can also be used by
/// library callers with their own artifact layout.
pub fn create_dataset_manifest(
    parts: &[DatasetPart<'_>],
    split: serde_json::Value,
    examples_file: Option<String>,
    example_count: usize,
    input_artifacts: Vec<serde_json::Value>,
    producing_run_id: Option<String>,
) -> Result<DatasetManifest> {
    let Some(first) = parts.first() else {
        anyhow::bail!("cannot create a dataset without graph parts");
    };
    if parts.iter().any(|part| {
        part.graph.manifest.schema_version != first.graph.manifest.schema_version
            || part.graph.line_features.cols != first.graph.line_features.cols
            || part.graph.station_features.cols != first.graph.station_features.cols
    }) {
        anyhow::bail!("dataset graph parts have incompatible feature schemas");
    }
    let entries = parts
        .iter()
        .map(|part| DatasetEntry {
            snapshot_id: part.graph.manifest.snapshot_id.clone(),
            network_system_id: part.graph.manifest.network_system_id.clone(),
            graph_directory: part.graph_directory.clone(),
            label_file: part.label_file.clone(),
            label_count: part.labels.len(),
            split: part.split.clone(),
        })
        .collect::<Vec<_>>();
    let mut snapshots = std::collections::BTreeSet::new();
    if entries
        .iter()
        .any(|entry| !snapshots.insert(&entry.snapshot_id))
    {
        anyhow::bail!("dataset graph parts must have unique snapshot IDs");
    }
    let objectives = serde_json::json!({
        "lineImpactLabels": parts.iter().map(|part| part.labels.len()).sum::<usize>(),
        "graphs": parts.len(),
        "examples": example_count
    });
    let content = serde_json::json!({
        "featureSchema": first.graph.manifest.schema_version,
        "entries": entries.clone(),
        "split": split.clone(),
        "objectives": objectives.clone(),
        "inputArtifacts": input_artifacts.clone()
    });
    let fingerprint = hex_digest(&sha256_bytes(&serde_json::to_vec(&content)?));
    Ok(DatasetManifest {
        schema_version: dataset_schema_version(),
        dataset_id: format!("dataset-{}", &fingerprint[..24]),
        fingerprint,
        feature_schema: first.graph.manifest.schema_version.clone(),
        snapshot_ids: parts
            .iter()
            .map(|part| part.graph.manifest.snapshot_id.clone())
            .collect(),
        split,
        objectives,
        created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        producing_run_id,
        input_artifacts,
        snapshot_id: first.graph.manifest.snapshot_id.clone(),
        graph_schema_version: first.graph.manifest.schema_version.clone(),
        label_count: first.labels.len(),
        label_file: first.label_file.clone(),
        graph_directory: first.graph_directory.clone(),
        entries,
        examples_file,
        example_count,
    })
}

pub fn save_dataset_manifest(path: &Path, manifest: &DatasetManifest) -> Result<()> {
    write_immutable_json(path, manifest)
}

fn write_immutable_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let encoded = serde_json::to_vec_pretty(value).context("encoding dataset manifest")?;
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(&encoded)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(path)?;
            let mut expected = encoded.clone();
            expected.push(b'\n');
            if existing == encoded || existing == expected {
                Ok(())
            } else {
                anyhow::bail!(
                    "refusing to overwrite immutable dataset manifest {}",
                    path.display()
                )
            }
        }
        Err(error) => Err(error.into()),
    }
}

pub fn load_labels(directory: &Path) -> Result<Vec<LineImpactLabel>> {
    load_jsonl(&directory.join("labels.jsonl"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn dataset_path(directory: &Path, raw: &str, field: &str) -> Result<PathBuf> {
    if raw.trim().is_empty() {
        anyhow::bail!("dataset {field} cannot be blank");
    }
    let relative = Path::new(raw);
    if raw.contains('\\')
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        anyhow::bail!("dataset {field} must be a relative path without dot segments");
    }
    let base = fs::canonicalize(directory)
        .with_context(|| format!("resolving dataset directory {}", directory.display()))?;
    let candidate = directory.join(relative);
    let canonical = fs::canonicalize(&candidate)
        .with_context(|| format!("resolving dataset {field} {}", candidate.display()))?;
    if !canonical.starts_with(&base) {
        anyhow::bail!("dataset {field} escapes the dataset directory");
    }
    Ok(candidate)
}

fn validate_manifest_header(manifest: &DatasetManifest) -> Result<()> {
    if manifest.schema_version != dataset_schema_version() {
        anyhow::bail!(
            "unsupported dataset manifest schema {}; expected {}",
            manifest.schema_version,
            dataset_schema_version()
        );
    }
    if manifest.dataset_id.trim().is_empty() {
        anyhow::bail!("dataset manifest dataset ID cannot be blank");
    }
    if !is_sha256(&manifest.fingerprint) {
        anyhow::bail!("dataset manifest fingerprint must be a SHA-256 hex digest");
    }
    if manifest.dataset_id != format!("dataset-{}", &manifest.fingerprint[..24]) {
        anyhow::bail!("dataset manifest dataset ID does not match its fingerprint");
    }
    if manifest.feature_schema.trim().is_empty() {
        anyhow::bail!("dataset manifest feature schema cannot be blank");
    }
    if manifest.snapshot_ids.is_empty() {
        anyhow::bail!("dataset manifest must name at least one snapshot");
    }
    let mut snapshot_ids = std::collections::BTreeSet::new();
    if manifest
        .snapshot_ids
        .iter()
        .any(|snapshot| snapshot.trim().is_empty() || !snapshot_ids.insert(snapshot))
    {
        anyhow::bail!("dataset manifest snapshot IDs must be non-empty and unique");
    }
    if manifest.graph_schema_version.trim().is_empty() {
        anyhow::bail!("dataset manifest graph schema version cannot be blank");
    }
    if manifest.graph_schema_version != manifest.feature_schema {
        anyhow::bail!("dataset manifest feature and graph schemas differ");
    }
    Ok(())
}

fn validate_label_rows(
    labels: &[LineImpactLabel],
    snapshot_id: &str,
    line_count: usize,
    field: &str,
) -> Result<()> {
    let mut lines = std::collections::BTreeSet::new();
    for label in labels {
        if label.snapshot != snapshot_id {
            anyhow::bail!("{field} contains labels from a different snapshot");
        }
        if label.line.0 as usize >= line_count {
            anyhow::bail!("{field} contains a line outside graph {snapshot_id}");
        }
        if !lines.insert(label.line)
            || ![
                label.accessibility_auc_loss,
                label.unreachable_share,
                label.mean_delay_reachable_seconds,
                label.p95_delay_reachable_seconds,
                label.mean_extra_transfers,
                label.stations_losing_all_service_share,
            ]
            .into_iter()
            .all(f32::is_finite)
        {
            anyhow::bail!("{field} contains duplicate lines or non-finite targets");
        }
    }
    Ok(())
}

fn validate_split(split: &str, field: &str) -> Result<()> {
    if !matches!(split, "train" | "validation" | "test") {
        anyhow::bail!("{field} must be train, validation, or test");
    }
    Ok(())
}

fn common_manifest_fingerprint(manifest: &DatasetManifest) -> Result<String> {
    let content = serde_json::json!({
        "featureSchema": manifest.feature_schema,
        "entries": manifest.entries,
        "split": manifest.split,
        "objectives": manifest.objectives,
        "inputArtifacts": manifest.input_artifacts
    });
    Ok(hex_digest(&sha256_bytes(&serde_json::to_vec(&content)?)))
}

fn legacy_manifest_fingerprint(
    manifest: &DatasetManifest,
    labels: &[LineImpactLabel],
) -> Result<String> {
    let content = serde_json::json!({
        "snapshotId": manifest.snapshot_id,
        "graphSchema": manifest.graph_schema_version,
        "labels": labels
    });
    Ok(hex_digest(&sha256_bytes(&serde_json::to_vec(&content)?)))
}

fn validate_examples(
    directory: &Path,
    manifest: &DatasetManifest,
    graphs: &[(String, GraphTensor)],
) -> Result<()> {
    let Some(examples_file) = manifest.examples_file.as_deref() else {
        if manifest.example_count != 0
            && !(manifest.entries.len() == 1
                && manifest.input_artifacts.is_empty()
                && manifest.split == serde_json::json!({"strategy": "snapshot"}))
        {
            anyhow::bail!("dataset example count is non-zero but examples file is missing");
        }
        return Ok(());
    };
    let path = dataset_path(directory, examples_file, "examplesFile")?;
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let examples: Vec<DatasetExample> =
        serde_json::from_slice(&bytes).with_context(|| format!("decoding {}", path.display()))?;
    if examples.len() != manifest.example_count {
        anyhow::bail!("dataset example count does not match examples file");
    }
    for example in &examples {
        let Some((_, graph)) = graphs
            .iter()
            .find(|(snapshot_id, _)| snapshot_id == &example.snapshot_id)
        else {
            anyhow::bail!("dataset example refers to an unknown snapshot");
        };
        if example.line_index >= graph.manifest.line_count {
            anyhow::bail!("dataset example line is outside its graph");
        }
        validate_split(&example.split, "dataset example split")?;
        let entry = manifest
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == example.snapshot_id)
            .context("dataset example has no manifest entry")?;
        if entry.split != example.split {
            anyhow::bail!("dataset example split does not match its manifest entry");
        }
    }
    Ok(())
}

fn load_manifest(directory: &Path) -> Result<DatasetManifest> {
    let manifest_path = directory.join("dataset-manifest.json");
    let manifest_bytes =
        fs::read(&manifest_path).with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: DatasetManifest =
        serde_json::from_slice(&manifest_bytes).context("decoding dataset manifest")?;
    validate_manifest_header(&manifest)?;
    Ok(manifest)
}

fn load_dataset_entry(
    directory: &Path,
    manifest: &DatasetManifest,
    snapshot_id: &str,
    graph_directory: &str,
    label_file: &str,
    label_count: usize,
    split: &str,
    network_system_id: &str,
) -> Result<LoadedDataset> {
    validate_split(split, "dataset entry split")?;
    let graph_path = dataset_path(directory, graph_directory, "graphDirectory")?;
    if !graph_path.is_dir() {
        anyhow::bail!("dataset graph directory is not a directory");
    }
    let label_path = dataset_path(directory, label_file, "labelFile")?;
    if !label_path.is_file() {
        anyhow::bail!("dataset label file is not a file");
    }
    let graph = GraphTensor::load(&graph_path)
        .with_context(|| format!("loading dataset graph {}", graph_path.display()))?;
    let labels = load_jsonl(&label_path)
        .with_context(|| format!("loading dataset labels {}", label_path.display()))?;
    if graph.manifest.snapshot_id != snapshot_id {
        anyhow::bail!("dataset entry and graph have different snapshot IDs");
    }
    if graph.manifest.schema_version != manifest.feature_schema {
        anyhow::bail!("dataset graph and manifest have different feature schemas");
    }
    if !network_system_id.is_empty() && network_system_id != graph.manifest.network_system_id {
        anyhow::bail!("dataset entry network system does not match its graph");
    }
    if labels.len() != label_count {
        anyhow::bail!("dataset entry label count does not match its label file");
    }
    validate_label_rows(
        &labels,
        snapshot_id,
        graph.manifest.line_count,
        "dataset labels",
    )?;
    let graph_network_system_id = graph.manifest.network_system_id.clone();
    let network_system_id = if network_system_id.is_empty() {
        graph_network_system_id
    } else {
        network_system_id.to_owned()
    };
    Ok(LoadedDataset {
        manifest: manifest.clone(),
        graph,
        labels,
        network_system_id,
        split: split.to_owned(),
    })
}

pub fn load_dataset(directory: &Path) -> Result<LoadedDataset> {
    let collection = load_dataset_collection(directory)?;
    if collection.entries.len() != 1 {
        anyhow::bail!("dataset contains multiple graph entries; use load_dataset_collection");
    }
    Ok(collection
        .entries
        .into_iter()
        .next()
        .expect("collection length was checked"))
}

#[derive(Clone, Debug)]
pub struct LoadedDatasetCollection {
    pub manifest: DatasetManifest,
    pub entries: Vec<LoadedDataset>,
}

pub fn load_dataset_collection(directory: &Path) -> Result<LoadedDatasetCollection> {
    let manifest = load_manifest(directory)?;
    let legacy_layout = manifest.entries.is_empty()
        || (manifest.entries.len() == 1
            && manifest.examples_file.is_none()
            && manifest.input_artifacts.is_empty()
            && manifest.split == serde_json::json!({"strategy": "snapshot"}));
    let raw_entries = if manifest.entries.is_empty() {
        if manifest.snapshot_ids.len() != 1
            || manifest.snapshot_id.trim().is_empty()
            || manifest.graph_directory.trim().is_empty()
            || manifest.label_file.trim().is_empty()
        {
            anyhow::bail!("legacy dataset manifest must describe one complete graph entry");
        }
        vec![DatasetEntry {
            snapshot_id: manifest.snapshot_id.clone(),
            network_system_id: String::new(),
            graph_directory: manifest.graph_directory.clone(),
            label_file: manifest.label_file.clone(),
            label_count: manifest.label_count,
            split: "train".into(),
        }]
    } else {
        manifest.entries.clone()
    };
    let mut snapshot_ids = std::collections::BTreeSet::new();
    let mut graph_paths = std::collections::BTreeSet::new();
    let mut label_paths = std::collections::BTreeSet::new();
    for entry in &raw_entries {
        if entry.snapshot_id.trim().is_empty()
            || !snapshot_ids.insert(entry.snapshot_id.clone())
            || !graph_paths.insert(entry.graph_directory.clone())
            || !label_paths.insert(entry.label_file.clone())
        {
            anyhow::bail!("dataset entries must have unique snapshots and artifact paths");
        }
        validate_split(&entry.split, "dataset entry split")?;
        let _ = dataset_path(directory, &entry.graph_directory, "graphDirectory")?;
        let _ = dataset_path(directory, &entry.label_file, "labelFile")?;
    }
    let manifest_snapshot_ids = manifest
        .snapshot_ids
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if snapshot_ids != manifest_snapshot_ids {
        anyhow::bail!("dataset manifest snapshot IDs do not match its entries");
    }
    if !legacy_layout && raw_entries.len() != manifest.entries.len() {
        anyhow::bail!("dataset manifest contains no entries");
    }
    let mut entries = Vec::with_capacity(raw_entries.len());
    let mut graph_pairs = Vec::with_capacity(raw_entries.len());
    for entry in &raw_entries {
        let loaded = load_dataset_entry(
            directory,
            &manifest,
            &entry.snapshot_id,
            &entry.graph_directory,
            &entry.label_file,
            entry.label_count,
            &entry.split,
            &entry.network_system_id,
        )?;
        graph_pairs.push((entry.snapshot_id.clone(), loaded.graph.clone()));
        entries.push(loaded);
    }
    if manifest.entries.len() == 1 {
        let entry = &manifest.entries[0];
        if manifest.snapshot_id != entry.snapshot_id
            || manifest.graph_directory != entry.graph_directory
            || manifest.label_file != entry.label_file
            || manifest.label_count != entry.label_count
        {
            anyhow::bail!("single-entry dataset manifest has inconsistent legacy fields");
        }
    }
    validate_examples(directory, &manifest, &graph_pairs)?;
    let labels = &entries[0].labels;
    let expected_fingerprint = if legacy_layout {
        legacy_manifest_fingerprint(&manifest, labels)?
    } else {
        common_manifest_fingerprint(&manifest)?
    };
    if manifest.fingerprint != expected_fingerprint {
        anyhow::bail!("dataset manifest fingerprint does not match its contents");
    }
    Ok(LoadedDatasetCollection { manifest, entries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use gtfs_compile::{compile, CompileOptions};
    use gtfs_ingest::GtfsFeed;
    use transit_graph::GraphTensor;

    fn fixture() -> (CompiledNetwork, GraphTensor) {
        let feed = GtfsFeed::from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/synthetic-feeds/basic"
        ))
        .expect("synthetic fixture loads");
        let network = compile(
            &feed,
            &CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap())
                .with_scope("synthetic fixture")
                .with_source_name("fixture"),
        )
        .expect("synthetic fixture compiles");
        let graph = GraphTensor::from_network(&network).expect("synthetic graph builds");
        (network, graph)
    }

    fn labels(snapshot: &str, count: usize) -> Vec<LineImpactLabel> {
        (0..count)
            .map(|line| LineImpactLabel {
                snapshot: snapshot.to_owned(),
                line: transit_domain::LineIndex(line as u32),
                accessibility_auc_loss: line as f32,
                unreachable_share: 0.0,
                mean_delay_reachable_seconds: 0.0,
                p95_delay_reachable_seconds: 0.0,
                mean_extra_transfers: 0.0,
                stations_losing_all_service_share: 0.0,
                query_count: 1,
                policy_fingerprint: "policy".into(),
            })
            .collect()
    }

    #[test]
    fn saved_dataset_round_trips_with_manifest_integrity_checks() {
        let (network, graph) = fixture();
        let directory = tempfile::tempdir().unwrap();
        let rows = labels(&network.snapshot_id, network.lines.len());
        save_dataset(&network, &graph, &rows, directory.path()).unwrap();

        let loaded = load_dataset_collection(directory.path()).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].labels.len(), rows.len());
        assert_eq!(loaded.manifest.snapshot_ids, vec![network.snapshot_id]);
    }

    #[test]
    fn dataset_loader_rejects_traversal_and_tampered_fingerprints() {
        let (network, graph) = fixture();
        let directory = tempfile::tempdir().unwrap();
        let rows = labels(&network.snapshot_id, network.lines.len());
        save_dataset(&network, &graph, &rows, directory.path()).unwrap();

        let manifest_path = directory.path().join("dataset-manifest.json");
        let mut manifest: DatasetManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.graph_directory = "../outside".into();
        manifest.entries[0].graph_directory = "../outside".into();
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert!(load_dataset_collection(directory.path())
            .unwrap_err()
            .to_string()
            .contains("relative path"));

        manifest.graph_directory = "graph".into();
        manifest.entries[0].graph_directory = "graph".into();
        manifest.fingerprint = "0".repeat(64);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert!(load_dataset_collection(directory.path())
            .unwrap_err()
            .to_string()
            .contains("fingerprint"));
    }
}
