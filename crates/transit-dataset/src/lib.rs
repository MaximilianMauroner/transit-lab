//! Dataset manifests joining graph tensors and simulator-generated labels.

use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
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
    pub snapshot_id: String,
    pub graph_schema_version: String,
    pub label_count: usize,
    pub label_file: String,
    pub graph_directory: String,
}

fn dataset_schema_version() -> u32 {
    1
}

#[derive(Clone, Debug)]
pub struct LoadedDataset {
    pub manifest: DatasetManifest,
    pub graph: GraphTensor,
    pub labels: Vec<LineImpactLabel>,
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
    };
    fs::write(
        directory.join("dataset-manifest.json"),
        serde_json::to_vec_pretty(&manifest).context("encoding dataset manifest")?,
    )?;
    Ok(manifest)
}

pub fn load_labels(directory: &Path) -> Result<Vec<LineImpactLabel>> {
    load_jsonl(&directory.join("labels.jsonl"))
}

pub fn load_dataset(directory: &Path) -> Result<LoadedDataset> {
    let manifest_bytes = fs::read(directory.join("dataset-manifest.json"))
        .with_context(|| format!("reading {}/dataset-manifest.json", directory.display()))?;
    let manifest: DatasetManifest =
        serde_json::from_slice(&manifest_bytes).context("decoding dataset manifest")?;
    let graph = GraphTensor::load(&directory.join(&manifest.graph_directory))?;
    let labels = load_jsonl(&directory.join(&manifest.label_file))?;
    if manifest.snapshot_id != graph.manifest.snapshot_id {
        anyhow::bail!("dataset manifest and graph have different snapshot IDs");
    }
    if !manifest.snapshot_ids.is_empty() && !manifest.snapshot_ids.contains(&manifest.snapshot_id) {
        anyhow::bail!("dataset manifest snapshot IDs do not contain the dataset snapshot");
    }
    if manifest.graph_schema_version != graph.manifest.schema_version {
        anyhow::bail!("dataset manifest and graph have different schema versions");
    }
    if manifest.label_count != labels.len() {
        anyhow::bail!("dataset manifest label count does not match the label file");
    }
    if labels
        .iter()
        .any(|label| label.snapshot != manifest.snapshot_id)
    {
        anyhow::bail!("dataset contains labels from a different snapshot");
    }
    Ok(LoadedDataset {
        manifest,
        graph,
        labels,
    })
}
