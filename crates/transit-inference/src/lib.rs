//! Fast inference and ranking over a compiled graph.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use transit_graph::GraphTensor;
use transit_model::{
    denormalize_criticality_targets, LineEmbedding, MaskSelection,
    ReferenceLineRepresentationEncoder, CRITICALITY_OUTPUTS,
};
use transit_training::ReferenceCheckpoint;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinePrediction {
    pub line: u32,
    pub metrics: Vec<f32>,
    #[serde(rename = "structuralUniqueness", alias = "structural_uniqueness")]
    pub structural_uniqueness: f32,
    #[serde(
        rename = "metricPercentiles",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub metric_percentiles: Vec<f32>,
    /// Named form of `metrics`, included for clients that should not depend
    /// on output ordering. The vector remains for backwards compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criticality: Option<CriticalityPrediction>,
    #[serde(default)]
    pub uncertainty: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriticalityPrediction {
    pub accessibility_loss: f32,
    pub unreachable_share: f32,
    pub mean_delay_seconds: f32,
    pub p95_delay_seconds: f32,
    pub extra_transfers: f32,
    pub isolated_station_share: f32,
    pub uncertainty: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineInference {
    pub snapshot_id: String,
    pub line_index: u32,
    pub embedding: LineEmbedding,
    pub criticality: CriticalityPrediction,
    pub anomaly_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineEmbeddingRecord {
    pub line: u32,
    pub line_name: String,
    pub embedding: LineEmbedding,
    pub anomaly_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionFile {
    #[serde(rename = "schemaVersion", default = "inference_schema_version")]
    pub schema_version: u32,
    #[serde(rename = "modelId", default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(rename = "snapshotId", alias = "snapshot_id")]
    pub snapshot_id: String,
    #[serde(rename = "metricNames", alias = "metric_names")]
    pub metric_names: Vec<String>,
    pub predictions: Vec<LinePrediction>,
    #[serde(
        rename = "lineNames",
        alias = "line_names",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub line_names: BTreeMap<String, String>,
    #[serde(
        rename = "lineEmbeddings",
        alias = "line_embeddings",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub line_embeddings: Vec<LineEmbeddingRecord>,
}

fn inference_schema_version() -> u32 {
    1
}

pub fn predict_reference(
    checkpoint: &ReferenceCheckpoint,
    graph: &GraphTensor,
) -> Result<PredictionFile> {
    let Some(head) = &checkpoint.head else {
        bail!("checkpoint has no criticality head");
    };
    let mask = MaskSelection::all_unmasked(graph);
    let embeddings = checkpoint.encoder.encode(graph, &mask)?;
    let structural = checkpoint
        .encoder
        .structural_uniqueness_scores(graph, &mask)?;
    let representation_set = if let Some(representation) = checkpoint.representation.as_ref() {
        representation.encode(graph, &embeddings)?
    } else {
        ReferenceLineRepresentationEncoder::default().encode(graph, &embeddings)?
    };
    let raw_predictions = if checkpoint.representation.is_some() {
        (0..graph.manifest.line_count)
            .map(|line| {
                let input = head.input_for_representation(
                    representation_set
                        .lines
                        .get(line)
                        .ok_or_else(|| anyhow::anyhow!("missing representation for line {line}"))?,
                    &representation_set.city,
                    graph,
                    line,
                )?;
                head.predict_inputs(&input)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        head.predict_all(&embeddings, graph)?
    };
    let predictions: Vec<LinePrediction> = raw_predictions
        .into_iter()
        .enumerate()
        .map(|(line, metrics)| {
            let metrics: Vec<f32> = denormalize_criticality_targets(metrics)
                .into_iter()
                .map(|value| value.max(0.0))
                .collect();
            let criticality = criticality_from_metrics(&metrics, 0.0);
            LinePrediction {
                line: line as u32,
                metrics,
                structural_uniqueness: structural.get(line).copied().unwrap_or(0.0),
                metric_percentiles: Vec::new(),
                criticality: Some(criticality),
                uncertainty: 0.0,
            }
        })
        .collect();
    let line_names = graph
        .line_names
        .iter()
        .enumerate()
        .map(|(line, name)| (line.to_string(), name.clone()))
        .collect();
    let line_embeddings = representation_set
        .lines
        .into_iter()
        .enumerate()
        .map(|(line, embedding)| LineEmbeddingRecord {
            line: line as u32,
            line_name: graph
                .line_names
                .get(line)
                .cloned()
                .unwrap_or_else(|| format!("Line {line}")),
            embedding,
            anomaly_score: structural.get(line).copied().unwrap_or(0.0),
        })
        .collect();
    let mut output = PredictionFile {
        schema_version: inference_schema_version(),
        model_id: None,
        snapshot_id: graph.manifest.snapshot_id.clone(),
        metric_names: vec![
            "accessibility_auc_loss".into(),
            "unreachable_share".into(),
            "mean_delay_reachable_seconds".into(),
            "p95_delay_reachable_seconds".into(),
            "mean_extra_transfers".into(),
            "stations_losing_all_service_share".into(),
        ],
        predictions,
        line_names,
        line_embeddings,
    };
    add_metric_percentiles(&mut output);
    Ok(output)
}

/// Add empirical percentiles to the Rust-owned inference result so clients do
/// not need to derive model-facing ranks from raw predictions.
pub fn add_metric_percentiles(predictions: &mut PredictionFile) {
    let row_count = predictions.predictions.len();
    if row_count == 0 {
        return;
    }
    for metric_index in 0..predictions.metric_names.len() {
        let mut values: Vec<f32> = predictions
            .predictions
            .iter()
            .filter_map(|prediction| prediction.metrics.get(metric_index).copied())
            .collect();
        values.sort_by(f32::total_cmp);
        for prediction in &mut predictions.predictions {
            let value = prediction.metrics.get(metric_index).copied().unwrap_or(0.0);
            let rank = values
                .iter()
                .filter(|candidate| **candidate <= value)
                .count();
            if prediction.metric_percentiles.len() < predictions.metric_names.len() {
                prediction.metric_percentiles = vec![0.0; predictions.metric_names.len()];
            }
            prediction.metric_percentiles[metric_index] = rank as f32 / row_count as f32;
        }
    }
}

fn criticality_from_metrics(metrics: &[f32], uncertainty: f32) -> CriticalityPrediction {
    CriticalityPrediction {
        accessibility_loss: metrics.first().copied().unwrap_or(0.0),
        unreachable_share: metrics.get(1).copied().unwrap_or(0.0),
        mean_delay_seconds: metrics.get(2).copied().unwrap_or(0.0),
        p95_delay_seconds: metrics.get(3).copied().unwrap_or(0.0),
        extra_transfers: metrics.get(4).copied().unwrap_or(0.0),
        isolated_station_share: metrics.get(5).copied().unwrap_or(0.0),
        uncertainty,
    }
}

pub fn rank_by_accessibility(predictions: &mut PredictionFile) {
    predictions.predictions.sort_by(|left, right| {
        right
            .metrics
            .first()
            .copied()
            .unwrap_or(0.0)
            .total_cmp(&left.metrics.first().copied().unwrap_or(0.0))
    });
}

pub fn save_predictions(path: &Path, predictions: &PredictionFile) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(predictions).context("encoding predictions")?,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn load_predictions(path: &Path) -> Result<PredictionFile> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).context("decoding predictions")
}

pub fn metric_count(predictions: &PredictionFile) -> usize {
    predictions
        .predictions
        .first()
        .map(|prediction| prediction.metrics.len())
        .unwrap_or(CRITICALITY_OUTPUTS)
}
