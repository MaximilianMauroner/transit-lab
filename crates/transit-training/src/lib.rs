//! Training orchestration for the dependency-free reference backend.
//!
//! The optional LibTorch backend is exposed by `transit-model`; this crate
//! keeps dataset iteration, masking schedules, city balancing, and checkpoints
//! backend-independent.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use transit_graph::GraphTensor;
use transit_labels::LineImpactLabel;
use transit_model::{
    normalize_criticality_targets, CriticalityHead, Embeddings, MaskConfig, MaskSelection,
    ModelConfig, ProjectionHead, RawLineFeatures, ReferenceLineRepresentationEncoder,
    ReferenceRelationalAutoencoder, RepresentationConfig, TrainableLineRepresentationModel,
    CRITICALITY_OUTPUTS,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PretrainingConfig {
    pub model: ModelConfig,
    pub mask: MaskConfig,
    pub steps: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

impl Default for PretrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            mask: MaskConfig::default(),
            steps: 200,
            learning_rate: 0.001,
            weight_decay: 0.00001,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriticalityTrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub ranking_weight: f32,
    pub seed: u64,
    #[serde(default = "default_max_ranking_pairs")]
    pub max_ranking_pairs: usize,
}

fn default_max_ranking_pairs() -> usize {
    512
}

impl Default for CriticalityTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.001,
            ranking_weight: 0.5,
            seed: 19,
            max_ranking_pairs: default_max_ranking_pairs(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiTaskTrainingConfig {
    pub pretraining: PretrainingConfig,
    pub representation: RepresentationConfig,
    pub metric_epochs: usize,
    pub metric_learning_rate: f32,
    pub metric_margin: f32,
    pub metric_weight_decay: f32,
    pub max_triplets: usize,
    pub criticality: CriticalityTrainingConfig,
}

impl Default for MultiTaskTrainingConfig {
    fn default() -> Self {
        Self {
            pretraining: PretrainingConfig::default(),
            representation: RepresentationConfig::default(),
            metric_epochs: 8,
            metric_learning_rate: 0.002,
            metric_margin: 0.25,
            metric_weight_decay: 0.00001,
            max_triplets: 512,
            criticality: CriticalityTrainingConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingReport {
    pub backend: String,
    pub steps: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceCheckpoint {
    pub encoder: ReferenceRelationalAutoencoder,
    pub head: Option<CriticalityHead>,
    pub report: Option<TrainingReport>,
    #[serde(default)]
    pub representation: Option<TrainableLineRepresentationModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiTaskTrainingReport {
    pub backend: String,
    pub dataset_count: usize,
    pub line_count: usize,
    pub pretraining: TrainingReport,
    pub metric_epochs: usize,
    pub metric_initial_loss: f32,
    pub metric_final_loss: f32,
    pub metric_triplets: usize,
    pub criticality: Option<TrainingReport>,
}

pub fn load_config<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("yaml") | Some("yml")
    ) {
        Ok(serde_yaml::from_slice(&bytes).context("decoding YAML config")?)
    } else {
        Ok(serde_json::from_slice(&bytes).context("decoding JSON config")?)
    }
}

pub fn train_reference_autoencoder(
    graph: &GraphTensor,
    config: &PretrainingConfig,
) -> Result<(ReferenceRelationalAutoencoder, TrainingReport)> {
    let mut model = ReferenceRelationalAutoencoder::new(config.model.clone());
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    for step in 0..config.steps {
        let mask =
            MaskSelection::sample(graph, &config.mask, config.seed.wrapping_add(step as u64));
        let loss = model.train_decoder_step(graph, &mask, config.learning_rate)?;
        if step == 0 {
            initial_loss = loss;
        }
        final_loss = loss;
        // Weight decay is applied to decoder parameters inside a full tensor
        // backend. The reference decoder intentionally keeps its update loop
        // small; retaining the config field preserves checkpoint compatibility.
        let _ = config.weight_decay;
    }
    Ok((
        model,
        TrainingReport {
            backend: "reference-cpu-decoder".into(),
            steps: config.steps,
            initial_loss,
            final_loss,
        },
    ))
}

/// Pretrain one shared encoder over multiple city/snapshot graphs. A step is
/// assigned round-robin so each dataset contributes equally even when feeds
/// have very different numbers of entities.
pub fn train_reference_autoencoder_multi(
    graphs: &[&GraphTensor],
    config: &PretrainingConfig,
) -> Result<(ReferenceRelationalAutoencoder, TrainingReport)> {
    let Some(first_graph) = graphs.first() else {
        anyhow::bail!("no graph datasets were provided");
    };
    for graph in graphs {
        graph.validate()?;
        validate_graph_schema(first_graph, graph)?;
    }
    let mut model = ReferenceRelationalAutoencoder::new(config.model.clone());
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    for step in 0..config.steps {
        let graph = graphs[step % graphs.len()];
        let mask = MaskSelection::sample(
            graph,
            &config.mask,
            config
                .seed
                .wrapping_add(step as u64)
                .wrapping_add((step % graphs.len()) as u64 * 7919),
        );
        let loss = model.train_decoder_step(graph, &mask, config.learning_rate)?;
        if step == 0 {
            initial_loss = loss;
        }
        final_loss = loss;
    }
    Ok((
        model,
        TrainingReport {
            backend: "reference-cpu-decoder-multi-dataset".into(),
            steps: config.steps,
            initial_loss,
            final_loss,
        },
    ))
}

fn validate_graph_schema(first: &GraphTensor, candidate: &GraphTensor) -> Result<()> {
    let same_schema = first.manifest.schema_version == candidate.manifest.schema_version
        && first.manifest.temporal_bins == candidate.manifest.temporal_bins
        && first.manifest.temporal_bin_seconds == candidate.manifest.temporal_bin_seconds
        && first.manifest.station_feature_names == candidate.manifest.station_feature_names
        && first.manifest.line_feature_names == candidate.manifest.line_feature_names
        && first.manifest.temporal_channel_names == candidate.manifest.temporal_channel_names
        && first.manifest.transit_edge_feature_names
            == candidate.manifest.transit_edge_feature_names
        && first.manifest.transfer_feature_names == candidate.manifest.transfer_feature_names
        && first.manifest.pattern_stop_feature_names
            == candidate.manifest.pattern_stop_feature_names
        && first.manifest.pattern_segment_feature_names
            == candidate.manifest.pattern_segment_feature_names;
    if !same_schema {
        anyhow::bail!("graph datasets have incompatible feature schemas");
    }
    Ok(())
}

pub fn save_checkpoint(path: &Path, checkpoint: &ReferenceCheckpoint) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(checkpoint).context("encoding model checkpoint")?,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<ReferenceCheckpoint> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).context("decoding model checkpoint")
}

pub fn train_criticality_head(
    encoder: &ReferenceRelationalAutoencoder,
    graph: &GraphTensor,
    labels: &[LineImpactLabel],
    config: &CriticalityTrainingConfig,
) -> Result<(CriticalityHead, TrainingReport)> {
    train_criticality_head_multi(encoder, &[(graph, labels)], config)
}

/// Train one shared head over several city/snapshot graphs. Each graph is
/// encoded independently, labels are restricted to that graph's snapshot ID,
/// and examples from each snapshot receive equal total weight.
pub fn train_criticality_head_multi(
    encoder: &ReferenceRelationalAutoencoder,
    datasets: &[(&GraphTensor, &[LineImpactLabel])],
    config: &CriticalityTrainingConfig,
) -> Result<(CriticalityHead, TrainingReport)> {
    let Some((first_graph, _)) = datasets.first() else {
        anyhow::bail!("no graph datasets were provided");
    };
    let first_mask = MaskSelection::all_unmasked(first_graph);
    let first_embeddings = encoder.encode(first_graph, &first_mask)?;
    let input_dimension = first_embeddings.line.first().map(Vec::len).unwrap_or(0) * 2
        + first_graph.line_features.cols;
    let mut head = CriticalityHead::new(input_dimension, CRITICALITY_OUTPUTS, config.seed);
    let mut examples = Vec::new();
    for (graph, labels) in datasets {
        if graph.line_features.cols != first_graph.line_features.cols {
            anyhow::bail!("graph datasets have incompatible line feature widths");
        }
        let mask = MaskSelection::all_unmasked(graph);
        let embeddings = encoder.encode(graph, &mask)?;
        examples.extend(training_examples(&head, &embeddings, graph, labels)?);
    }
    if examples.is_empty() {
        anyhow::bail!("no labels match the graph snapshot datasets");
    }

    let (initial_loss, final_loss) = fit_criticality_head(&mut head, &examples, config)?;
    Ok((
        head,
        TrainingReport {
            backend: "reference-cpu-head".into(),
            steps: config.epochs,
            initial_loss,
            final_loss,
        },
    ))
}

/// Train the complete dependency-free multi-task workflow over one or more
/// compiled snapshots. Exact simulator labels are optional for pretraining
/// and retrieval, but are required if a criticality head is requested.
pub fn train_reference_multitask(
    datasets: &[(&GraphTensor, &[LineImpactLabel])],
    config: &MultiTaskTrainingConfig,
) -> Result<(ReferenceCheckpoint, MultiTaskTrainingReport)> {
    let Some((first_graph, _)) = datasets.first() else {
        anyhow::bail!("no graph datasets were provided");
    };
    let graphs: Vec<&GraphTensor> = datasets.iter().map(|(graph, _)| *graph).collect();
    let (encoder, pretraining_report) =
        train_reference_autoencoder_multi(&graphs, &config.pretraining)?;
    let mut embeddings = Vec::with_capacity(graphs.len());
    for graph in &graphs {
        embeddings.push(encoder.encode(graph, &MaskSelection::all_unmasked(graph))?);
    }
    let representation = {
        let extractor = ReferenceLineRepresentationEncoder::new(config.representation.clone());
        let raw = extractor.raw_features(first_graph, &embeddings[0])?;
        TrainableLineRepresentationModel::from_raw_features(&raw, config.representation.clone())?
    };
    let representation_inputs: Vec<(&GraphTensor, &Embeddings, &[LineImpactLabel])> = datasets
        .iter()
        .zip(&embeddings)
        .map(|((graph, labels), embedding)| (*graph, embedding, *labels))
        .collect();
    let samples = collect_representation_samples(&representation_inputs, &representation)?;
    let mut representation = representation;
    let (metric_initial_loss, metric_final_loss, metric_triplets) =
        fit_metric_heads(&mut representation, &samples, config)?;
    let criticality = if datasets.iter().any(|(_, labels)| !labels.is_empty()) {
        Some(train_criticality_head_multi_representation(
            &representation,
            &representation_inputs,
            &config.criticality,
        )?)
    } else {
        None
    };
    let criticality_report = criticality.as_ref().map(|(_, report)| report.clone());
    let checkpoint = ReferenceCheckpoint {
        encoder,
        head: criticality.map(|(head, _)| head),
        report: Some(pretraining_report.clone()),
        representation: Some(representation),
    };
    let report = MultiTaskTrainingReport {
        backend: "reference-cpu-multitask".into(),
        dataset_count: datasets.len(),
        line_count: samples.len(),
        pretraining: pretraining_report,
        metric_epochs: config.metric_epochs,
        metric_initial_loss,
        metric_final_loss,
        metric_triplets,
        criticality: criticality_report,
    };
    Ok((checkpoint, report))
}

/// Train a criticality head against the learned base representation. This
/// keeps the legacy `train_criticality_head_multi` API available for old
/// checkpoints while making the multi-task checkpoint use the requested
/// `[base | city | measured line features]` input.
pub fn train_criticality_head_multi_representation(
    representation: &TrainableLineRepresentationModel,
    datasets: &[(&GraphTensor, &Embeddings, &[LineImpactLabel])],
    config: &CriticalityTrainingConfig,
) -> Result<(CriticalityHead, TrainingReport)> {
    let Some((first_graph, first_embeddings, _)) = datasets.first() else {
        anyhow::bail!("no graph datasets were provided");
    };
    let first_representations = representation.encode(first_graph, first_embeddings)?;
    let Some(first_line) = first_representations.lines.first() else {
        anyhow::bail!("cannot train criticality without line representations");
    };
    let input_dimension =
        first_line.base.len() + first_representations.city.len() + first_graph.line_features.cols;
    let mut head = CriticalityHead::new(input_dimension, CRITICALITY_OUTPUTS, config.seed);
    let mut snapshot_counts = HashMap::<String, usize>::new();
    for (graph, _, labels) in datasets {
        for label in labels.iter().filter(|label| {
            label.snapshot == graph.manifest.snapshot_id
                && (label.line.0 as usize) < graph.manifest.line_count
        }) {
            *snapshot_counts.entry(label.snapshot.clone()).or_default() += 1;
        }
    }
    let mut examples = Vec::new();
    for (graph, embeddings, labels) in datasets {
        if graph.line_features.cols != first_graph.line_features.cols {
            anyhow::bail!("graph datasets have incompatible line feature widths");
        }
        let representations = representation.encode(graph, embeddings)?;
        for label in labels
            .iter()
            .filter(|label| label.snapshot == graph.manifest.snapshot_id)
        {
            let line = label.line.0 as usize;
            let Some(embedding) = representations.lines.get(line) else {
                continue;
            };
            let input =
                head.input_for_representation(embedding, &representations.city, graph, line)?;
            let weight = 1.0 / snapshot_counts.get(&label.snapshot).copied().unwrap_or(1) as f32;
            examples.push(Example {
                input,
                target: normalize_criticality_targets(label_targets(label)),
                weight,
                snapshot: label.snapshot.clone(),
            });
        }
    }
    if examples.is_empty() {
        anyhow::bail!("no labels match the graph snapshot datasets");
    }
    let (initial_loss, final_loss) = fit_criticality_head(&mut head, &examples, config)?;
    Ok((
        head,
        TrainingReport {
            backend: "reference-cpu-representation-head".into(),
            steps: config.epochs,
            initial_loss,
            final_loss,
        },
    ))
}

fn label_targets(label: &LineImpactLabel) -> [f32; CRITICALITY_OUTPUTS] {
    [
        label.accessibility_auc_loss,
        label.unreachable_share,
        label.mean_delay_reachable_seconds,
        label.p95_delay_reachable_seconds,
        label.mean_extra_transfers,
        label.stations_losing_all_service_share,
    ]
}

#[derive(Clone, Copy, Debug)]
enum MetricFacet {
    Base,
    General,
    Role,
    Service,
    Geometry,
    Resilience,
}

impl MetricFacet {
    const ALL: [Self; 6] = [
        Self::Base,
        Self::General,
        Self::Role,
        Self::Service,
        Self::Geometry,
        Self::Resilience,
    ];
}

struct RepresentationSample {
    raw: RawLineFeatures,
    line_key: String,
    snapshot: String,
    criticality: Option<[f32; CRITICALITY_OUTPUTS]>,
}

fn collect_representation_samples(
    datasets: &[(&GraphTensor, &Embeddings, &[LineImpactLabel])],
    representation: &TrainableLineRepresentationModel,
) -> Result<Vec<RepresentationSample>> {
    let extractor = ReferenceLineRepresentationEncoder::new(representation.config.clone());
    let mut samples = Vec::new();
    for (graph, embeddings, labels) in datasets {
        let raw = extractor.raw_features(graph, embeddings)?;
        let mut label_by_line = HashMap::<u32, [f32; CRITICALITY_OUTPUTS]>::new();
        for label in labels
            .iter()
            .filter(|label| label.snapshot == graph.manifest.snapshot_id)
        {
            label_by_line.insert(label.line.0, label_targets(label));
        }
        for (line, raw_line) in raw.lines.into_iter().enumerate() {
            let line_key = graph
                .line_names
                .get(line)
                .map(|name| name.trim().to_ascii_lowercase())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| format!("line-{line}"));
            samples.push(RepresentationSample {
                raw: raw_line,
                line_key,
                snapshot: graph.manifest.snapshot_id.clone(),
                criticality: label_by_line.get(&(line as u32)).copied(),
            });
        }
    }
    Ok(samples)
}

fn fit_metric_heads(
    representation: &mut TrainableLineRepresentationModel,
    samples: &[RepresentationSample],
    config: &MultiTaskTrainingConfig,
) -> Result<(f32, f32, usize)> {
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    let triplets_by_facet: Vec<(MetricFacet, Vec<[usize; 3]>)> = MetricFacet::ALL
        .into_iter()
        .map(|facet| (facet, build_triplets(samples, facet, config.max_triplets)))
        .collect();
    let total_triplets = triplets_by_facet
        .iter()
        .map(|(_, triplets)| triplets.len())
        .sum();
    for epoch in 0..config.metric_epochs {
        let mut loss_sum = 0.0_f64;
        let mut loss_count = 0_usize;
        for (facet, triplets) in &triplets_by_facet {
            // Inputs depend on lower-level projections (role/resilience/general),
            // so refresh them once per facet and epoch rather than once per
            // triplet. This is the difference between a useful CPU smoke run
            // and repeated O(lines * dimensions) work.
            let inputs: Vec<Vec<f32>> = samples
                .iter()
                .map(|sample| facet_input(representation, sample, *facet))
                .collect::<Result<Vec<_>>>()?;
            for [anchor, positive, negative] in triplets {
                let anchor_input = &inputs[*anchor];
                let positive_input = &inputs[*positive];
                let negative_input = &inputs[*negative];
                let (anchor_raw, positive_raw, negative_raw) = {
                    let head = facet_head(representation, *facet);
                    (
                        head.forward_raw(anchor_input)?,
                        head.forward_raw(positive_input)?,
                        head.forward_raw(negative_input)?,
                    )
                };
                let anchor_output = normalized(&anchor_raw);
                let positive_output = normalized(&positive_raw);
                let negative_output = normalized(&negative_raw);
                let positive_distance = squared_distance(&anchor_output, &positive_output);
                let negative_distance = squared_distance(&anchor_output, &negative_output);
                let loss = config.metric_margin + positive_distance - negative_distance;
                if !loss.is_finite() || loss <= 0.0 {
                    continue;
                }
                let anchor_gradient: Vec<f32> = anchor_output
                    .iter()
                    .zip(&positive_output)
                    .zip(&negative_output)
                    .map(|((_anchor, positive), negative)| 2.0 * (negative - positive))
                    .collect();
                let positive_gradient: Vec<f32> = positive_output
                    .iter()
                    .zip(&anchor_output)
                    .map(|(positive, anchor)| 2.0 * (positive - anchor))
                    .collect();
                let negative_gradient: Vec<f32> = anchor_output
                    .iter()
                    .zip(&negative_output)
                    .map(|(anchor, negative)| 2.0 * (anchor - negative))
                    .collect();
                let anchor_gradient = normalization_gradient(&anchor_raw, &anchor_gradient);
                let positive_gradient = normalization_gradient(&positive_raw, &positive_gradient);
                let negative_gradient = normalization_gradient(&negative_raw, &negative_gradient);
                let head = facet_head_mut(representation, *facet);
                head.apply_gradient_from_activated(
                    anchor_input,
                    &anchor_raw,
                    &anchor_gradient,
                    config.metric_learning_rate,
                    config.metric_weight_decay,
                )?;
                head.apply_gradient_from_activated(
                    positive_input,
                    &positive_raw,
                    &positive_gradient,
                    config.metric_learning_rate,
                    config.metric_weight_decay,
                )?;
                head.apply_gradient_from_activated(
                    negative_input,
                    &negative_raw,
                    &negative_gradient,
                    config.metric_learning_rate,
                    config.metric_weight_decay,
                )?;
                loss_sum += f64::from(loss);
                loss_count += 1;
            }
        }
        let epoch_loss = if loss_count == 0 {
            0.0
        } else {
            (loss_sum / loss_count as f64) as f32
        };
        if epoch == 0 {
            initial_loss = epoch_loss;
        }
        final_loss = epoch_loss;
    }
    Ok((initial_loss, final_loss, total_triplets))
}

fn build_triplets(
    samples: &[RepresentationSample],
    facet: MetricFacet,
    maximum: usize,
) -> Vec<[usize; 3]> {
    if samples.len() < 3 || maximum == 0 {
        return Vec::new();
    }
    let mut triplets = Vec::new();
    for anchor in 0..samples.len() {
        let mut candidates: Vec<(usize, f32)> = (0..samples.len())
            .filter(|candidate| *candidate != anchor)
            .map(|candidate| {
                (
                    candidate,
                    sample_distance(&samples[anchor], &samples[candidate], facet),
                )
            })
            .collect();
        candidates.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let positive = candidates
            .iter()
            .find(|(candidate, _)| {
                samples[*candidate].line_key == samples[anchor].line_key
                    && samples[*candidate].snapshot != samples[anchor].snapshot
            })
            .map(|(candidate, _)| *candidate)
            .unwrap_or(candidates[0].0);
        let positive_distance = candidates
            .iter()
            .find(|(candidate, _)| *candidate == positive)
            .map(|(_, distance)| *distance)
            .unwrap_or(0.0);
        let negative = candidates
            .iter()
            .rev()
            .find(|(candidate, distance)| {
                *candidate != positive && *distance >= positive_distance + 0.001
            })
            .map(|(candidate, _)| *candidate)
            .or_else(|| {
                candidates
                    .iter()
                    .find(|(candidate, _)| *candidate != positive)
                    .map(|(candidate, _)| *candidate)
            });
        if let Some(negative) = negative {
            triplets.push([anchor, positive, negative]);
            if triplets.len() >= maximum {
                break;
            }
        }
    }
    triplets
}

fn sample_distance(
    left: &RepresentationSample,
    right: &RepresentationSample,
    facet: MetricFacet,
) -> f32 {
    match facet {
        MetricFacet::Base => vector_distance(&left.raw.base, &right.raw.base),
        MetricFacet::Role => vector_distance(&left.raw.role, &right.raw.role),
        MetricFacet::Service => vector_distance(&left.raw.service, &right.raw.service),
        MetricFacet::Geometry => vector_distance(&left.raw.geometry, &right.raw.geometry),
        MetricFacet::Resilience => match (left.criticality, right.criticality) {
            (Some(left), Some(right)) => vector_distance(&left, &right),
            _ => vector_distance(&left.raw.resilience, &right.raw.resilience),
        },
        MetricFacet::General => {
            let distances = [
                vector_distance(&left.raw.base, &right.raw.base),
                vector_distance(&left.raw.role, &right.raw.role),
                vector_distance(&left.raw.service, &right.raw.service),
                vector_distance(&left.raw.geometry, &right.raw.geometry),
                sample_distance(left, right, MetricFacet::Resilience),
            ];
            distances.iter().sum::<f32>() / distances.len() as f32
        }
    }
}

fn vector_distance(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 1.0;
    }
    // Pair bootstrapping is a target-generation heuristic, not the model
    // forward pass. A deterministic stride keeps it cheap for the 512-bin
    // timetable channels while retaining the full input for learned heads.
    let stride = (left.len() / 256).max(1);
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for index in (0..left.len()).step_by(stride) {
        let left = left[index];
        let right = right[index];
        let left = bounded(left);
        let right = bounded(right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator <= f32::EPSILON {
        1.0
    } else {
        (1.0 - dot / denominator).clamp(0.0, 2.0)
    }
}

fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / left.len().max(1) as f32
}

fn normalized(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| value / norm).collect()
    }
}

fn normalization_gradient(raw: &[f32], gradient: &[f32]) -> Vec<f32> {
    let norm = raw
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    let normalized_raw: Vec<f32> = raw.iter().map(|value| value / norm).collect();
    let projection = normalized_raw
        .iter()
        .zip(gradient)
        .map(|(value, gradient)| value * gradient)
        .sum::<f32>();
    normalized_raw
        .iter()
        .zip(gradient)
        .map(|(value, gradient)| (gradient - value * projection) / norm / raw.len().max(1) as f32)
        .collect()
}

fn bounded(value: f32) -> f32 {
    if value.is_finite() {
        value / (1.0 + value.abs())
    } else {
        0.0
    }
}

fn facet_input(
    representation: &TrainableLineRepresentationModel,
    sample: &RepresentationSample,
    facet: MetricFacet,
) -> Result<Vec<f32>> {
    let base = representation.base.forward(&sample.raw.base)?;
    match facet {
        MetricFacet::Base => Ok(sample.raw.base.clone()),
        MetricFacet::Service => Ok(sample.raw.service.clone()),
        MetricFacet::Geometry => Ok(sample.raw.geometry.clone()),
        MetricFacet::Role => {
            let mut input = base;
            input.extend(&sample.raw.role);
            Ok(input)
        }
        MetricFacet::Resilience => {
            let mut input = base;
            input.extend(&sample.raw.resilience);
            Ok(input)
        }
        MetricFacet::General => {
            let role = representation.role.forward(&{
                let mut input = base.clone();
                input.extend(&sample.raw.role);
                input
            })?;
            let service = representation.service.forward(&sample.raw.service)?;
            let geometry = representation.geometry.forward(&sample.raw.geometry)?;
            let resilience = representation.resilience.forward(&{
                let mut input = base.clone();
                input.extend(&sample.raw.resilience);
                input
            })?;
            let mut input = base;
            input.extend(role);
            input.extend(service);
            input.extend(geometry);
            input.extend(resilience);
            Ok(input)
        }
    }
}

fn facet_head(
    representation: &TrainableLineRepresentationModel,
    facet: MetricFacet,
) -> &ProjectionHead {
    match facet {
        MetricFacet::Base => &representation.base,
        MetricFacet::General => &representation.general,
        MetricFacet::Role => &representation.role,
        MetricFacet::Service => &representation.service,
        MetricFacet::Geometry => &representation.geometry,
        MetricFacet::Resilience => &representation.resilience,
    }
}

fn facet_head_mut(
    representation: &mut TrainableLineRepresentationModel,
    facet: MetricFacet,
) -> &mut ProjectionHead {
    match facet {
        MetricFacet::Base => &mut representation.base,
        MetricFacet::General => &mut representation.general,
        MetricFacet::Role => &mut representation.role,
        MetricFacet::Service => &mut representation.service,
        MetricFacet::Geometry => &mut representation.geometry,
        MetricFacet::Resilience => &mut representation.resilience,
    }
}

fn fit_criticality_head(
    head: &mut CriticalityHead,
    examples: &[Example],
    config: &CriticalityTrainingConfig,
) -> Result<(f32, f32)> {
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for example in examples {
            let prediction = head.predict_inputs(&example.input)?;
            let mut errors = vec![0.0; head.output_dimension];
            for output in 0..head.output_dimension.min(CRITICALITY_OUTPUTS) {
                let error = prediction[output] - example.target[output];
                errors[output] = huber_gradient(error) * example.weight;
                epoch_loss += f64::from(huber_loss(error) * example.weight);
            }
            weight_sum += f64::from(example.weight);
            update_head(head, &example.input, &errors, config.learning_rate)?;
        }
        epoch_loss /= weight_sum.max(f64::EPSILON);
        if epoch == 0 {
            initial_loss = epoch_loss as f32;
        }
        final_loss = epoch_loss as f32;
        if config.ranking_weight > 0.0 {
            apply_pairwise_ranking(
                head,
                examples,
                config.learning_rate,
                config.ranking_weight,
                config.max_ranking_pairs,
            )?;
        }
    }
    Ok((initial_loss, final_loss))
}

struct Example {
    input: Vec<f32>,
    target: [f32; CRITICALITY_OUTPUTS],
    weight: f32,
    snapshot: String,
}

fn training_examples(
    head: &CriticalityHead,
    embeddings: &Embeddings,
    graph: &GraphTensor,
    labels: &[LineImpactLabel],
) -> Result<Vec<Example>> {
    let snapshot_id = graph.manifest.snapshot_id.as_str();
    let relevant_labels: Vec<&LineImpactLabel> = labels
        .iter()
        .filter(|label| label.snapshot == snapshot_id)
        .collect();
    let mut snapshot_counts = HashMap::<&str, usize>::new();
    for label in &relevant_labels {
        *snapshot_counts.entry(label.snapshot.as_str()).or_default() += 1;
    }
    relevant_labels
        .iter()
        .filter(|label| (label.line.0 as usize) < graph.manifest.line_count)
        .map(|label| {
            let line = label.line.0 as usize;
            let input = head.input_for_line(embeddings, graph, line)?;
            let target = normalize_criticality_targets([
                label.accessibility_auc_loss,
                label.unreachable_share,
                label.mean_delay_reachable_seconds,
                label.p95_delay_reachable_seconds,
                label.mean_extra_transfers,
                label.stations_losing_all_service_share,
            ]);
            let city_weight = 1.0 / snapshot_counts[label.snapshot.as_str()].max(1) as f32;
            Ok(Example {
                input,
                target,
                weight: city_weight,
                snapshot: label.snapshot.clone(),
            })
        })
        .collect()
}

fn huber_loss(error: f32) -> f32 {
    let absolute = error.abs();
    if absolute <= 1.0 {
        0.5 * error * error
    } else {
        absolute - 0.5
    }
}

fn huber_gradient(error: f32) -> f32 {
    error.clamp(-1.0, 1.0)
}

fn update_head(
    head: &mut CriticalityHead,
    input: &[f32],
    gradient: &[f32],
    learning_rate: f32,
) -> Result<()> {
    head.apply_gradient(input, gradient, learning_rate)
}

fn apply_pairwise_ranking(
    head: &mut CriticalityHead,
    examples: &[Example],
    learning_rate: f32,
    ranking_weight: f32,
    maximum_pairs: usize,
) -> Result<()> {
    if maximum_pairs == 0 {
        return Ok(());
    }
    let pair_scale = 1.0 / maximum_pairs as f32;
    let mut pair_count = 0_usize;
    'left: for left in 0..examples.len() {
        for right in (left + 1)..examples.len() {
            if examples[left].snapshot != examples[right].snapshot {
                continue;
            }
            if pair_count >= maximum_pairs {
                break 'left;
            }
            let target_delta = examples[left].target[0] - examples[right].target[0];
            let direction = target_delta.signum();
            if direction == 0.0 {
                continue;
            }
            pair_count += 1;
            let left_prediction = head.predict_inputs(&examples[left].input)?;
            let right_prediction = head.predict_inputs(&examples[right].input)?;
            let margin = direction * (left_prediction[0] - right_prediction[0]);
            let logistic_gradient = if margin >= 0.0 {
                -direction * (-margin).exp() / (1.0 + (-margin).exp())
            } else {
                -direction / (1.0 + margin.exp())
            } * pair_scale;
            update_single_output(
                head,
                &examples[left].input,
                0,
                logistic_gradient * ranking_weight * examples[left].weight,
                learning_rate,
            )?;
            update_single_output(
                head,
                &examples[right].input,
                0,
                -logistic_gradient * ranking_weight * examples[right].weight,
                learning_rate,
            )?;
        }
    }
    Ok(())
}

fn update_single_output(
    head: &mut CriticalityHead,
    input: &[f32],
    output: usize,
    gradient: f32,
    learning_rate: f32,
) -> Result<()> {
    if output >= head.output_dimension {
        anyhow::bail!("criticality output index {output} is out of bounds");
    }
    let mut output_gradient = vec![0.0; head.output_dimension];
    output_gradient[output] = gradient;
    head.apply_gradient(input, &output_gradient, learning_rate)
}

#[cfg(feature = "tch-backend")]
pub mod tch_training {
    use super::*;
    use tch::nn::OptimizerConfig;
    use tch::{Device, Kind, Tensor};
    use transit_model::tch_backend::TchRelationalAutoencoder;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct TchTrainingReport {
        pub backend: String,
        pub steps: usize,
        pub initial_loss: f64,
        pub final_loss: f64,
    }

    pub fn train_tch_autoencoder(
        graph: &GraphTensor,
        config: &PretrainingConfig,
        device: Device,
        checkpoint: Option<&Path>,
    ) -> Result<TchTrainingReport> {
        let model = TchRelationalAutoencoder::new(device, graph, &config.model);
        let mut optimizer = tch::nn::Adam::default()
            .wd(config.weight_decay as f64)
            .build(&model.var_store, config.learning_rate as f64)
            .context("building LibTorch Adam optimizer")?;
        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;
        for step in 0..config.steps {
            let mask =
                MaskSelection::sample(graph, &config.mask, config.seed.wrapping_add(step as u64));
            let reconstruction = model.forward(graph, &mask, true)?;
            let station_target = Tensor::from_slice(&graph.station_features.values)
                .to_device(device)
                .reshape([
                    graph.station_features.rows as i64,
                    graph.station_features.cols as i64,
                ]);
            let line_target = Tensor::from_slice(&graph.line_features.values)
                .to_device(device)
                .reshape([
                    graph.line_features.rows as i64,
                    graph.line_features.cols as i64,
                ]);
            let station_loss = masked_mse(
                &reconstruction.station_features,
                &station_target,
                &mask.station_rows,
                device,
            );
            let line_loss = masked_mse(
                &reconstruction.line_features,
                &line_target,
                &mask.line_rows,
                device,
            );
            let loss = station_loss + line_loss;
            let value = loss.double_value(&[]);
            if step == 0 {
                initial_loss = value;
            }
            final_loss = value;
            optimizer.backward_step(&loss);
        }
        if let Some(path) = checkpoint {
            model.save(path)?;
        }
        Ok(TchTrainingReport {
            backend: "tch-rs-libtorch".into(),
            steps: config.steps,
            initial_loss,
            final_loss,
        })
    }

    fn masked_mse(prediction: &Tensor, target: &Tensor, rows: &[bool], device: Device) -> Tensor {
        let row_mask = Tensor::from_slice(
            &rows
                .iter()
                .map(|masked| if *masked { 1.0_f32 } else { 0.0 })
                .collect::<Vec<_>>(),
        )
        .to_device(device)
        .unsqueeze(1);
        let difference = (prediction - target) * &row_mask;
        (&difference * &difference).sum(Kind::Float) / row_mask.sum(Kind::Float).clamp_min(1.0)
    }
}
