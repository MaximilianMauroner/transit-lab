//! Model contracts shared by the reference CPU path and the optional LibTorch
//! implementation.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use transit_domain::SERVICE_DAY_BINS;
use transit_graph::{FeatureMatrix, GraphTensor, TEMPORAL_CHANNELS};

mod representations;

pub use representations::{
    EmbeddingFacet, LineEmbedding, LineRepresentationSet, ProjectionHead, RawLineFeatureSet,
    RawLineFeatures, ReferenceLineRepresentationEncoder, RepresentationConfig,
    TrainableLineRepresentationModel, BASE_LINE_EMBEDDING_DIM, CITY_EMBEDDING_DIM,
    GENERAL_EMBEDDING_DIM, GEOMETRY_EMBEDDING_DIM, RESILIENCE_EMBEDDING_DIM, ROLE_EMBEDDING_DIM,
    SERVICE_EMBEDDING_DIM,
};

pub const CRITICALITY_OUTPUTS: usize = 6;
pub const CRITICALITY_TARGET_SCALES: [f32; CRITICALITY_OUTPUTS] =
    [1.0, 1.0, 3_600.0, 3_600.0, 1.0, 1.0];

pub fn normalize_criticality_targets(
    targets: [f32; CRITICALITY_OUTPUTS],
) -> [f32; CRITICALITY_OUTPUTS] {
    std::array::from_fn(|index| targets[index] / CRITICALITY_TARGET_SCALES[index])
}

pub fn denormalize_criticality_targets(targets: Vec<f32>) -> Vec<f32> {
    targets
        .into_iter()
        .enumerate()
        .map(|(index, value)| value * CRITICALITY_TARGET_SCALES.get(index).copied().unwrap_or(1.0))
        .collect()
}

fn bounded_criticality_input(value: f32) -> f32 {
    if value.is_finite() {
        value / (1.0 + value.abs())
    } else {
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_dimension: usize,
    pub temporal_dimension: usize,
    pub graph_layers: usize,
    pub dropout: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dimension: 128,
            temporal_dimension: 32,
            graph_layers: 4,
            dropout: 0.15,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskConfig {
    pub station_feature_probability: f32,
    pub line_feature_probability: f32,
    pub temporal_block_probability: f32,
    pub served_edge_probability: f32,
    pub transfer_edge_probability: f32,
    pub temporal_block_bins: usize,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            station_feature_probability: 0.30,
            line_feature_probability: 0.40,
            temporal_block_probability: 0.25,
            served_edge_probability: 0.10,
            transfer_edge_probability: 0.05,
            temporal_block_bins: 8,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskSelection {
    pub station_rows: Vec<bool>,
    pub line_rows: Vec<bool>,
    pub station_temporal_blocks: Vec<bool>,
    pub line_temporal_blocks: Vec<bool>,
    pub served_edges: Vec<bool>,
    pub transfer_edges: Vec<bool>,
}

impl MaskSelection {
    pub fn all_unmasked(graph: &GraphTensor) -> Self {
        Self {
            station_rows: vec![false; graph.manifest.station_count],
            line_rows: vec![false; graph.manifest.line_count],
            station_temporal_blocks: vec![false; graph.manifest.station_count * SERVICE_DAY_BINS],
            line_temporal_blocks: vec![false; graph.manifest.line_count * SERVICE_DAY_BINS],
            served_edges: vec![false; graph.serves_src.len()],
            transfer_edges: vec![false; graph.transfer_src.len()],
        }
    }

    pub fn sample(graph: &GraphTensor, config: &MaskConfig, seed: u64) -> Self {
        let mut random = SplitMix64::new(seed);
        let mut selection = Self::all_unmasked(graph);
        for value in &mut selection.station_rows {
            *value = random.next_f32() < config.station_feature_probability;
        }
        for value in &mut selection.line_rows {
            *value = random.next_f32() < config.line_feature_probability;
        }
        for row in 0..graph.manifest.station_count {
            for block in 0..SERVICE_DAY_BINS {
                if random.next_f32() < config.temporal_block_probability / SERVICE_DAY_BINS as f32 {
                    let end = (block + config.temporal_block_bins).min(SERVICE_DAY_BINS);
                    for current in block..end {
                        selection.station_temporal_blocks[row * SERVICE_DAY_BINS + current] = true;
                    }
                }
            }
        }
        for row in 0..graph.manifest.line_count {
            for block in 0..SERVICE_DAY_BINS {
                if random.next_f32() < config.temporal_block_probability / SERVICE_DAY_BINS as f32 {
                    let end = (block + config.temporal_block_bins).min(SERVICE_DAY_BINS);
                    for current in block..end {
                        selection.line_temporal_blocks[row * SERVICE_DAY_BINS + current] = true;
                    }
                }
            }
        }
        for value in &mut selection.served_edges {
            *value = random.next_f32() < config.served_edge_probability;
        }
        for value in &mut selection.transfer_edges {
            *value = random.next_f32() < config.transfer_edge_probability;
        }
        selection
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embeddings {
    pub station: Vec<Vec<f32>>,
    pub line: Vec<Vec<f32>>,
    pub city: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reconstruction {
    pub station_features: FeatureMatrix,
    pub line_features: FeatureMatrix,
    pub served_by_logits: Vec<f32>,
    pub transfer_logits: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceRelationalAutoencoder {
    pub config: ModelConfig,
    #[serde(default)]
    station_decoder_weights: Option<Vec<f32>>,
    #[serde(default)]
    station_decoder_bias: Option<Vec<f32>>,
    #[serde(default)]
    line_decoder_weights: Option<Vec<f32>>,
    #[serde(default)]
    line_decoder_bias: Option<Vec<f32>>,
}

impl ReferenceRelationalAutoencoder {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            station_decoder_weights: None,
            station_decoder_bias: None,
            line_decoder_weights: None,
            line_decoder_bias: None,
        }
    }

    pub fn initialize_for_graph(&mut self, graph: &GraphTensor) {
        let hidden = self.config.hidden_dimension;
        if self.station_decoder_weights.is_none() {
            self.station_decoder_weights =
                Some(decoder_weights(hidden, graph.station_features.cols, 401));
            self.station_decoder_bias = Some(vec![0.0; graph.station_features.cols]);
        }
        if self.line_decoder_weights.is_none() {
            self.line_decoder_weights =
                Some(decoder_weights(hidden, graph.line_features.cols, 431));
            self.line_decoder_bias = Some(vec![0.0; graph.line_features.cols]);
        }
    }

    pub fn encode(&self, graph: &GraphTensor, mask: &MaskSelection) -> Result<Embeddings> {
        graph.validate()?;
        validate_mask(graph, mask)?;
        self.encode_inner(graph, mask)
    }

    fn encode_inner(&self, graph: &GraphTensor, mask: &MaskSelection) -> Result<Embeddings> {
        let hidden = self.config.hidden_dimension;
        let mut station = graph
            .station_features
            .values
            .chunks(graph.station_features.cols)
            .enumerate()
            .map(|(row, values)| {
                let mut features = values.to_vec();
                if mask.station_rows[row] {
                    features.fill(0.0);
                }
                append_temporal_summary(
                    &mut features,
                    graph.station_temporal.row(row),
                    &mask.station_temporal_blocks
                        [row * SERVICE_DAY_BINS..(row + 1) * SERVICE_DAY_BINS],
                );
                project_features(&features, hidden, 17)
            })
            .collect::<Vec<_>>();
        let mut line = graph
            .line_features
            .values
            .chunks(graph.line_features.cols)
            .enumerate()
            .map(|(row, values)| {
                let mut features = values.to_vec();
                if mask.line_rows[row] {
                    features.fill(0.0);
                }
                append_temporal_summary(
                    &mut features,
                    graph.line_temporal.row(row),
                    &mask.line_temporal_blocks
                        [row * SERVICE_DAY_BINS..(row + 1) * SERVICE_DAY_BINS],
                );
                project_features(&features, hidden, 31)
            })
            .collect::<Vec<_>>();

        // The station-line graph is permutation equivariant, but it does not
        // retain the order in which a line visits its stations. Fold the
        // persisted canonical patterns through a small recurrent update before
        // relation message passing so branches and stop restrictions remain
        // visible to the shared line state.
        let pattern_context = encode_pattern_sequences(graph, &station, &line, hidden);
        for (line_value, pattern_value) in line.iter_mut().zip(pattern_context) {
            for (value, context) in line_value.iter_mut().zip(pattern_value) {
                *value = (*value + context * 0.22).tanh();
            }
        }

        for layer in 0..self.config.graph_layers {
            let station_to_line = aggregate(
                &station,
                &graph.serves_src,
                &graph.serves_dst,
                line.len(),
                Some(&mask.served_edges),
            );
            let line_to_station = aggregate(
                &line,
                &graph.serves_dst,
                &graph.serves_src,
                station.len(),
                Some(&mask.served_edges),
            );
            let transfer = aggregate(
                &station,
                &graph.transfer_src,
                &graph.transfer_dst,
                station.len(),
                Some(&mask.transfer_edges),
            );
            let transit = aggregate_transit(&station, &line, graph);
            let interchange = aggregate(
                &line,
                &graph.interchange_src,
                &graph.interchange_dst,
                line.len(),
                None,
            );
            station = station
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    combine_vectors(
                        value,
                        [
                            (&line_to_station[row], 0.14),
                            (&transfer[row], 0.12),
                            (&transit[row], 0.16),
                        ],
                        layer as f32,
                    )
                })
                .collect();
            line = line
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    combine_vectors(
                        value,
                        [(&station_to_line[row], 0.20), (&interchange[row], 0.14)],
                        layer as f32 + 0.5,
                    )
                })
                .collect();
        }
        let city = city_pool(&station, &line, hidden);
        Ok(Embeddings {
            station,
            line,
            city,
        })
    }

    pub fn reconstruct(
        &self,
        graph: &GraphTensor,
        mask: &MaskSelection,
    ) -> Result<(Embeddings, Reconstruction)> {
        graph.validate()?;
        validate_mask(graph, mask)?;
        self.reconstruct_inner(graph, mask)
    }

    fn reconstruct_inner(
        &self,
        graph: &GraphTensor,
        mask: &MaskSelection,
    ) -> Result<(Embeddings, Reconstruction)> {
        let embeddings = self.encode_inner(graph, mask)?;
        let station_values = embeddings
            .station
            .iter()
            .map(|value| {
                decode_with_optional_weights(
                    value,
                    graph.station_features.cols,
                    self.station_decoder_weights.as_deref(),
                    self.station_decoder_bias.as_deref(),
                    101,
                )
            })
            .collect::<Vec<_>>();
        let line_values = embeddings
            .line
            .iter()
            .map(|value| {
                decode_with_optional_weights(
                    value,
                    graph.line_features.cols,
                    self.line_decoder_weights.as_deref(),
                    self.line_decoder_bias.as_deref(),
                    151,
                )
            })
            .collect::<Vec<_>>();
        let served_by_logits = graph
            .serves_src
            .iter()
            .zip(&graph.serves_dst)
            .map(|(station, line)| {
                dot(
                    &embeddings.station[*station as usize],
                    &embeddings.line[*line as usize],
                )
            })
            .collect();
        let transfer_logits = graph
            .transfer_src
            .iter()
            .zip(&graph.transfer_dst)
            .map(|(from, to)| {
                dot(
                    &embeddings.station[*from as usize],
                    &embeddings.station[*to as usize],
                )
            })
            .collect();
        Ok((
            embeddings,
            Reconstruction {
                station_features: FeatureMatrix::from_rows(station_values)?,
                line_features: FeatureMatrix::from_rows(line_values)?,
                served_by_logits,
                transfer_logits,
            },
        ))
    }

    /// Train the reference backend's decoder with a small deterministic SGD
    /// step. The encoder remains fixed; GPU/LibTorch training is provided by
    /// the optional `tch-backend` feature.
    pub fn train_decoder_step(
        &mut self,
        graph: &GraphTensor,
        mask: &MaskSelection,
        learning_rate: f32,
    ) -> Result<f32> {
        graph.validate()?;
        validate_mask(graph, mask)?;
        self.initialize_for_graph(graph);
        self.validate_decoder_dimensions(graph)?;
        let embeddings = self.encode_inner(graph, mask)?;
        let mut loss = 0.0_f64;
        let station_weights = self
            .station_decoder_weights
            .as_mut()
            .expect("decoder initialized");
        let station_bias = self
            .station_decoder_bias
            .as_mut()
            .expect("decoder initialized");
        for (row, embedding) in embeddings.station.iter().enumerate() {
            if !mask.station_rows[row] {
                continue;
            }
            for (output, bias) in station_bias
                .iter_mut()
                .enumerate()
                .take(graph.station_features.cols)
            {
                let offset = output * self.config.hidden_dimension;
                let prediction = *bias
                    + station_weights[offset..offset + self.config.hidden_dimension]
                        .iter()
                        .zip(embedding)
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>();
                let error = prediction - graph.station_features.row(row)[output];
                loss += f64::from(error * error);
                *bias -= learning_rate * error;
                for (weight, value) in station_weights
                    [offset..offset + self.config.hidden_dimension]
                    .iter_mut()
                    .zip(embedding)
                {
                    *weight -= learning_rate * error * value;
                }
            }
        }
        let line_weights = self
            .line_decoder_weights
            .as_mut()
            .expect("decoder initialized");
        let line_bias = self
            .line_decoder_bias
            .as_mut()
            .expect("decoder initialized");
        for (row, embedding) in embeddings.line.iter().enumerate() {
            if !mask.line_rows[row] {
                continue;
            }
            for (output, bias) in line_bias
                .iter_mut()
                .enumerate()
                .take(graph.line_features.cols)
            {
                let offset = output * self.config.hidden_dimension;
                let prediction = *bias
                    + line_weights[offset..offset + self.config.hidden_dimension]
                        .iter()
                        .zip(embedding)
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>();
                let error = prediction - graph.line_features.row(row)[output];
                loss += f64::from(error * error);
                *bias -= learning_rate * error;
                for (weight, value) in line_weights[offset..offset + self.config.hidden_dimension]
                    .iter_mut()
                    .zip(embedding)
                {
                    *weight -= learning_rate * error * value;
                }
            }
        }
        let count = mask.station_rows.iter().filter(|value| **value).count()
            * graph.station_features.cols
            + mask.line_rows.iter().filter(|value| **value).count() * graph.line_features.cols;
        Ok((loss / count.max(1) as f64) as f32)
    }

    fn validate_decoder_dimensions(&self, graph: &GraphTensor) -> Result<()> {
        let hidden = self.config.hidden_dimension;
        let station_width = hidden * graph.station_features.cols;
        let line_width = hidden * graph.line_features.cols;
        if self.station_decoder_weights.as_ref().map(Vec::len) != Some(station_width)
            || self.station_decoder_bias.as_ref().map(Vec::len) != Some(graph.station_features.cols)
            || self.line_decoder_weights.as_ref().map(Vec::len) != Some(line_width)
            || self.line_decoder_bias.as_ref().map(Vec::len) != Some(graph.line_features.cols)
        {
            bail!("decoder dimensions do not match the graph feature schema");
        }
        Ok(())
    }

    pub fn structural_uniqueness_scores(
        &self,
        graph: &GraphTensor,
        mask: &MaskSelection,
    ) -> Result<Vec<f32>> {
        // Validate the caller's shape even though the score is computed from a
        // clean view. Otherwise a previously sampled corruption mask would
        // make the score measure unrelated missing attributes as well as the
        // line being scored. A single clean reconstruction is intentionally
        // used here: the residual is a feed-anomaly signal, and recomputing a
        // full graph encoder once per line made inference O(lines * graph).
        graph.validate()?;
        validate_mask(graph, mask)?;
        let clean_mask = MaskSelection::all_unmasked(graph);
        let (_, reconstruction) = self.reconstruct(graph, &clean_mask)?;
        Ok((0..graph.manifest.line_count)
            .map(|line| {
                let actual = graph.line_features.row(line);
                let predicted = reconstruction.line_features.row(line);
                actual
                    .iter()
                    .zip(predicted)
                    .map(|(left, right)| (left - right).abs())
                    .sum::<f32>()
                    / actual.len().max(1) as f32
            })
            .collect())
    }
}

fn validate_mask(graph: &GraphTensor, mask: &MaskSelection) -> Result<()> {
    if mask.station_rows.len() != graph.manifest.station_count
        || mask.line_rows.len() != graph.manifest.line_count
        || mask.station_temporal_blocks.len() != graph.manifest.station_count * SERVICE_DAY_BINS
        || mask.line_temporal_blocks.len() != graph.manifest.line_count * SERVICE_DAY_BINS
        || mask.served_edges.len() != graph.serves_src.len()
        || mask.transfer_edges.len() != graph.transfer_src.len()
    {
        bail!("mask shape does not match graph shape");
    }
    Ok(())
}

fn append_temporal_summary(features: &mut Vec<f32>, temporal: &[f32], blocks: &[bool]) {
    for channel in 0..TEMPORAL_CHANNELS {
        let start = channel * SERVICE_DAY_BINS;
        let end = start + SERVICE_DAY_BINS;
        let values = &temporal[start..end];
        let visible: Vec<f32> = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!blocks[index]).then_some(*value))
            .collect();
        let mean = if visible.is_empty() {
            0.0
        } else {
            visible.iter().sum::<f32>() / visible.len() as f32
        };
        let max = visible.into_iter().fold(0.0_f32, f32::max);
        features.push(mean);
        features.push(max);
    }
}

fn project_features(features: &[f32], hidden: usize, seed: u32) -> Vec<f32> {
    (0..hidden)
        .map(|hidden_index| {
            let sum = features
                .iter()
                .enumerate()
                .map(|(feature_index, value)| {
                    let angle = (feature_index as f32 + seed as f32 + 1.0)
                        * (hidden_index as f32 + 1.0)
                        * 0.017;
                    value * angle.sin() * 0.05
                })
                .sum::<f32>();
            sum.tanh()
        })
        .collect()
}

fn encode_pattern_sequences(
    graph: &GraphTensor,
    station: &[Vec<f32>],
    line: &[Vec<f32>],
    hidden: usize,
) -> Vec<Vec<f32>> {
    let mut output = vec![vec![0.0; hidden]; graph.manifest.line_count];
    let mut weights = vec![0_u32; graph.manifest.line_count];
    for pattern in 0..graph.manifest.pattern_count {
        let Some(&start) = graph.pattern_offsets.get(pattern) else {
            continue;
        };
        let Some(&end) = graph.pattern_offsets.get(pattern + 1) else {
            continue;
        };
        let start = start as usize;
        let end = end as usize;
        if start >= end || end > graph.pattern_stops.len() {
            continue;
        }
        let Some(&line_index) = graph.pattern_lines.get(pattern) else {
            continue;
        };
        let line_index = line_index as usize;
        if line_index >= output.len() || line_index >= line.len() {
            continue;
        }
        let mut state = vec![0.0_f32; hidden];
        for position in start..end {
            let station_index = graph.pattern_stops[position] as usize;
            let station_value = station.get(station_index);
            let stop_features = graph.pattern_stop_features.row(position);
            let segment_index = position.saturating_sub(start);
            let segment_value = if segment_index < end - start - 1 {
                graph.pattern_segment_features.row(pattern_segment_row(
                    graph,
                    pattern,
                    segment_index,
                ))
            } else {
                &[]
            };
            for hidden_index in 0..hidden {
                let station_component = station_value
                    .and_then(|values| values.get(hidden_index))
                    .copied()
                    .unwrap_or(0.0);
                let segment_component = segment_value
                    .get(hidden_index % segment_value.len().max(1))
                    .copied()
                    .unwrap_or(0.0);
                let stop_component = stop_features
                    .iter()
                    .enumerate()
                    .map(|(feature, value)| {
                        value * ((hidden_index + 1) as f32 * (feature + 1) as f32 * 0.37).sin()
                    })
                    .sum::<f32>();
                let line_component = line[line_index].get(hidden_index).copied().unwrap_or(0.0);
                let position_component =
                    (position - start) as f32 / (end - start - 1).max(1) as f32;
                state[hidden_index] = (state[hidden_index] * 0.58
                    + station_component * 0.34
                    + segment_component * 0.08
                    + stop_component * 0.03
                    + line_component * 0.05
                    + position_component * 0.002)
                    .tanh();
            }
        }
        let pattern_weight = graph
            .pattern_trip_counts
            .get(pattern)
            .copied()
            .unwrap_or(1)
            .max(1);
        for (target, value) in output[line_index].iter_mut().zip(state) {
            *target += value * pattern_weight as f32;
        }
        weights[line_index] = weights[line_index].saturating_add(pattern_weight);
    }
    for (value, weight) in output.iter_mut().zip(weights) {
        if weight > 0 {
            for component in value {
                *component /= weight as f32;
            }
        }
    }
    output
}

fn pattern_segment_row(graph: &GraphTensor, pattern: usize, segment: usize) -> usize {
    graph
        .pattern_offsets
        .iter()
        .take(pattern)
        .zip(graph.pattern_offsets.iter().skip(1))
        .map(|(left, right)| (*right as usize - *left as usize).saturating_sub(1))
        .sum::<usize>()
        + segment
}

fn aggregate(
    source: &[Vec<f32>],
    src: &[u32],
    dst: &[u32],
    destination_count: usize,
    edge_mask: Option<&[bool]>,
) -> Vec<Vec<f32>> {
    let hidden = source.first().map(Vec::len).unwrap_or(0);
    let mut sums = vec![vec![0.0; hidden]; destination_count];
    let mut counts = vec![0_u32; destination_count];
    for (edge, (&from, &to)) in src.iter().zip(dst).enumerate() {
        if edge_mask
            .and_then(|mask| mask.get(edge))
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        if let (Some(value), Some(sum)) = (source.get(from as usize), sums.get_mut(to as usize)) {
            for (target, input) in sum.iter_mut().zip(value) {
                *target += input;
            }
            counts[to as usize] += 1;
        }
    }
    for (sum, count) in sums.iter_mut().zip(counts) {
        if count > 0 {
            for value in sum {
                *value /= count as f32;
            }
        }
    }
    sums
}

fn aggregate_transit(source: &[Vec<f32>], line: &[Vec<f32>], graph: &GraphTensor) -> Vec<Vec<f32>> {
    let hidden = source.first().map(Vec::len).unwrap_or(0);
    let mut sums = vec![vec![0.0; hidden]; graph.manifest.station_count];
    let mut counts = vec![0_u32; graph.manifest.station_count];
    for edge in 0..graph.transit_src.len() {
        let from = graph.transit_src[edge] as usize;
        let to = graph.transit_dst[edge] as usize;
        let line_index = graph.transit_line[edge] as usize;
        if from >= source.len() || to >= sums.len() || line_index >= line.len() {
            continue;
        }
        let edge_row = graph.transit_features.row(edge);
        for index in 0..hidden {
            let edge_value = edge_row[index % edge_row.len().max(1)] * 0.001;
            sums[to][index] +=
                source[from][index] * 0.5 + line[line_index][index] * 0.25 + edge_value;
        }
        counts[to] += 1;
    }
    for (sum, count) in sums.iter_mut().zip(counts) {
        if count > 0 {
            for value in sum {
                *value /= count as f32;
            }
        }
    }
    sums
}

fn combine_vectors<const N: usize>(
    self_value: &[f32],
    relations: [(&Vec<f32>, f32); N],
    layer: f32,
) -> Vec<f32> {
    (0..self_value.len())
        .map(|index| {
            let mut value = self_value[index] * 0.58;
            for (relation, weight) in relations {
                value += relation[index] * weight;
            }
            (value + (index as f32 * 0.0001 + layer * 0.01).sin() * 0.001).tanh()
        })
        .collect()
}

fn city_pool(station: &[Vec<f32>], line: &[Vec<f32>], hidden: usize) -> Vec<f32> {
    let mut output = vec![0.0; hidden];
    for (index, value) in output.iter_mut().enumerate().take(hidden) {
        let station_mean = mean_column(station, index);
        let station_max = max_column(station, index);
        let line_mean = mean_column(line, index);
        let line_max = max_column(line, index);
        *value = (station_mean + station_max + line_mean + line_max) / 4.0;
    }
    output
}

fn mean_column(values: &[Vec<f32>], column: usize) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|row| row[column]).sum::<f32>() / values.len() as f32
}

fn max_column(values: &[Vec<f32>], column: usize) -> f32 {
    values.iter().map(|row| row[column]).fold(0.0_f32, f32::max)
}

fn decode_features(embedding: &[f32], output_width: usize, seed: u32) -> Vec<f32> {
    (0..output_width)
        .map(|output| {
            embedding
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    value
                        * ((index as f32 + 1.0) * (output as f32 + seed as f32 + 1.0) * 0.013).cos()
                        * 0.1
                })
                .sum::<f32>()
        })
        .collect()
}

fn decode_with_optional_weights(
    embedding: &[f32],
    output_width: usize,
    weights: Option<&[f32]>,
    bias: Option<&[f32]>,
    seed: u32,
) -> Vec<f32> {
    match (weights, bias) {
        (Some(weights), Some(bias))
            if weights.len() == embedding.len() * output_width && bias.len() == output_width =>
        {
            (0..output_width)
                .map(|output| {
                    let offset = output * embedding.len();
                    bias[output]
                        + weights[offset..offset + embedding.len()]
                            .iter()
                            .zip(embedding)
                            .map(|(weight, value)| weight * value)
                            .sum::<f32>()
                })
                .collect()
        }
        _ => decode_features(embedding, output_width, seed),
    }
}

fn decoder_weights(hidden: usize, output_width: usize, seed: u64) -> Vec<f32> {
    let mut random = SplitMix64::new(seed);
    (0..hidden * output_width)
        .map(|_| (random.next_f32() - 0.5) * 0.02)
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriticalityHead {
    pub input_dimension: usize,
    pub output_dimension: usize,
    /// Direct residual input-to-output weights. They keep the small reference
    /// head easy to optimize while the hidden path supplies the requested MLP
    /// capacity.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    #[serde(default)]
    pub hidden_dimension: usize,
    #[serde(default)]
    pub hidden_weights: Vec<f32>,
    #[serde(default)]
    pub hidden_bias: Vec<f32>,
    #[serde(default)]
    pub hidden_output_weights: Vec<f32>,
}

impl CriticalityHead {
    pub fn new(input_dimension: usize, output_dimension: usize, seed: u64) -> Self {
        let mut random = SplitMix64::new(seed);
        let weights = (0..input_dimension * output_dimension)
            .map(|_| (random.next_f32() - 0.5) * 0.02)
            .collect();
        let hidden_dimension = input_dimension.max(16).min(64);
        let hidden_scale = 0.08 / (input_dimension.max(1) as f32).sqrt();
        let hidden_weights = (0..input_dimension * hidden_dimension)
            .map(|_| (random.next_f32() - 0.5) * hidden_scale)
            .collect();
        let hidden_output_scale = 0.08 / (hidden_dimension as f32).sqrt();
        let hidden_output_weights = (0..hidden_dimension * output_dimension)
            .map(|_| (random.next_f32() - 0.5) * hidden_output_scale)
            .collect();
        Self {
            input_dimension,
            output_dimension,
            weights,
            bias: vec![0.0; output_dimension],
            hidden_dimension,
            hidden_weights,
            hidden_bias: vec![0.0; hidden_dimension],
            hidden_output_weights,
        }
    }

    pub fn input_for_line(
        &self,
        embeddings: &Embeddings,
        graph: &GraphTensor,
        line: usize,
    ) -> Result<Vec<f32>> {
        if line >= embeddings.line.len() {
            bail!("line index out of bounds: {line}");
        }
        let mut input =
            Vec::with_capacity(embeddings.line[line].len() * 2 + graph.line_features.cols);
        input.extend(&embeddings.line[line]);
        input.extend(&embeddings.city);
        input.extend(graph.line_features.row(line));
        if input.len() != self.input_dimension {
            bail!(
                "criticality input width {} does not match {}",
                input.len(),
                self.input_dimension
            );
        }
        Ok(input)
    }

    /// Build the criticality input from the reusable line representation. This
    /// is the production path for the multi-task model: it keeps the shared
    /// base state, city context, and directly measured line features together
    /// while leaving the similarity projections task-specific.
    pub fn input_for_representation(
        &self,
        embedding: &LineEmbedding,
        city: &[f32],
        graph: &GraphTensor,
        line: usize,
    ) -> Result<Vec<f32>> {
        if line >= graph.manifest.line_count {
            bail!("line index out of bounds: {line}");
        }
        let mut input =
            Vec::with_capacity(embedding.base.len() + city.len() + graph.line_features.cols);
        input.extend(&embedding.base);
        input.extend(city);
        input.extend(graph.line_features.row(line));
        if input.len() != self.input_dimension {
            bail!(
                "representation criticality input width {} does not match {}",
                input.len(),
                self.input_dimension
            );
        }
        Ok(input)
    }

    pub fn predict_inputs(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        self.validate_inputs(inputs)?;
        if self.is_legacy_linear() {
            return Ok(self.linear_outputs(inputs, false));
        }
        self.validate_mlp_parameters()?;
        let hidden = self.hidden_outputs(inputs);
        Ok(self.mlp_outputs(inputs, &hidden))
    }

    /// Apply a gradient with respect to the six raw criticality outputs. The
    /// residual linear path is intentional: it gives the dependency-free
    /// reference backend a stable optimization path while the hidden path
    /// captures interactions between network, city, and measured line inputs.
    pub fn apply_gradient(
        &mut self,
        inputs: &[f32],
        output_gradient: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        self.validate_inputs(inputs)?;
        if output_gradient.len() != self.output_dimension {
            bail!("criticality gradient width does not match the head");
        }
        if self.is_legacy_linear() {
            self.apply_linear_gradient(inputs, output_gradient, learning_rate);
            return Ok(());
        }
        self.validate_mlp_parameters()?;
        let hidden = self.hidden_outputs(inputs);
        let mut hidden_gradient = vec![0.0; self.hidden_dimension];
        for (output, gradient) in output_gradient.iter().enumerate() {
            let output_offset = output * self.hidden_dimension;
            for (hidden_index, hidden_gradient) in hidden_gradient.iter_mut().enumerate() {
                *hidden_gradient +=
                    *gradient * self.hidden_output_weights[output_offset + hidden_index];
            }
        }

        for (output, gradient) in output_gradient.iter().enumerate() {
            let direct_offset = output * self.input_dimension;
            for (weight, input) in self.weights[direct_offset..direct_offset + self.input_dimension]
                .iter_mut()
                .zip(inputs)
            {
                *weight -= learning_rate * *gradient * bounded_criticality_input(*input);
            }
            let hidden_offset = output * self.hidden_dimension;
            for (weight, hidden_value) in self.hidden_output_weights
                [hidden_offset..hidden_offset + self.hidden_dimension]
                .iter_mut()
                .zip(&hidden)
            {
                *weight -= learning_rate * *gradient * hidden_value;
            }
            self.bias[output] -= learning_rate * *gradient;
        }

        for (hidden_index, gradient) in hidden_gradient.iter().enumerate() {
            let activated = hidden[hidden_index];
            let delta = *gradient * (1.0 - activated * activated);
            let offset = hidden_index * self.input_dimension;
            for (weight, input) in self.hidden_weights[offset..offset + self.input_dimension]
                .iter_mut()
                .zip(inputs)
            {
                *weight -= learning_rate * delta * bounded_criticality_input(*input);
            }
            self.hidden_bias[hidden_index] -= learning_rate * delta;
        }
        Ok(())
    }

    fn validate_inputs(&self, inputs: &[f32]) -> Result<()> {
        if inputs.len() != self.input_dimension {
            bail!(
                "criticality input width {} does not match {}",
                inputs.len(),
                self.input_dimension
            );
        }
        if self.weights.len() != self.input_dimension * self.output_dimension
            || self.bias.len() != self.output_dimension
        {
            bail!("criticality head parameters do not match their declared shape");
        }
        Ok(())
    }

    fn is_legacy_linear(&self) -> bool {
        self.hidden_dimension == 0
            && self.hidden_weights.is_empty()
            && self.hidden_bias.is_empty()
            && self.hidden_output_weights.is_empty()
    }

    fn validate_mlp_parameters(&self) -> Result<()> {
        if self.hidden_dimension == 0
            || self.hidden_weights.len() != self.input_dimension * self.hidden_dimension
            || self.hidden_bias.len() != self.hidden_dimension
            || self.hidden_output_weights.len() != self.hidden_dimension * self.output_dimension
        {
            bail!("criticality MLP parameters do not match their declared shape");
        }
        Ok(())
    }

    fn linear_outputs(&self, inputs: &[f32], bounded_inputs: bool) -> Vec<f32> {
        (0..self.output_dimension)
            .map(|output| {
                let start = output * self.input_dimension;
                self.bias[output]
                    + self.weights[start..start + self.input_dimension]
                        .iter()
                        .zip(inputs)
                        .map(|(weight, input)| {
                            weight
                                * if bounded_inputs {
                                    bounded_criticality_input(*input)
                                } else {
                                    *input
                                }
                        })
                        .sum::<f32>()
            })
            .collect()
    }

    fn hidden_outputs(&self, inputs: &[f32]) -> Vec<f32> {
        (0..self.hidden_dimension)
            .map(|hidden| {
                let start = hidden * self.input_dimension;
                (self.hidden_bias[hidden]
                    + self.hidden_weights[start..start + self.input_dimension]
                        .iter()
                        .zip(inputs)
                        .map(|(weight, input)| weight * bounded_criticality_input(*input))
                        .sum::<f32>())
                .tanh()
            })
            .collect()
    }

    fn mlp_outputs(&self, inputs: &[f32], hidden: &[f32]) -> Vec<f32> {
        (0..self.output_dimension)
            .map(|output| {
                let direct_start = output * self.input_dimension;
                let hidden_start = output * self.hidden_dimension;
                self.bias[output]
                    + self.weights[direct_start..direct_start + self.input_dimension]
                        .iter()
                        .zip(inputs)
                        .map(|(weight, input)| weight * bounded_criticality_input(*input))
                        .sum::<f32>()
                    + self.hidden_output_weights[hidden_start..hidden_start + self.hidden_dimension]
                        .iter()
                        .zip(hidden)
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>()
            })
            .collect()
    }

    fn apply_linear_gradient(
        &mut self,
        inputs: &[f32],
        output_gradient: &[f32],
        learning_rate: f32,
    ) {
        for (output, gradient) in output_gradient.iter().enumerate() {
            let start = output * self.input_dimension;
            for (weight, input) in self.weights[start..start + self.input_dimension]
                .iter_mut()
                .zip(inputs)
            {
                *weight -= learning_rate * *gradient * *input;
            }
            self.bias[output] -= learning_rate * *gradient;
        }
    }

    pub fn predict_all(
        &self,
        embeddings: &Embeddings,
        graph: &GraphTensor,
    ) -> Result<Vec<Vec<f32>>> {
        (0..graph.manifest.line_count)
            .map(|line| {
                self.input_for_line(embeddings, graph, line)
                    .and_then(|input| self.predict_inputs(&input))
            })
            .collect()
    }
}

#[cfg(feature = "tch-backend")]
pub mod tch_backend;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use gtfs_compile::{compile, CompileOptions};
    use gtfs_ingest::GtfsFeed;
    use transit_labels::LineImpactLabel;

    fn graph() -> GraphTensor {
        let feed = GtfsFeed::from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/synthetic-feeds/basic"
        ))
        .unwrap();
        let network = compile(
            &feed,
            &CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap()),
        )
        .unwrap();
        GraphTensor::from_network(&network).unwrap()
    }

    #[test]
    fn reference_encoder_preserves_typed_embedding_shapes() {
        let graph = graph();
        let model = ReferenceRelationalAutoencoder::new(ModelConfig {
            hidden_dimension: 8,
            graph_layers: 2,
            ..ModelConfig::default()
        });
        let mask = MaskSelection::sample(&graph, &MaskConfig::default(), 17);
        let (embeddings, reconstruction) = model.reconstruct(&graph, &mask).unwrap();
        assert_eq!(embeddings.station.len(), graph.manifest.station_count);
        assert_eq!(embeddings.station[0].len(), 8);
        assert_eq!(embeddings.line.len(), graph.manifest.line_count);
        assert_eq!(embeddings.city.len(), 8);
        assert_eq!(reconstruction.line_features.rows, graph.manifest.line_count);
        let representations = ReferenceLineRepresentationEncoder::default()
            .encode(&graph, &embeddings)
            .unwrap();
        assert_eq!(representations.lines.len(), graph.manifest.line_count);
        assert_eq!(representations.lines[0].base.len(), BASE_LINE_EMBEDDING_DIM);
        assert_eq!(representations.lines[0].role.len(), ROLE_EMBEDDING_DIM);
        assert!(representations.lines.iter().all(|line| line
            .base
            .iter()
            .chain(&line.general)
            .chain(&line.role)
            .chain(&line.service)
            .chain(&line.geometry)
            .chain(&line.resilience)
            .all(|value| value.is_finite())));
        assert!(model
            .structural_uniqueness_scores(&graph, &mask)
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn reference_decoder_step_is_finite() {
        let graph = graph();
        let mut model = ReferenceRelationalAutoencoder::new(ModelConfig {
            hidden_dimension: 8,
            graph_layers: 1,
            ..ModelConfig::default()
        });
        let mask = MaskSelection::sample(&graph, &MaskConfig::default(), 23);
        let loss = model.train_decoder_step(&graph, &mask, 0.00001).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn criticality_head_has_trainable_mlp_and_keeps_legacy_linear_compatibility() {
        let mut head = CriticalityHead::new(4, 2, 41);
        assert!(head.hidden_dimension > 0);
        assert_eq!(head.hidden_weights.len(), 4 * head.hidden_dimension);
        let input = [0.1, -0.4, 0.7, 0.2];
        let before = head.predict_inputs(&input).unwrap();
        head.apply_gradient(&input, &[0.5, -0.25], 0.01).unwrap();
        let after = head.predict_inputs(&input).unwrap();
        assert_ne!(before, after);

        let legacy = CriticalityHead {
            input_dimension: 2,
            output_dimension: 1,
            weights: vec![0.5, -0.25],
            bias: vec![0.1],
            hidden_dimension: 0,
            hidden_weights: Vec::new(),
            hidden_bias: Vec::new(),
            hidden_output_weights: Vec::new(),
        };
        assert_eq!(legacy.predict_inputs(&[2.0, 4.0]).unwrap(), vec![0.1]);
    }

    #[test]
    fn learned_representation_has_separate_facets_without_label_leakage() {
        let graph = graph();
        let encoder = ReferenceRelationalAutoencoder::new(ModelConfig {
            hidden_dimension: 8,
            graph_layers: 1,
            ..ModelConfig::default()
        });
        let embeddings = encoder
            .encode(&graph, &MaskSelection::all_unmasked(&graph))
            .unwrap();
        let representation = TrainableLineRepresentationModel::from_graph(
            &graph,
            &embeddings,
            RepresentationConfig {
                base_dimension: 16,
                city_dimension: 8,
                general_dimension: 12,
                role_dimension: 10,
                service_dimension: 8,
                geometry_dimension: 8,
                resilience_dimension: 8,
                seed: 5,
            },
        )
        .unwrap();
        let clean = representation.encode(&graph, &embeddings).unwrap();
        let labelled = representation
            .encode_with_labels(
                &graph,
                &embeddings,
                &[LineImpactLabel {
                    snapshot: graph.manifest.snapshot_id.clone(),
                    line: 0.into(),
                    accessibility_auc_loss: 999.0,
                    unreachable_share: 999.0,
                    mean_delay_reachable_seconds: 999.0,
                    p95_delay_reachable_seconds: 999.0,
                    mean_extra_transfers: 999.0,
                    stations_losing_all_service_share: 999.0,
                    query_count: 1,
                }],
            )
            .unwrap();
        assert_eq!(clean.lines[0].base, labelled.lines[0].base);
        assert_eq!(clean.lines[0].role, labelled.lines[0].role);
        assert_eq!(clean.lines[0].service, labelled.lines[0].service);
        assert_eq!(clean.lines[0].geometry, labelled.lines[0].geometry);
        assert_eq!(clean.lines[0].resilience, labelled.lines[0].resilience);
        assert_eq!(labelled.criticality_accessibility_loss[0], Some(999.0));
    }
}
