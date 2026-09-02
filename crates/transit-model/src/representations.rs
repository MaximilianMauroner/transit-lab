//! Reusable line representations built on top of the shared GTFS encoder.
//!
//! The LibTorch model can learn these projections. The reference backend uses
//! deterministic projections so that the full data and retrieval pipeline can
//! be exercised on a CPU without pretending that a hand-written projection is
//! a trained neural network.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use transit_domain::SERVICE_DAY_BINS;
use transit_graph::{GraphTensor, TEMPORAL_CHANNELS};
use transit_labels::LineImpactLabel;

use crate::Embeddings;

pub const BASE_LINE_EMBEDDING_DIM: usize = 192;
pub const CITY_EMBEDDING_DIM: usize = 128;
pub const GENERAL_EMBEDDING_DIM: usize = 64;
pub const ROLE_EMBEDDING_DIM: usize = 48;
pub const SERVICE_EMBEDDING_DIM: usize = 32;
pub const GEOMETRY_EMBEDDING_DIM: usize = 32;
pub const RESILIENCE_EMBEDDING_DIM: usize = 32;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepresentationConfig {
    pub base_dimension: usize,
    pub city_dimension: usize,
    pub general_dimension: usize,
    pub role_dimension: usize,
    pub service_dimension: usize,
    pub geometry_dimension: usize,
    pub resilience_dimension: usize,
    pub seed: u64,
}

impl Default for RepresentationConfig {
    fn default() -> Self {
        Self {
            base_dimension: BASE_LINE_EMBEDDING_DIM,
            city_dimension: CITY_EMBEDDING_DIM,
            general_dimension: GENERAL_EMBEDDING_DIM,
            role_dimension: ROLE_EMBEDDING_DIM,
            service_dimension: SERVICE_EMBEDDING_DIM,
            geometry_dimension: GEOMETRY_EMBEDDING_DIM,
            resilience_dimension: RESILIENCE_EMBEDDING_DIM,
            seed: 73,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingFacet {
    General,
    NetworkRole,
    Service,
    Geometry,
    Resilience,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineEmbedding {
    pub base: Vec<f32>,
    pub general: Vec<f32>,
    pub role: Vec<f32>,
    pub service: Vec<f32>,
    pub geometry: Vec<f32>,
    pub resilience: Vec<f32>,
}

impl LineEmbedding {
    pub fn facet(&self, facet: EmbeddingFacet) -> &[f32] {
        match facet {
            EmbeddingFacet::General => &self.general,
            EmbeddingFacet::NetworkRole => &self.role,
            EmbeddingFacet::Service => &self.service,
            EmbeddingFacet::Geometry => &self.geometry,
            EmbeddingFacet::Resilience => &self.resilience,
        }
    }

    pub fn dimensions(&self) -> [usize; 6] {
        [
            self.base.len(),
            self.general.len(),
            self.role.len(),
            self.service.len(),
            self.geometry.len(),
            self.resilience.len(),
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineRepresentationSet {
    pub snapshot_id: String,
    pub city: Vec<f32>,
    pub lines: Vec<LineEmbedding>,
    /// Available when exact simulator labels were supplied during encoding.
    /// `None` means this set is unsupervised for the resilience comparison.
    pub criticality_accessibility_loss: Vec<Option<f32>>,
}

/// Feed-derived inputs used by the task-specific projection heads. These are
/// deliberately kept separate from simulator labels: labels supervise the
/// heads during training, but can never leak into an inference embedding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawLineFeatures {
    pub base: Vec<f32>,
    pub role: Vec<f32>,
    pub service: Vec<f32>,
    pub geometry: Vec<f32>,
    pub resilience: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawLineFeatureSet {
    pub city: Vec<f32>,
    pub lines: Vec<RawLineFeatures>,
}

impl LineRepresentationSet {
    pub fn validate(&self, expected_lines: usize) -> Result<()> {
        if self.lines.len() != expected_lines
            || self.criticality_accessibility_loss.len() != expected_lines
        {
            bail!("line representation count does not match the graph");
        }
        if self.city.iter().any(|value| !value.is_finite())
            || self.lines.iter().any(|line| {
                line.base
                    .iter()
                    .chain(&line.general)
                    .chain(&line.role)
                    .chain(&line.service)
                    .chain(&line.geometry)
                    .chain(&line.resilience)
                    .any(|value| !value.is_finite())
            })
        {
            bail!("line representations contain a non-finite value");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceLineRepresentationEncoder {
    pub config: RepresentationConfig,
}

impl Default for ReferenceLineRepresentationEncoder {
    fn default() -> Self {
        Self::new(RepresentationConfig::default())
    }
}

impl ReferenceLineRepresentationEncoder {
    pub fn new(config: RepresentationConfig) -> Self {
        Self { config }
    }

    pub fn encode(
        &self,
        graph: &GraphTensor,
        embeddings: &Embeddings,
    ) -> Result<LineRepresentationSet> {
        self.encode_with_labels(graph, embeddings, &[])
    }

    pub fn raw_features(
        &self,
        graph: &GraphTensor,
        embeddings: &Embeddings,
    ) -> Result<RawLineFeatureSet> {
        graph.validate()?;
        if embeddings.station.len() != graph.manifest.station_count
            || embeddings.line.len() != graph.manifest.line_count
        {
            bail!("graph embeddings do not match the graph manifest");
        }
        let dimensions = [
            self.config.base_dimension,
            self.config.city_dimension,
            self.config.general_dimension,
            self.config.role_dimension,
            self.config.service_dimension,
            self.config.geometry_dimension,
            self.config.resilience_dimension,
        ];
        if dimensions.contains(&0) {
            bail!("representation dimensions must be positive");
        }

        let mut lines = Vec::with_capacity(graph.manifest.line_count);
        for line in 0..graph.manifest.line_count {
            let service = service_features(graph, line);
            let geometry = geometry_features(graph, line);
            let role = role_features(graph, embeddings, line);
            let resilience = resilience_features(graph, line, &role, &service);
            let mut base = Vec::new();
            base.extend(&embeddings.line[line]);
            base.extend(&embeddings.city);
            base.extend(&service);
            base.extend(&geometry);
            base.extend(&role);
            base.extend(&resilience);
            lines.push(RawLineFeatures {
                base,
                role,
                service,
                geometry,
                resilience,
            });
        }
        Ok(RawLineFeatureSet {
            city: embeddings.city.clone(),
            lines,
        })
    }

    pub fn encode_with_labels(
        &self,
        graph: &GraphTensor,
        embeddings: &Embeddings,
        labels: &[LineImpactLabel],
    ) -> Result<LineRepresentationSet> {
        let raw = self.raw_features(graph, embeddings)?;
        let city = project(&raw.city, self.config.city_dimension, self.config.seed + 1);
        let mut lines = Vec::with_capacity(graph.manifest.line_count);
        let mut criticality_accessibility_loss = vec![None; graph.manifest.line_count];

        for label in labels {
            if label.snapshot == graph.manifest.snapshot_id {
                if let Some(slot) = criticality_accessibility_loss.get_mut(label.line.0 as usize) {
                    *slot = Some(label.accessibility_auc_loss);
                }
            }
        }

        for line in 0..graph.manifest.line_count {
            let service_input = &raw.lines[line].service;
            let geometry_input = &raw.lines[line].geometry;
            let role_input = &raw.lines[line].role;
            let resilience_input = &raw.lines[line].resilience;
            let base_input = &raw.lines[line].base;
            let base = project(
                base_input,
                self.config.base_dimension,
                self.config.seed + 11,
            );

            let mut role_projection = base.clone();
            role_projection.extend(role_input.iter().copied());
            let role = project(
                &role_projection,
                self.config.role_dimension,
                self.config.seed + 17,
            );

            let service = project(
                &service_input,
                self.config.service_dimension,
                self.config.seed + 23,
            );
            let geometry = project(
                &geometry_input,
                self.config.geometry_dimension,
                self.config.seed + 29,
            );

            let mut resilience_projection = resilience_input.clone();
            resilience_projection.extend(&base);
            let resilience = project(
                &resilience_projection,
                self.config.resilience_dimension,
                self.config.seed + 31,
            );

            let mut general_projection = base.clone();
            general_projection.extend(&role);
            general_projection.extend(&service);
            general_projection.extend(&geometry);
            general_projection.extend(&resilience);
            let general = project(
                &general_projection,
                self.config.general_dimension,
                self.config.seed + 37,
            );

            lines.push(LineEmbedding {
                base,
                general,
                role,
                service,
                geometry,
                resilience,
            });
        }

        let result = LineRepresentationSet {
            snapshot_id: graph.manifest.snapshot_id.clone(),
            city,
            lines,
            criticality_accessibility_loss,
        };
        result.validate(graph.manifest.line_count)?;
        Ok(result)
    }
}

/// A small trainable projection used by the dependency-free CPU backend.
/// `forward_raw` is useful for metric-learning updates; `forward` returns the
/// normalized vector used by cosine retrieval.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionHead {
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl ProjectionHead {
    pub fn new(input_dimension: usize, output_dimension: usize, seed: u64) -> Self {
        let mut random = SplitMix64::new(seed);
        let scale = 0.08 / (input_dimension.max(1) as f32).sqrt();
        Self {
            input_dimension,
            output_dimension,
            weights: (0..input_dimension * output_dimension)
                .map(|_| (random.next_f32() - 0.5) * scale)
                .collect(),
            bias: vec![0.0; output_dimension],
        }
    }

    pub fn forward_raw(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.validate_input(input)?;
        Ok((0..self.output_dimension)
            .map(|output| {
                let offset = output * self.input_dimension;
                (self.bias[output]
                    + self.weights[offset..offset + self.input_dimension]
                        .iter()
                        .zip(input)
                        .map(|(weight, value)| weight * bounded(*value))
                        .sum::<f32>())
                .tanh()
            })
            .collect())
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = self.forward_raw(input)?;
        normalize_vector(&mut output);
        Ok(output)
    }

    /// Apply a gradient with respect to the unnormalized `tanh` output.
    pub fn apply_gradient(
        &mut self,
        input: &[f32],
        output_gradient: &[f32],
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<()> {
        self.validate_input(input)?;
        if output_gradient.len() != self.output_dimension {
            bail!("projection gradient width does not match the projection head");
        }
        let activated = self.forward_raw(input)?;
        self.apply_gradient_from_activated(
            input,
            &activated,
            output_gradient,
            learning_rate,
            weight_decay,
        )
    }

    pub fn apply_gradient_from_activated(
        &mut self,
        input: &[f32],
        activated: &[f32],
        output_gradient: &[f32],
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<()> {
        self.validate_input(input)?;
        if activated.len() != self.output_dimension
            || output_gradient.len() != self.output_dimension
        {
            bail!("projection gradient width does not match the projection head");
        }
        for (output, gradient) in output_gradient.iter().enumerate() {
            let delta = *gradient * (1.0 - activated[output] * activated[output]);
            let offset = output * self.input_dimension;
            for (weight, feature) in self.weights[offset..offset + self.input_dimension]
                .iter_mut()
                .zip(input)
            {
                *weight -= learning_rate * (delta * bounded(*feature) + weight_decay * *weight);
            }
            self.bias[output] -= learning_rate * delta;
        }
        Ok(())
    }

    fn validate_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.input_dimension {
            bail!(
                "projection input width {} does not match {}",
                input.len(),
                self.input_dimension
            );
        }
        if self.weights.len() != self.input_dimension * self.output_dimension
            || self.bias.len() != self.output_dimension
        {
            bail!("projection head parameters do not match their declared shape");
        }
        Ok(())
    }
}

fn bounded(value: f32) -> f32 {
    if value.is_finite() {
        value / (1.0 + value.abs())
    } else {
        0.0
    }
}

/// Learned task-specific projections over the shared graph encoder. The
/// backbone remains inductive: all input widths are derived from the stable
/// graph schema, never from a city or a raw GTFS identifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainableLineRepresentationModel {
    pub config: RepresentationConfig,
    pub city: ProjectionHead,
    pub base: ProjectionHead,
    pub role: ProjectionHead,
    pub service: ProjectionHead,
    pub geometry: ProjectionHead,
    pub resilience: ProjectionHead,
    pub general: ProjectionHead,
}

impl TrainableLineRepresentationModel {
    pub fn from_graph(
        graph: &GraphTensor,
        embeddings: &Embeddings,
        config: RepresentationConfig,
    ) -> Result<Self> {
        let extractor = ReferenceLineRepresentationEncoder::new(config.clone());
        let raw = extractor.raw_features(graph, embeddings)?;
        Self::from_raw_features(&raw, config)
    }

    pub fn from_raw_features(
        raw: &RawLineFeatureSet,
        config: RepresentationConfig,
    ) -> Result<Self> {
        let Some(first) = raw.lines.first() else {
            bail!("cannot initialize representation heads without line features");
        };
        if raw.lines.iter().any(|line| {
            line.base.len() != first.base.len()
                || line.role.len() != first.role.len()
                || line.service.len() != first.service.len()
                || line.geometry.len() != first.geometry.len()
                || line.resilience.len() != first.resilience.len()
        }) {
            bail!("raw line feature widths are inconsistent");
        }
        if [
            config.base_dimension,
            config.city_dimension,
            config.general_dimension,
            config.role_dimension,
            config.service_dimension,
            config.geometry_dimension,
            config.resilience_dimension,
        ]
        .contains(&0)
        {
            bail!("representation dimensions must be positive");
        }
        let role_width = config.base_dimension + first.role.len();
        let resilience_width = config.base_dimension + first.resilience.len();
        let general_width = config.base_dimension
            + config.role_dimension
            + config.service_dimension
            + config.geometry_dimension
            + config.resilience_dimension;
        Ok(Self {
            city: ProjectionHead::new(raw.city.len(), config.city_dimension, config.seed + 101),
            base: ProjectionHead::new(first.base.len(), config.base_dimension, config.seed + 103),
            role: ProjectionHead::new(role_width, config.role_dimension, config.seed + 107),
            service: ProjectionHead::new(
                first.service.len(),
                config.service_dimension,
                config.seed + 109,
            ),
            geometry: ProjectionHead::new(
                first.geometry.len(),
                config.geometry_dimension,
                config.seed + 113,
            ),
            resilience: ProjectionHead::new(
                resilience_width,
                config.resilience_dimension,
                config.seed + 127,
            ),
            general: ProjectionHead::new(
                general_width,
                config.general_dimension,
                config.seed + 131,
            ),
            config,
        })
    }

    pub fn encode(
        &self,
        graph: &GraphTensor,
        embeddings: &Embeddings,
    ) -> Result<LineRepresentationSet> {
        self.encode_with_labels(graph, embeddings, &[])
    }

    pub fn encode_with_labels(
        &self,
        graph: &GraphTensor,
        embeddings: &Embeddings,
        labels: &[LineImpactLabel],
    ) -> Result<LineRepresentationSet> {
        let raw = self.raw_features(graph, embeddings)?;
        let city = self.city.forward(&raw.city)?;
        let mut lines = Vec::with_capacity(raw.lines.len());
        let mut criticality_accessibility_loss = vec![None; raw.lines.len()];
        for label in labels {
            if label.snapshot == graph.manifest.snapshot_id {
                if let Some(slot) = criticality_accessibility_loss.get_mut(label.line.0 as usize) {
                    *slot = Some(label.accessibility_auc_loss);
                }
            }
        }
        for raw_line in &raw.lines {
            let base = self.base.forward(&raw_line.base)?;
            let mut role_input = base.clone();
            role_input.extend(&raw_line.role);
            let role = self.role.forward(&role_input)?;
            let service = self.service.forward(&raw_line.service)?;
            let geometry = self.geometry.forward(&raw_line.geometry)?;
            let mut resilience_input = base.clone();
            resilience_input.extend(&raw_line.resilience);
            let resilience = self.resilience.forward(&resilience_input)?;
            let mut general_input = base.clone();
            general_input.extend(&role);
            general_input.extend(&service);
            general_input.extend(&geometry);
            general_input.extend(&resilience);
            let general = self.general.forward(&general_input)?;
            lines.push(LineEmbedding {
                base,
                general,
                role,
                service,
                geometry,
                resilience,
            });
        }
        let result = LineRepresentationSet {
            snapshot_id: graph.manifest.snapshot_id.clone(),
            city,
            lines,
            criticality_accessibility_loss,
        };
        result.validate(graph.manifest.line_count)?;
        Ok(result)
    }

    pub fn raw_features(
        &self,
        graph: &GraphTensor,
        embeddings: &Embeddings,
    ) -> Result<RawLineFeatureSet> {
        ReferenceLineRepresentationEncoder::new(self.config.clone()).raw_features(graph, embeddings)
    }
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

fn service_features(graph: &GraphTensor, line: usize) -> Vec<f32> {
    let mut output = graph.line_features.row(line).to_vec();
    let temporal = graph.line_temporal.row(line);
    output.extend_from_slice(temporal);
    for channel in 0..TEMPORAL_CHANNELS {
        let values = &temporal[channel * SERVICE_DAY_BINS..(channel + 1) * SERVICE_DAY_BINS];
        append_summary(&mut output, values);
    }
    output
}

/// Encode ordered transit segments into fixed-width sequence bins. The graph
/// compiler stores relative route position on each segment, so this preserves
/// route order without adding variable-length tensors to the base graph file.
fn geometry_features(graph: &GraphTensor, line: usize) -> Vec<f32> {
    const SEQUENCE_BINS: usize = 16;
    let mut edges: Vec<usize> = graph
        .transit_line
        .iter()
        .enumerate()
        .filter_map(|(edge, value)| (*value as usize == line).then_some(edge))
        .collect();
    edges.sort_by(|left, right| {
        graph.transit_features.row(*left)[4]
            .total_cmp(&graph.transit_features.row(*right)[4])
            .then_with(|| left.cmp(right))
    });

    let mut bins = vec![vec![0.0_f32; graph.transit_features.cols]; SEQUENCE_BINS];
    let mut counts = vec![0_u32; SEQUENCE_BINS];
    let mut output = Vec::with_capacity(SEQUENCE_BINS * graph.transit_features.cols + 24);
    for edge in edges {
        let row = graph.transit_features.row(edge);
        let position = row.get(4).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let bin = ((position * SEQUENCE_BINS as f32).floor() as usize).min(SEQUENCE_BINS - 1);
        for (sum, value) in bins[bin].iter_mut().zip(row) {
            *sum += *value;
        }
        counts[bin] += 1;
    }
    for (bin, count) in bins.iter_mut().zip(counts) {
        if count > 0 {
            for value in bin.iter_mut() {
                *value /= count as f32;
            }
        }
        output.extend(bin.iter().copied());
    }

    let line_row = graph.line_features.row(line);
    output.extend(
        [5_usize, 7, 8, 9, 15, 16, 17]
            .into_iter()
            .filter_map(|index| line_row.get(index).copied()),
    );
    let summary_source = output.clone();
    append_summary(&mut output, &summary_source);
    output
}

fn role_features(graph: &GraphTensor, embeddings: &Embeddings, line: usize) -> Vec<f32> {
    let hidden = embeddings.line.first().map(Vec::len).unwrap_or(0);
    let mut station_values = Vec::<&Vec<f32>>::new();
    for (station, destination) in graph.serves_src.iter().zip(&graph.serves_dst) {
        if *destination as usize == line {
            if let Some(value) = embeddings.station.get(*station as usize) {
                station_values.push(value);
            }
        }
    }
    let mut interchange_values = Vec::<&Vec<f32>>::new();
    for (from, to) in graph.interchange_src.iter().zip(&graph.interchange_dst) {
        if *from as usize == line {
            if let Some(value) = embeddings.line.get(*to as usize) {
                interchange_values.push(value);
            }
        }
        if *to as usize == line {
            if let Some(value) = embeddings.line.get(*from as usize) {
                interchange_values.push(value);
            }
        }
    }

    let mut output = Vec::with_capacity(hidden * 4 + 24);
    append_vector_pool(&mut output, &station_values, hidden);
    append_vector_pool(&mut output, &interchange_values, hidden);

    let mut station_context = Vec::new();
    for (station, destination) in graph.serves_src.iter().zip(&graph.serves_dst) {
        if *destination as usize != line {
            continue;
        }
        let row = graph.station_features.row(*station as usize);
        station_context.extend([
            row.get(0).copied().unwrap_or(0.0),
            row.get(1).copied().unwrap_or(0.0),
            row.get(3).copied().unwrap_or(0.0),
            row.get(4).copied().unwrap_or(0.0),
            row.get(14).copied().unwrap_or(0.0),
            row.get(15).copied().unwrap_or(0.0),
        ]);
    }
    append_summary(&mut output, &station_context);
    output.push(
        graph
            .interchange_src
            .iter()
            .filter(|value| **value as usize == line)
            .count() as f32,
    );
    output.push(
        graph
            .interchange_dst
            .iter()
            .filter(|value| **value as usize == line)
            .count() as f32,
    );
    output
}

fn resilience_features(
    graph: &GraphTensor,
    line: usize,
    role: &[f32],
    service: &[f32],
) -> Vec<f32> {
    let mut output = graph.line_features.row(line).to_vec();
    append_summary(&mut output, role);
    append_summary(&mut output, service);
    output
}

fn append_vector_pool(output: &mut Vec<f32>, values: &[&Vec<f32>], width: usize) {
    if values.is_empty() {
        output.extend(std::iter::repeat(0.0).take(width * 2));
        return;
    }
    for index in 0..width {
        output.push(
            values
                .iter()
                .map(|value| value.get(index).copied().unwrap_or(0.0))
                .sum::<f32>()
                / values.len() as f32,
        );
    }
    for index in 0..width {
        output.push(
            values
                .iter()
                .map(|value| value.get(index).copied().unwrap_or(0.0))
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0),
        );
    }
}

fn append_summary(output: &mut Vec<f32>, values: &[f32]) {
    if values.is_empty() {
        output.extend([0.0, 0.0, 0.0, 0.0]);
        return;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    output.extend([mean, max, min, variance.sqrt()]);
}

fn project(input: &[f32], output_width: usize, seed: u64) -> Vec<f32> {
    if input.is_empty() {
        return vec![0.0; output_width];
    }
    let input_scale = (input.len() as f32).sqrt().max(1.0);
    let mut result = Vec::with_capacity(output_width);
    for output in 0..output_width {
        let mut value = 0.0;
        for (index, feature) in input.iter().enumerate() {
            let bounded = *feature / (1.0 + feature.abs());
            let angle =
                (index as f32 + 1.0) * (output as f32 + 1.0) * (seed as f32 * 0.001 + 0.017);
            value += bounded * (angle.sin() + 0.5 * (angle * 0.61).cos());
        }
        result.push((value / input_scale).tanh());
    }
    normalize_vector(&mut result);
    result
}

fn normalize_vector(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in values {
            *value /= norm;
        }
    }
}
