//! Cross-snapshot line retrieval over task-specific representation facets.
//!
//! Retrieval deliberately scans the candidate lines directly. That keeps the
//! result auditable and is sufficient for the first tens of thousands of
//! lines; an ANN index can be added later without changing the scoring API.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use transit_graph::GraphTensor;
use transit_model::{EmbeddingFacet, LineRepresentationSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimilarityProfile {
    General,
    #[serde(rename = "network-role")]
    NetworkRole,
    Service,
    Geometry,
    Resilience,
    Weighted {
        role: f32,
        service: f32,
        geometry: f32,
        resilience: f32,
    },
}

impl SimilarityProfile {
    pub fn facet(&self) -> Option<EmbeddingFacet> {
        match self {
            Self::General => Some(EmbeddingFacet::General),
            Self::NetworkRole => Some(EmbeddingFacet::NetworkRole),
            Self::Service => Some(EmbeddingFacet::Service),
            Self::Geometry => Some(EmbeddingFacet::Geometry),
            Self::Resilience | Self::Weighted { .. } => None,
        }
    }

    fn score(&self, facets: &FacetScores) -> Result<f32> {
        match self {
            Self::General => Ok(facets.general),
            Self::NetworkRole => Ok(facets.role),
            Self::Service => Ok(facets.service),
            Self::Geometry => Ok(facets.geometry),
            Self::Resilience => Ok(facets.resilience),
            Self::Weighted {
                role,
                service,
                geometry,
                resilience,
            } => {
                let weights = [*role, *service, *geometry, *resilience];
                if weights
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
                {
                    bail!("similarity weights must be finite and non-negative");
                }
                let total = weights.iter().sum::<f32>();
                if total <= f32::EPSILON {
                    bail!("at least one similarity weight must be positive");
                }
                Ok((role * facets.role
                    + service * facets.service
                    + geometry * facets.geometry
                    + resilience * facets.resilience)
                    / total)
            }
        }
    }
}

impl FromStr for SimilarityProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        if let Some(weights) = normalized.strip_prefix("weighted:") {
            let values: Vec<f32> = weights
                .split(',')
                .map(|value| {
                    value
                        .trim()
                        .parse::<f32>()
                        .map_err(|_| format!("invalid weighted profile value {value:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != 4 {
                return Err(
                    "weighted profile expects role,service,geometry,resilience weights".to_owned(),
                );
            }
            return Ok(Self::Weighted {
                role: values[0],
                service: values[1],
                geometry: values[2],
                resilience: values[3],
            });
        }
        match normalized.as_str() {
            "general" => Ok(Self::General),
            "role" | "network-role" | "network_role" => Ok(Self::NetworkRole),
            "service" => Ok(Self::Service),
            "geometry" => Ok(Self::Geometry),
            "resilience" => Ok(Self::Resilience),
            value => Err(format!(
                "unknown similarity profile {value}; expected general, network-role, service, geometry, or resilience"
            )),
        }
    }
}

impl fmt::Display for SimilarityProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::General => "general",
            Self::NetworkRole => "network-role",
            Self::Service => "service",
            Self::Geometry => "geometry",
            Self::Resilience => "resilience",
            Self::Weighted { .. } => "weighted",
        };
        formatter.write_str(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FacetScores {
    pub general: f32,
    pub role: f32,
    pub service: f32,
    pub geometry: f32,
    pub resilience: f32,
}

impl FacetScores {
    pub fn between(
        left: &transit_model::LineEmbedding,
        right: &transit_model::LineEmbedding,
    ) -> Self {
        Self {
            general: cosine_similarity(
                left.facet(EmbeddingFacet::General),
                right.facet(EmbeddingFacet::General),
            ),
            role: cosine_similarity(
                left.facet(EmbeddingFacet::NetworkRole),
                right.facet(EmbeddingFacet::NetworkRole),
            ),
            service: cosine_similarity(
                left.facet(EmbeddingFacet::Service),
                right.facet(EmbeddingFacet::Service),
            ),
            geometry: cosine_similarity(
                left.facet(EmbeddingFacet::Geometry),
                right.facet(EmbeddingFacet::Geometry),
            ),
            resilience: cosine_similarity(
                left.facet(EmbeddingFacet::Resilience),
                right.facet(EmbeddingFacet::Resilience),
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineComparison {
    pub same_mode: bool,
    pub transfer_degree_percentile_difference: f32,
    pub frequency_profile_distance: f32,
    pub route_length_ratio: f32,
    pub criticality_percentile_difference: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineMatch {
    pub line: u32,
    pub line_name: String,
    pub similarity: f32,
    pub facet_scores: FacetScores,
    pub comparison: LineComparison,
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (&a, &b) in left.iter().zip(right) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator <= f32::EPSILON {
        0.0
    } else {
        dot / denominator
    }
}

pub fn rank_similar_lines(
    query_graph: &GraphTensor,
    query_representations: &LineRepresentationSet,
    query_line: usize,
    candidate_graph: &GraphTensor,
    candidate_representations: &LineRepresentationSet,
    profile: &SimilarityProfile,
    top_k: usize,
) -> Result<Vec<LineMatch>> {
    query_graph.validate()?;
    candidate_graph.validate()?;
    query_representations.validate(query_graph.manifest.line_count)?;
    candidate_representations.validate(candidate_graph.manifest.line_count)?;
    let query_embedding = query_representations
        .lines
        .get(query_line)
        .ok_or_else(|| anyhow::anyhow!("query line index {query_line} is out of bounds"))?;

    let mut matches = Vec::with_capacity(candidate_representations.lines.len());
    for (line, candidate_embedding) in candidate_representations.lines.iter().enumerate() {
        if query_graph.manifest.snapshot_id == candidate_graph.manifest.snapshot_id
            && query_line == line
        {
            continue;
        }
        let facets = FacetScores::between(query_embedding, candidate_embedding);
        matches.push(LineMatch {
            line: line as u32,
            line_name: candidate_graph
                .line_names
                .get(line)
                .cloned()
                .unwrap_or_else(|| format!("Line {line}")),
            similarity: profile.score(&facets)?,
            facet_scores: facets,
            comparison: compare_lines(
                query_graph,
                query_representations,
                query_line,
                candidate_graph,
                candidate_representations,
                line,
            ),
        });
    }
    matches.sort_by(|left, right| {
        right
            .similarity
            .total_cmp(&left.similarity)
            .then_with(|| left.line_name.cmp(&right.line_name))
            .then_with(|| left.line.cmp(&right.line))
    });
    matches.truncate(top_k);
    Ok(matches)
}

fn compare_lines(
    query_graph: &GraphTensor,
    query_representations: &LineRepresentationSet,
    query_line: usize,
    candidate_graph: &GraphTensor,
    candidate_representations: &LineRepresentationSet,
    candidate_line: usize,
) -> LineComparison {
    LineComparison {
        same_mode: mode_index(query_graph, query_line)
            == mode_index(candidate_graph, candidate_line),
        transfer_degree_percentile_difference: (feature_percentile(query_graph, 15, query_line)
            - feature_percentile(candidate_graph, 15, candidate_line))
        .abs(),
        frequency_profile_distance: profile_distance(
            query_graph,
            query_line,
            candidate_graph,
            candidate_line,
        ),
        route_length_ratio: ratio(query_graph, 7, query_line, candidate_graph, candidate_line),
        criticality_percentile_difference: criticality_difference(
            query_representations,
            query_line,
            candidate_representations,
            candidate_line,
        ),
    }
}

fn mode_index(graph: &GraphTensor, line: usize) -> Option<usize> {
    graph.line_features.row(line).get(..5).and_then(|values| {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
    })
}

fn feature_percentile(graph: &GraphTensor, feature: usize, line: usize) -> f32 {
    let Some(value) = graph.line_features.row(line).get(feature).copied() else {
        return 0.0;
    };
    let values: Vec<f32> = (0..graph.manifest.line_count)
        .filter_map(|index| graph.line_features.row(index).get(feature).copied())
        .collect();
    if values.len() <= 1 {
        return 0.5;
    }
    let below = values.iter().filter(|other| **other < value).count() as f32;
    let equal = values.iter().filter(|other| **other == value).count() as f32;
    (below + equal * 0.5) / values.len() as f32
}

fn profile_distance(
    left_graph: &GraphTensor,
    left_line: usize,
    right_graph: &GraphTensor,
    right_line: usize,
) -> f32 {
    let left = service_profile(left_graph, left_line);
    let right = service_profile(right_graph, right_line);
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 1.0;
    }
    (left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / left.len() as f32)
        .sqrt()
}

fn service_profile(graph: &GraphTensor, line: usize) -> Vec<f32> {
    let row = graph.line_temporal.row(line);
    let bins = graph.manifest.temporal_bins;
    let mut output = Vec::with_capacity(bins * 2);
    for channel in [0, 2] {
        let values = row.get(channel * bins..(channel + 1) * bins).unwrap_or(&[]);
        let maximum = values.iter().copied().fold(0.0_f32, f32::max).max(1.0);
        output.extend(values.iter().map(|value| *value / maximum));
    }
    output
}

fn ratio(
    left_graph: &GraphTensor,
    feature: usize,
    left_line: usize,
    right_graph: &GraphTensor,
    right_line: usize,
) -> f32 {
    let left = left_graph
        .line_features
        .row(left_line)
        .get(feature)
        .copied()
        .unwrap_or(0.0)
        .abs();
    let right = right_graph
        .line_features
        .row(right_line)
        .get(feature)
        .copied()
        .unwrap_or(0.0)
        .abs();
    if left <= f32::EPSILON || right <= f32::EPSILON {
        1.0
    } else {
        (left / right).max(right / left)
    }
}

fn criticality_difference(
    left: &LineRepresentationSet,
    left_line: usize,
    right: &LineRepresentationSet,
    right_line: usize,
) -> Option<f32> {
    let left_percentile = optional_percentile(&left.criticality_accessibility_loss, left_line)?;
    let right_percentile = optional_percentile(&right.criticality_accessibility_loss, right_line)?;
    Some((left_percentile - right_percentile).abs())
}

fn optional_percentile(values: &[Option<f32>], index: usize) -> Option<f32> {
    let value = values.get(index).copied().flatten()?;
    let available: Vec<f32> = values.iter().flatten().copied().collect();
    if available.len() <= 1 {
        return Some(0.5);
    }
    let below = available.iter().filter(|other| **other < value).count() as f32;
    let equal = available.iter().filter(|other| **other == value).count() as f32;
    Some((below + equal * 0.5) / available.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_is_zero_for_invalid_shapes() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parses_profiles_without_collapsing_facets() {
        assert_eq!(
            "network-role"
                .parse::<SimilarityProfile>()
                .unwrap()
                .to_string(),
            "network-role"
        );
        assert_eq!(
            "service".parse::<SimilarityProfile>().unwrap().to_string(),
            "service"
        );
        assert!("unknown".parse::<SimilarityProfile>().is_err());
    }

    #[test]
    fn parses_explicit_weighted_profiles() {
        let profile = "weighted:0.4,0.3,0.2,0.1"
            .parse::<SimilarityProfile>()
            .unwrap();
        assert_eq!(profile.to_string(), "weighted");
        assert!("weighted:1,0".parse::<SimilarityProfile>().is_err());
    }
}
