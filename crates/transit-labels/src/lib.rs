//! Aggregate counterfactual labels generated from exact timetable routing.

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use transit_domain::{LineIndex, StationIndex, INF_TIME};
use transit_router::{OneToAllResult, Router};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelGenerationConfig {
    pub accessibility_thresholds_seconds: Vec<u32>,
    pub maximum_origins: usize,
}

impl Default for LabelGenerationConfig {
    fn default() -> Self {
        Self {
            accessibility_thresholds_seconds: vec![15, 30, 45, 60, 90]
                .into_iter()
                .map(|minutes| minutes * 60)
                .collect(),
            maximum_origins: 256,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineImpactLabel {
    pub snapshot: String,
    pub line: LineIndex,
    pub accessibility_auc_loss: f32,
    pub unreachable_share: f32,
    pub mean_delay_reachable_seconds: f32,
    pub p95_delay_reachable_seconds: f32,
    pub mean_extra_transfers: f32,
    /// Fraction of all canonical stations whose active service is provided
    /// only by the removed line.
    pub stations_losing_all_service_share: f32,
    pub query_count: u32,
}

#[derive(Clone, Debug)]
struct BaselineQuery {
    origin: StationIndex,
    departure: u32,
    result: OneToAllResult,
}

pub fn generate_line_removal_labels(
    router: &Router,
    snapshot: impl Into<String>,
    origins: &[StationIndex],
    departures: &[u32],
    config: &LabelGenerationConfig,
) -> Vec<LineImpactLabel> {
    let origins = select_origins(origins, config.maximum_origins, router.data.station_count);
    if origins.is_empty() || departures.is_empty() || router.data.line_count == 0 {
        return Vec::new();
    }
    let queries: Vec<(StationIndex, u32)> = origins
        .iter()
        .copied()
        .flat_map(|origin| {
            departures
                .iter()
                .copied()
                .map(move |departure| (origin, departure))
        })
        .collect();
    let baselines: Vec<BaselineQuery> = queries
        .par_iter()
        .map(|(origin, departure)| BaselineQuery {
            origin: *origin,
            departure: *departure,
            result: router.one_to_all(
                *origin,
                *departure,
                &transit_domain::LineMask::empty(router.data.line_count),
            ),
        })
        .collect();
    let snapshot = snapshot.into();
    (0..router.data.line_count)
        .into_par_iter()
        .map(|line| {
            let line_index = LineIndex(line as u32);
            let stations_losing_all_service_share =
                station_losing_all_service_share(router, line_index);
            let mut auc_loss = 0.0;
            let mut unreachable = 0_u64;
            let mut baseline_reachable = 0_u64;
            let mut delay_values = Vec::new();
            let mut extra_transfer_sum = 0.0_f64;
            let mut extra_transfer_count = 0_u64;

            for baseline in &baselines {
                let disrupted = router.one_to_all(
                    baseline.origin,
                    baseline.departure,
                    &transit_domain::LineMask::single(router.data.line_count, line_index),
                );
                let destination_count = baseline.result.arrival_seconds.len().max(1) as f64;
                for threshold in &config.accessibility_thresholds_seconds {
                    let intact = count_within(&baseline.result, baseline.departure, *threshold)
                        as f64
                        / destination_count;
                    let damaged = count_within(&disrupted, baseline.departure, *threshold) as f64
                        / destination_count;
                    auc_loss += (intact - damaged).max(0.0)
                        / config.accessibility_thresholds_seconds.len().max(1) as f64;
                }
                for destination in 0..baseline.result.arrival_seconds.len() {
                    let intact_arrival = baseline.result.arrival_seconds[destination];
                    if intact_arrival == INF_TIME {
                        continue;
                    }
                    baseline_reachable += 1;
                    let damaged_arrival = disrupted.arrival_seconds[destination];
                    if damaged_arrival == INF_TIME {
                        unreachable += 1;
                        continue;
                    }
                    delay_values.push(damaged_arrival.saturating_sub(intact_arrival) as f32);
                    let intact_transfers = baseline.result.transfers[destination];
                    let damaged_transfers = disrupted.transfers[destination];
                    if intact_transfers != u8::MAX && damaged_transfers != u8::MAX {
                        extra_transfer_sum +=
                            damaged_transfers.saturating_sub(intact_transfers) as f64;
                        extra_transfer_count += 1;
                    }
                }
            }
            delay_values.sort_by(f32::total_cmp);
            let p95_index = if delay_values.is_empty() {
                0
            } else {
                ((delay_values.len() as f32 * 0.95).ceil() as usize)
                    .saturating_sub(1)
                    .min(delay_values.len() - 1)
            };
            let delay_sum: f32 = delay_values.iter().sum();
            let unreachable_share = unreachable as f32 / baseline_reachable.max(1) as f32;
            LineImpactLabel {
                snapshot: snapshot.clone(),
                line: line_index,
                accessibility_auc_loss: (auc_loss / baselines.len().max(1) as f64) as f32,
                unreachable_share,
                mean_delay_reachable_seconds: delay_sum / delay_values.len().max(1) as f32,
                p95_delay_reachable_seconds: delay_values.get(p95_index).copied().unwrap_or(0.0),
                mean_extra_transfers: extra_transfer_sum as f32
                    / extra_transfer_count.max(1) as f32,
                stations_losing_all_service_share,
                query_count: baselines.len() as u32,
            }
        })
        .collect()
}

fn station_losing_all_service_share(router: &Router, disabled_line: LineIndex) -> f32 {
    if router.data.station_count == 0 {
        return 0.0;
    }
    let mut served_by_disabled_line = vec![false; router.data.station_count];
    let mut served_by_other_line = vec![false; router.data.station_count];
    for pattern in &router.data.patterns {
        let target = if pattern.line == disabled_line {
            &mut served_by_disabled_line
        } else {
            &mut served_by_other_line
        };
        for station in &pattern.stops {
            if let Some(value) = target.get_mut(station.0 as usize) {
                *value = true;
            }
        }
    }
    let lost = served_by_disabled_line
        .into_iter()
        .zip(served_by_other_line)
        .filter(|(disabled, other)| *disabled && !*other)
        .count();
    lost as f32 / router.data.station_count as f32
}

fn select_origins(
    origins: &[StationIndex],
    maximum: usize,
    station_count: usize,
) -> Vec<StationIndex> {
    if maximum == 0 || station_count == 0 {
        return Vec::new();
    }
    let mut unique = Vec::with_capacity(origins.len());
    for &origin in origins {
        if origin.0 as usize >= station_count || unique.contains(&origin) {
            continue;
        }
        unique.push(origin);
    }
    if unique.len() <= maximum {
        return unique;
    }
    (0..maximum)
        .map(|index| unique[index * unique.len() / maximum])
        .collect()
}

fn count_within(result: &OneToAllResult, departure: u32, threshold: u32) -> usize {
    result
        .arrival_seconds
        .iter()
        .filter(|arrival| {
            **arrival != INF_TIME && **arrival >= departure && **arrival - departure <= threshold
        })
        .count()
}

pub fn save_jsonl(path: &Path, labels: &[LineImpactLabel]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    for label in labels {
        serde_json::to_writer(&mut file, label).context("encoding line label")?;
        file.write_all(b"\n")
            .context("writing line label newline")?;
    }
    Ok(())
}

pub fn load_jsonl(path: &Path) -> Result<Vec<LineImpactLabel>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut labels = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading label line {}", line_number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        labels.push(
            serde_json::from_str(&line)
                .with_context(|| format!("decoding label line {}", line_number + 1))?,
        );
    }
    Ok(labels)
}

/// Spearman rank correlation with average ranks for ties.
pub fn spearman_rank(prediction: &[f32], target: &[f32]) -> Option<f32> {
    if prediction.len() != target.len() || prediction.len() < 2 {
        return None;
    }
    let prediction_ranks = average_ranks(prediction);
    let target_ranks = average_ranks(target);
    let prediction_mean = prediction_ranks.iter().sum::<f32>() / prediction_ranks.len() as f32;
    let target_mean = target_ranks.iter().sum::<f32>() / target_ranks.len() as f32;
    let mut numerator = 0.0;
    let mut prediction_variance = 0.0;
    let mut target_variance = 0.0;
    for (left, right) in prediction_ranks.iter().zip(target_ranks) {
        let left_delta = *left - prediction_mean;
        let right_delta = right - target_mean;
        numerator += left_delta * right_delta;
        prediction_variance += left_delta * left_delta;
        target_variance += right_delta * right_delta;
    }
    let denominator = (prediction_variance * target_variance).sqrt();
    (denominator > 0.0).then_some(numerator / denominator)
}

pub fn ndcg_at_k(prediction: &[f32], target: &[f32], k: usize) -> Option<f32> {
    if prediction.len() != target.len() || prediction.is_empty() {
        return None;
    }
    let mut predicted_order: Vec<usize> = (0..prediction.len()).collect();
    predicted_order.sort_by(|left, right| prediction[*right].total_cmp(&prediction[*left]));
    let mut ideal_order: Vec<usize> = (0..target.len()).collect();
    ideal_order.sort_by(|left, right| target[*right].total_cmp(&target[*left]));
    let dcg = discounted_gain(&predicted_order, target, k);
    let ideal = discounted_gain(&ideal_order, target, k);
    (ideal > 0.0).then_some(dcg / ideal)
}

pub fn top_k_recall(prediction: &[f32], target: &[f32], k: usize) -> Option<f32> {
    if prediction.len() != target.len() || prediction.is_empty() {
        return None;
    }
    let mut predicted_order: Vec<usize> = (0..prediction.len()).collect();
    predicted_order.sort_by(|left, right| prediction[*right].total_cmp(&prediction[*left]));
    let mut target_order: Vec<usize> = (0..target.len()).collect();
    target_order.sort_by(|left, right| target[*right].total_cmp(&target[*left]));
    let predicted: std::collections::HashSet<usize> = predicted_order.into_iter().take(k).collect();
    let target: std::collections::HashSet<usize> = target_order.into_iter().take(k).collect();
    Some(predicted.intersection(&target).count() as f32 / target.len().max(1) as f32)
}

fn average_ranks(values: &[f32]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end - 1) as f32 / 2.0 + 1.0;
        for index in start..end {
            ranks[order[index]] = rank;
        }
        start = end;
    }
    ranks
}

fn discounted_gain(order: &[usize], target: &[f32], k: usize) -> f32 {
    order
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, index)| {
            let relevance = target[*index].max(0.0);
            (2.0_f32.powf(relevance) - 1.0) / ((rank + 2) as f32).log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use transit_domain::StopTime;
    use transit_router::{RouterConfig, RoutingData, RoutingPattern, RoutingTrip};

    fn trip() -> RoutingTrip {
        RoutingTrip {
            stop_times: vec![
                StopTime {
                    arrival: 100,
                    departure: 100,
                    pickup_type: 0,
                    dropoff_type: 0,
                },
                StopTime {
                    arrival: 200,
                    departure: 200,
                    pickup_type: 0,
                    dropoff_type: 0,
                },
            ],
        }
    }

    #[test]
    fn generates_one_aggregate_label_per_line() {
        let router = Router::new(
            RoutingData {
                station_count: 2,
                line_count: 1,
                patterns: vec![RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1)],
                    trips: vec![trip()],
                }],
                transfers: Vec::new(),
            },
            RouterConfig::default(),
        );
        let labels = generate_line_removal_labels(
            &router,
            "synthetic",
            &[StationIndex(0)],
            &[90],
            &LabelGenerationConfig {
                accessibility_thresholds_seconds: vec![900],
                maximum_origins: 1,
            },
        );
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].query_count, 1);
        assert!(labels[0].unreachable_share > 0.0);
        assert_eq!(labels[0].stations_losing_all_service_share, 1.0);
    }

    #[test]
    fn caps_and_deduplicates_sampled_origins() {
        let router = Router::new(
            RoutingData {
                station_count: 3,
                line_count: 1,
                patterns: vec![RoutingPattern {
                    line: LineIndex(0),
                    stops: vec![StationIndex(0), StationIndex(1)],
                    trips: vec![trip()],
                }],
                transfers: Vec::new(),
            },
            RouterConfig::default(),
        );
        let labels = generate_line_removal_labels(
            &router,
            "synthetic",
            &[
                StationIndex(0),
                StationIndex(0),
                StationIndex(1),
                StationIndex(2),
                StationIndex(99),
            ],
            &[90],
            &LabelGenerationConfig {
                maximum_origins: 2,
                ..LabelGenerationConfig::default()
            },
        );
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].query_count, 2);
    }

    #[test]
    fn station_service_loss_is_not_od_unreachability() {
        let router = Router::new(
            RoutingData {
                station_count: 3,
                line_count: 2,
                patterns: vec![
                    RoutingPattern {
                        line: LineIndex(0),
                        stops: vec![StationIndex(0), StationIndex(1)],
                        trips: vec![trip()],
                    },
                    RoutingPattern {
                        line: LineIndex(1),
                        stops: vec![StationIndex(1), StationIndex(2)],
                        trips: vec![trip()],
                    },
                ],
                transfers: Vec::new(),
            },
            RouterConfig::default(),
        );
        let labels = generate_line_removal_labels(
            &router,
            "synthetic",
            &[StationIndex(0)],
            &[90],
            &LabelGenerationConfig::default(),
        );
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].stations_losing_all_service_share, 1.0 / 3.0);
        assert!(labels[0].unreachable_share > labels[0].stations_losing_all_service_share);
    }

    #[test]
    fn ranking_metrics_reward_the_correct_order() {
        assert_eq!(
            spearman_rank(&[3.0, 2.0, 1.0], &[30.0, 20.0, 10.0]),
            Some(1.0)
        );
        assert_eq!(
            ndcg_at_k(&[3.0, 2.0, 1.0], &[30.0, 20.0, 10.0], 2),
            Some(1.0)
        );
        assert_eq!(
            top_k_recall(&[3.0, 2.0, 1.0], &[30.0, 20.0, 10.0], 2),
            Some(1.0)
        );
    }
}
