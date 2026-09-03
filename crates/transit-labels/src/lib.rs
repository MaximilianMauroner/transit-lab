//! Aggregate counterfactual labels generated from exact timetable routing.

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use transit_domain::{hex_digest, sha256_bytes, LineIndex, StationIndex, INF_TIME};
pub use transit_router::ROUTER_ALGORITHM_VERSION;
use transit_router::{OneToAllResult, Router, RouterConfig};

pub const ROUTING_BASELINE_SCHEMA_VERSION: &str = "routing-baseline-v2";
pub const LABEL_MANIFEST_SCHEMA_VERSION: &str = "line-impact-labels-v2";
pub const LABEL_BATCH_SCHEMA_VERSION: &str = "label-batch-v2";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelGenerationConfig {
    pub accessibility_thresholds_seconds: Vec<u32>,
    pub maximum_origins: usize,
    #[serde(default)]
    pub origin_sampling: OriginSamplingConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OriginSamplingConfig {
    pub seed: u64,
    pub geographic_fraction: f32,
    pub interchange_fraction: f32,
    pub uniform_fraction: f32,
}

impl Default for OriginSamplingConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            geographic_fraction: 0.5,
            interchange_fraction: 0.25,
            uniform_fraction: 0.25,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OriginCandidate {
    pub index: StationIndex,
    pub latitude: f64,
    pub longitude: f64,
    pub transfer_degree: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelManifest {
    pub schema_version: String,
    pub router_algorithm_version: String,
    pub policy_fingerprint: String,
    pub config: LabelGenerationConfig,
    pub origin_count: usize,
    #[serde(default)]
    pub snapshot_id: String,
    #[serde(default)]
    pub departure_times_seconds: Vec<u32>,
}

/// Durable progress for an append-only counterfactual label batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelBatchManifest {
    pub schema_version: String,
    pub router_algorithm_version: String,
    pub snapshot_id: String,
    pub policy_fingerprint: String,
    pub config: LabelGenerationConfig,
    pub origins: Vec<StationIndex>,
    pub departure_times_seconds: Vec<u32>,
    pub router_config: RouterConfig,
    pub baseline_fingerprint: String,
    pub line_count: usize,
    pub completed_lines: Vec<LineIndex>,
    pub status: String,
}

pub fn label_policy_fingerprint(config: &LabelGenerationConfig) -> String {
    let value = serde_json::json!({
        "router_algorithm_version": ROUTER_ALGORITHM_VERSION,
        "config": config,
    });
    let encoded = serde_json::to_vec(&value).expect("label configuration is serializable");
    hex_digest(&sha256_bytes(&encoded))
}

impl Default for LabelGenerationConfig {
    fn default() -> Self {
        Self {
            accessibility_thresholds_seconds: vec![15, 30, 45, 60, 90]
                .into_iter()
                .map(|minutes| minutes * 60)
                .collect(),
            maximum_origins: 256,
            origin_sampling: OriginSamplingConfig::default(),
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
    /// Identifies the routing semantics used to produce this row. A missing
    /// value is retained during deserialization only so validation can report
    /// that an old artifact is incompatible.
    #[serde(default)]
    pub router_algorithm_version: String,
    /// Identifies the exact label policy, including origin sampling, used to
    /// produce this row. Older JSONL files deserialize to an empty value.
    #[serde(default)]
    pub policy_fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingBaselineQuery {
    pub origin: StationIndex,
    pub departure: u32,
    pub result: OneToAllResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingBaseline {
    pub schema_version: String,
    pub router_algorithm_version: String,
    pub snapshot_id: String,
    pub origins: Vec<StationIndex>,
    pub departures: Vec<u32>,
    pub router_config: RouterConfig,
    pub queries: Vec<RoutingBaselineQuery>,
    pub fingerprint: String,
}

impl RoutingBaseline {
    pub fn validate(&self, router: &Router) -> Result<()> {
        if self.schema_version != ROUTING_BASELINE_SCHEMA_VERSION {
            anyhow::bail!(
                "unsupported routing baseline schema {}; expected {}",
                self.schema_version,
                ROUTING_BASELINE_SCHEMA_VERSION
            );
        }
        if self.router_algorithm_version != ROUTER_ALGORITHM_VERSION {
            anyhow::bail!(
                "routing baseline was generated with router {}; expected {}",
                self.router_algorithm_version,
                ROUTER_ALGORITHM_VERSION
            );
        }
        if self.router_config.maximum_transfers != router.config.maximum_transfers
            || self.router_config.maximum_journey_seconds != router.config.maximum_journey_seconds
        {
            anyhow::bail!("routing baseline was generated with a different router configuration");
        }
        if self.queries.len() != self.origins.len().saturating_mul(self.departures.len()) {
            anyhow::bail!("routing baseline query count does not match its origin/departure grid");
        }
        if self.queries.iter().any(|query| {
            query.result.arrival_seconds.len() != router.data.station_count
                || query.result.transfers.len() != router.data.station_count
        }) {
            anyhow::bail!("routing baseline contains a result with the wrong station count");
        }
        Ok(())
    }
}

pub fn build_routing_baseline(
    router: &Router,
    snapshot: impl Into<String>,
    origins: &[StationIndex],
    departures: &[u32],
) -> RoutingBaseline {
    let origins = select_origins(origins, origins.len(), router.data.station_count);
    let departures = departures.to_vec();
    let queries = origins
        .iter()
        .copied()
        .flat_map(|origin| {
            departures
                .iter()
                .copied()
                .map(move |departure| (origin, departure))
        })
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            || router.workspace(),
            |workspace, (origin, departure)| RoutingBaselineQuery {
                origin,
                departure,
                result: router.one_to_all_with_workspace(
                    origin,
                    departure,
                    &transit_domain::LineMask::empty(router.data.line_count),
                    workspace,
                ),
            },
        )
        .collect::<Vec<_>>();
    let snapshot_id = snapshot.into();
    let mut baseline = RoutingBaseline {
        schema_version: ROUTING_BASELINE_SCHEMA_VERSION.into(),
        router_algorithm_version: ROUTER_ALGORITHM_VERSION.into(),
        snapshot_id,
        origins,
        departures,
        router_config: router.config.clone(),
        queries,
        fingerprint: String::new(),
    };
    baseline.fingerprint = routing_baseline_fingerprint(&baseline);
    baseline
}

pub fn routing_baseline_fingerprint(baseline: &RoutingBaseline) -> String {
    let mut value = baseline.clone();
    value.fingerprint.clear();
    let bytes = serde_json::to_vec(&value).expect("routing baseline is serializable");
    hex_digest(&sha256_bytes(&bytes))
}

pub fn save_routing_baseline(path: &Path, baseline: &RoutingBaseline) -> Result<()> {
    baseline.validate_fingerprint()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let encoded = serde_json::to_vec_pretty(baseline).context("encoding routing baseline")?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, encoded)
        .with_context(|| format!("writing temporary baseline {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("committing routing baseline {}", path.display()))?;
    Ok(())
}

pub fn load_routing_baseline(path: &Path, router: &Router) -> Result<RoutingBaseline> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let baseline: RoutingBaseline = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding routing baseline {}", path.display()))?;
    baseline.validate(router)?;
    baseline.validate_fingerprint()?;
    Ok(baseline)
}

impl RoutingBaseline {
    fn validate_fingerprint(&self) -> Result<()> {
        if self.fingerprint != routing_baseline_fingerprint(self) {
            anyhow::bail!("routing baseline fingerprint does not match its contents");
        }
        Ok(())
    }
}

pub fn generate_line_removal_labels(
    router: &Router,
    snapshot: impl Into<String>,
    origins: &[StationIndex],
    departures: &[u32],
    config: &LabelGenerationConfig,
) -> Vec<LineImpactLabel> {
    let lines = (0..router.data.line_count)
        .map(|line| LineIndex(line as u32))
        .collect::<Vec<_>>();
    generate_selected_line_removal_labels(router, snapshot, origins, departures, config, &lines)
}

/// Generate labels only for the requested interventions. This keeps top-K
/// verification proportional to the number of lines being checked instead of
/// silently recomputing every line and filtering afterward.
pub fn generate_selected_line_removal_labels(
    router: &Router,
    snapshot: impl Into<String>,
    origins: &[StationIndex],
    departures: &[u32],
    config: &LabelGenerationConfig,
    selected_lines: &[LineIndex],
) -> Vec<LineImpactLabel> {
    let snapshot = snapshot.into();
    let origins = select_origins(origins, config.maximum_origins, router.data.station_count);
    if origins.is_empty() || departures.is_empty() || router.data.line_count == 0 {
        return Vec::new();
    }
    let mut lines = selected_lines
        .iter()
        .copied()
        .filter(|line| (line.0 as usize) < router.data.line_count)
        .collect::<Vec<_>>();
    lines.sort_unstable();
    lines.dedup();
    if lines.is_empty() {
        return Vec::new();
    }
    let baseline = build_routing_baseline(router, snapshot.clone(), &origins, departures);
    generate_selected_line_removal_labels_from_baseline(router, snapshot, config, &lines, &baseline)
}

/// Generate counterfactual rows from a previously materialized intact-network
/// baseline. This is the normal path for experiments: changing a line subset
/// no longer recomputes the expensive intact route queries.
pub fn generate_selected_line_removal_labels_from_baseline(
    router: &Router,
    snapshot: impl Into<String>,
    config: &LabelGenerationConfig,
    selected_lines: &[LineIndex],
    baseline: &RoutingBaseline,
) -> Vec<LineImpactLabel> {
    let snapshot = snapshot.into();
    if baseline.snapshot_id != snapshot
        || baseline.validate(router).is_err()
        || baseline.origins.is_empty()
        || baseline.departures.is_empty()
    {
        return Vec::new();
    }
    let mut lines = selected_lines
        .iter()
        .copied()
        .filter(|line| (line.0 as usize) < router.data.line_count)
        .collect::<Vec<_>>();
    lines.sort_unstable();
    lines.dedup();
    if lines.is_empty() {
        return Vec::new();
    }
    let policy_fingerprint = label_policy_fingerprint(config);
    lines
        .into_par_iter()
        .map_init(
            || router.workspace(),
            |workspace, line_index| {
                let stations_losing_all_service_share =
                    station_losing_all_service_share(router, line_index);
                let mut auc_loss = 0.0;
                let mut unreachable = 0_u64;
                let mut baseline_reachable = 0_u64;
                let mut delay_values = Vec::new();
                let mut extra_transfer_sum = 0.0_f64;
                let mut extra_transfer_count = 0_u64;

                for baseline in &baseline.queries {
                    let disrupted = router.one_to_all_with_workspace(
                        baseline.origin,
                        baseline.departure,
                        &transit_domain::LineMask::single(router.data.line_count, line_index),
                        workspace,
                    );
                    let destination_count = baseline.result.arrival_seconds.len().max(1) as f64;
                    for threshold in &config.accessibility_thresholds_seconds {
                        let intact = count_within(&baseline.result, baseline.departure, *threshold)
                            as f64
                            / destination_count;
                        let damaged = count_within(&disrupted, baseline.departure, *threshold)
                            as f64
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
                    accessibility_auc_loss: (auc_loss / baseline.queries.len().max(1) as f64)
                        as f32,
                    unreachable_share,
                    mean_delay_reachable_seconds: delay_sum / delay_values.len().max(1) as f32,
                    p95_delay_reachable_seconds: delay_values
                        .get(p95_index)
                        .copied()
                        .unwrap_or(0.0),
                    mean_extra_transfers: extra_transfer_sum as f32
                        / extra_transfer_count.max(1) as f32,
                    stations_losing_all_service_share,
                    query_count: baseline.queries.len() as u32,
                    router_algorithm_version: ROUTER_ALGORITHM_VERSION.into(),
                    policy_fingerprint: policy_fingerprint.clone(),
                }
            },
        )
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

/// Select a reproducible mix of geographically spread, high-interchange, and
/// uniform origins. The stable ordering uses coordinates and transfer degree,
/// not raw GTFS identifiers or compiled station indexes, so an ID-only feed
/// rewrite does not silently change the experiment.
pub fn sample_origins(
    candidates: &[OriginCandidate],
    maximum: usize,
    config: &OriginSamplingConfig,
) -> Vec<StationIndex> {
    if maximum == 0 || candidates.is_empty() {
        return Vec::new();
    }
    let mut candidates: Vec<OriginCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.latitude.is_finite() && candidate.longitude.is_finite())
        .cloned()
        .collect();
    candidates.sort_by_key(stable_candidate_key);
    candidates.dedup_by_key(|candidate| candidate.index);
    if candidates.len() <= maximum {
        return candidates
            .into_iter()
            .map(|candidate| candidate.index)
            .collect();
    }

    let geographic_count = ((maximum as f32 * config.geographic_fraction.clamp(0.0, 1.0)).round()
        as usize)
        .min(maximum);
    let interchange_count = ((maximum as f32 * config.interchange_fraction.clamp(0.0, 1.0)).round()
        as usize)
        .min(maximum.saturating_sub(geographic_count));
    let uniform_count = maximum
        .saturating_sub(geographic_count)
        .saturating_sub(interchange_count)
        .min(maximum);

    let mut selected = Vec::with_capacity(maximum);
    let mut selected_indices = std::collections::BTreeSet::new();
    let add = |candidate: &OriginCandidate,
               selected: &mut Vec<StationIndex>,
               selected_indices: &mut std::collections::BTreeSet<StationIndex>| {
        if selected.len() < maximum && selected_indices.insert(candidate.index) {
            selected.push(candidate.index);
        }
    };

    for candidate in geographic_order(&candidates, geographic_count, config.seed) {
        add(candidate, &mut selected, &mut selected_indices);
    }

    let mut hubs = candidates.iter().collect::<Vec<_>>();
    hubs.sort_by(|left, right| {
        right
            .transfer_degree
            .cmp(&left.transfer_degree)
            .then_with(|| stable_candidate_key(left).cmp(&stable_candidate_key(right)))
    });
    for candidate in hubs.into_iter().take(interchange_count) {
        add(candidate, &mut selected, &mut selected_indices);
    }

    let mut uniform = candidates.iter().collect::<Vec<_>>();
    uniform.sort_by_key(|candidate| seeded_key(config.seed, candidate));
    for candidate in uniform.into_iter().take(uniform_count) {
        add(candidate, &mut selected, &mut selected_indices);
    }

    // Fractions can overlap heavily on small or hub-dense networks. Fill the
    // remainder deterministically rather than returning fewer origins than
    // requested.
    for candidate in candidates.iter() {
        add(candidate, &mut selected, &mut selected_indices);
    }
    selected
}

fn geographic_order(
    candidates: &[OriginCandidate],
    maximum: usize,
    seed: u64,
) -> Vec<&OriginCandidate> {
    if maximum == 0 || candidates.is_empty() {
        return Vec::new();
    }
    let first = candidates
        .iter()
        .min_by_key(|candidate| seeded_key(seed ^ 0x9e37_79b9_7f4a_7c15, candidate))
        .expect("candidate list is not empty");
    let mut selected = vec![first];
    while selected.len() < maximum.min(candidates.len()) {
        let next = candidates
            .iter()
            .filter(|candidate| {
                !selected
                    .iter()
                    .any(|chosen| chosen.index == candidate.index)
            })
            .max_by(|left, right| {
                min_distance(left, &selected)
                    .total_cmp(&min_distance(right, &selected))
                    .then_with(|| stable_candidate_key(right).cmp(&stable_candidate_key(left)))
            });
        let Some(next) = next else { break };
        selected.push(next);
    }
    selected
}

fn min_distance(candidate: &OriginCandidate, selected: &[&OriginCandidate]) -> f64 {
    selected
        .iter()
        .map(|other| {
            let latitude_scale = ((candidate.latitude + other.latitude) * 0.5)
                .to_radians()
                .cos();
            let dx = (candidate.longitude - other.longitude) * latitude_scale;
            let dy = candidate.latitude - other.latitude;
            dx * dx + dy * dy
        })
        .fold(f64::INFINITY, f64::min)
}

fn stable_candidate_key(candidate: &OriginCandidate) -> (i64, i64, u32) {
    (
        (candidate.latitude * 1_000_000.0).round() as i64,
        (candidate.longitude * 1_000_000.0).round() as i64,
        candidate.transfer_degree,
    )
}

fn seeded_key(seed: u64, candidate: &OriginCandidate) -> u64 {
    let (latitude, longitude, degree) = stable_candidate_key(candidate);
    let mut value = seed
        ^ (latitude as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (longitude as u64).rotate_left(17)
        ^ u64::from(degree).rotate_left(31);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
    for (index, label) in labels.iter().enumerate() {
        validate_label_identity(label, &format!("label row {}", index + 1))?;
    }
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

pub fn save_label_manifest(
    path: &Path,
    config: &LabelGenerationConfig,
    origin_count: usize,
) -> Result<()> {
    save_label_manifest_with_metadata(path, config, origin_count, "", &[])
}

/// Save a label manifest with the immutable inputs that make a label batch
/// reusable. The three-argument function above remains as a compatibility
/// wrapper for older callers and fixtures.
pub fn save_label_manifest_with_metadata(
    path: &Path,
    config: &LabelGenerationConfig,
    origin_count: usize,
    snapshot_id: impl Into<String>,
    departure_times_seconds: &[u32],
) -> Result<()> {
    let manifest_path = label_manifest_path(path);
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let manifest = LabelManifest {
        schema_version: LABEL_MANIFEST_SCHEMA_VERSION.into(),
        router_algorithm_version: ROUTER_ALGORITHM_VERSION.into(),
        policy_fingerprint: label_policy_fingerprint(config),
        config: config.clone(),
        origin_count,
        snapshot_id: snapshot_id.into(),
        departure_times_seconds: departure_times_seconds.to_vec(),
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("writing {}", manifest_path.display()))?;
    Ok(())
}

fn label_manifest_path(path: &Path) -> std::path::PathBuf {
    let mut manifest_path = path.to_path_buf();
    manifest_path.set_extension("manifest.json");
    manifest_path
}

fn label_batch_manifest_path(path: &Path) -> std::path::PathBuf {
    let mut manifest_path = path.to_path_buf();
    manifest_path.set_extension("batch.json");
    manifest_path
}

pub fn save_label_batch_manifest(path: &Path, manifest: &LabelBatchManifest) -> Result<()> {
    if manifest.schema_version != LABEL_BATCH_SCHEMA_VERSION {
        anyhow::bail!("unsupported label batch schema {}", manifest.schema_version);
    }
    if manifest.router_algorithm_version != ROUTER_ALGORITHM_VERSION {
        anyhow::bail!(
            "label batch was generated with router {}; expected {}",
            manifest.router_algorithm_version,
            ROUTER_ALGORITHM_VERSION
        );
    }
    if manifest.snapshot_id.trim().is_empty()
        || manifest.baseline_fingerprint.trim().is_empty()
        || manifest.line_count == 0
    {
        anyhow::bail!("label batch manifest is missing required identity fields");
    }
    if manifest
        .completed_lines
        .iter()
        .any(|line| line.0 as usize >= manifest.line_count)
    {
        anyhow::bail!("label batch contains an out-of-range completed line");
    }
    let manifest_path = label_batch_manifest_path(path);
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let temporary = manifest_path.with_extension(format!("manifest.tmp-{}", std::process::id()));
    let encoded = serde_json::to_vec_pretty(manifest).context("encoding label batch manifest")?;
    {
        let mut file = File::create(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        file.write_all(&encoded)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temporary, &manifest_path).with_context(|| {
        format!(
            "committing label batch manifest {}",
            manifest_path.display()
        )
    })?;
    Ok(())
}

pub fn load_label_batch_manifest(path: &Path) -> Result<LabelBatchManifest> {
    let manifest_path = label_batch_manifest_path(path);
    let bytes = fs::read(&manifest_path)
        .with_context(|| format!("reading label batch manifest {}", manifest_path.display()))?;
    let manifest: LabelBatchManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding label batch manifest {}", manifest_path.display()))?;
    if manifest.schema_version != LABEL_BATCH_SCHEMA_VERSION {
        anyhow::bail!(
            "unsupported label batch schema {}; expected {}",
            manifest.schema_version,
            LABEL_BATCH_SCHEMA_VERSION
        );
    }
    if manifest.router_algorithm_version != ROUTER_ALGORITHM_VERSION {
        anyhow::bail!(
            "label batch was generated with router {}; expected {}",
            manifest.router_algorithm_version,
            ROUTER_ALGORITHM_VERSION
        );
    }
    Ok(manifest)
}

/// Generate a counterfactual batch incrementally. Existing rows and the
/// committed cursor are validated before work starts, and each new row is
/// flushed before its cursor is atomically committed.
pub fn generate_line_removal_labels_resumable(
    router: &Router,
    snapshot: impl Into<String>,
    config: &LabelGenerationConfig,
    selected_lines: &[LineIndex],
    baseline: &RoutingBaseline,
    output: &Path,
) -> Result<Vec<LineImpactLabel>> {
    let snapshot = snapshot.into();
    baseline.validate(router)?;
    if baseline.snapshot_id != snapshot {
        anyhow::bail!("routing baseline snapshot does not match label snapshot");
    }
    let mut lines = selected_lines
        .iter()
        .copied()
        .filter(|line| (line.0 as usize) < router.data.line_count)
        .collect::<Vec<_>>();
    lines.sort_unstable();
    lines.dedup();
    if lines.is_empty() {
        return Ok(Vec::new());
    }
    let policy_fingerprint = label_policy_fingerprint(config);
    let baseline_fingerprint = baseline.fingerprint.clone();
    let manifest_exists = label_batch_manifest_path(output).exists();
    let mut manifest = if manifest_exists {
        load_label_batch_manifest(output)?
    } else {
        new_label_batch_manifest(snapshot.clone(), config, baseline, router.data.line_count)
    };
    if manifest.snapshot_id != snapshot
        || manifest.router_algorithm_version != ROUTER_ALGORITHM_VERSION
        || manifest.policy_fingerprint != policy_fingerprint
        || manifest.baseline_fingerprint != baseline_fingerprint
        || manifest.origins != baseline.origins
        || manifest.departure_times_seconds != baseline.departures
        || manifest.router_config.maximum_transfers != router.config.maximum_transfers
        || manifest.router_config.maximum_journey_seconds != router.config.maximum_journey_seconds
        || manifest.line_count != router.data.line_count
    {
        anyhow::bail!("existing label batch is incompatible with requested inputs");
    }
    let mut rows = if output.exists() {
        load_jsonl(output)
            .with_context(|| format!("loading partial labels {}", output.display()))?
    } else {
        Vec::new()
    };
    validate_partial_labels(
        &rows,
        &snapshot,
        &policy_fingerprint,
        router.data.line_count,
    )?;
    let mut completed = rows
        .iter()
        .map(|label| label.line)
        .collect::<std::collections::BTreeSet<_>>();
    for line in &manifest.completed_lines {
        if !completed.contains(line) {
            anyhow::bail!("label batch manifest claims missing line {}", line.0);
        }
    }
    manifest.completed_lines = completed.iter().copied().collect();
    manifest.status = if lines.iter().all(|line| completed.contains(line)) {
        "committed".into()
    } else {
        "in-progress".into()
    };
    save_label_batch_manifest(output, &manifest)?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output)
        .with_context(|| format!("opening label output {}", output.display()))?;
    for line in lines.iter().copied() {
        if completed.contains(&line) {
            continue;
        }
        let Some(label) = generate_selected_line_removal_labels_from_baseline(
            router,
            snapshot.clone(),
            config,
            &[line],
            baseline,
        )
        .into_iter()
        .next() else {
            continue;
        };
        serde_json::to_writer(&mut file, &label).context("encoding resumed line label")?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        rows.push(label);
        completed.insert(line);
        manifest.completed_lines = completed.iter().copied().collect();
        manifest.status = if lines.iter().all(|candidate| completed.contains(candidate)) {
            "committed".into()
        } else {
            "in-progress".into()
        };
        save_label_batch_manifest(output, &manifest)?;
    }
    rows.sort_by_key(|label| label.line);
    Ok(rows)
}

fn new_label_batch_manifest(
    snapshot: String,
    config: &LabelGenerationConfig,
    baseline: &RoutingBaseline,
    line_count: usize,
) -> LabelBatchManifest {
    LabelBatchManifest {
        schema_version: LABEL_BATCH_SCHEMA_VERSION.into(),
        router_algorithm_version: ROUTER_ALGORITHM_VERSION.into(),
        snapshot_id: snapshot,
        policy_fingerprint: label_policy_fingerprint(config),
        config: config.clone(),
        origins: baseline.origins.clone(),
        departure_times_seconds: baseline.departures.clone(),
        router_config: baseline.router_config.clone(),
        baseline_fingerprint: baseline.fingerprint.clone(),
        line_count,
        completed_lines: Vec::new(),
        status: "in-progress".into(),
    }
}

fn validate_partial_labels(
    rows: &[LineImpactLabel],
    snapshot: &str,
    policy_fingerprint: &str,
    line_count: usize,
) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        if row.snapshot != snapshot {
            anyhow::bail!("partial label output contains a different snapshot");
        }
        if row.router_algorithm_version != ROUTER_ALGORITHM_VERSION {
            anyhow::bail!(
                "partial label output was generated with router {}; expected {}",
                row.router_algorithm_version,
                ROUTER_ALGORITHM_VERSION
            );
        }
        if row.policy_fingerprint.is_empty() {
            anyhow::bail!("partial label output is missing its label policy fingerprint");
        }
        if row.policy_fingerprint != policy_fingerprint {
            anyhow::bail!("partial label output contains a different label policy");
        }
        if row.line.0 as usize >= line_count || !seen.insert(row.line) {
            anyhow::bail!("partial label output contains a duplicate or invalid line");
        }
    }
    Ok(())
}

fn validate_label_identity(label: &LineImpactLabel, field: &str) -> Result<()> {
    if label.router_algorithm_version != ROUTER_ALGORITHM_VERSION {
        anyhow::bail!(
            "{field} was generated with router {}; expected {}",
            label.router_algorithm_version,
            ROUTER_ALGORITHM_VERSION
        );
    }
    if label.policy_fingerprint.trim().is_empty() {
        anyhow::bail!("{field} is missing its label policy fingerprint");
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
        let label: LineImpactLabel = serde_json::from_str(&line)
            .with_context(|| format!("decoding label line {}", line_number + 1))?;
        validate_label_identity(&label, &format!("label line {}", line_number + 1))?;
        labels.push(label);
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
    use std::fs;
    use tempfile::tempdir;
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
                ..LabelGenerationConfig::default()
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
    fn origin_sampling_is_seeded_and_coordinate_stable() {
        let candidates = vec![
            OriginCandidate {
                index: StationIndex(10),
                latitude: 48.10,
                longitude: 16.10,
                transfer_degree: 0,
            },
            OriginCandidate {
                index: StationIndex(11),
                latitude: 48.20,
                longitude: 16.10,
                transfer_degree: 4,
            },
            OriginCandidate {
                index: StationIndex(12),
                latitude: 48.10,
                longitude: 16.30,
                transfer_degree: 2,
            },
            OriginCandidate {
                index: StationIndex(13),
                latitude: 48.30,
                longitude: 16.30,
                transfer_degree: 0,
            },
            OriginCandidate {
                index: StationIndex(14),
                latitude: 48.25,
                longitude: 16.20,
                transfer_degree: 1,
            },
        ];
        let config = OriginSamplingConfig::default();
        let first = sample_origins(&candidates, 3, &config);
        let second = sample_origins(&candidates, 3, &config);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert!(first.contains(&StationIndex(11)));

        let renamed_indexes = candidates
            .iter()
            .map(|candidate| OriginCandidate {
                index: StationIndex(candidate.index.0 + 100),
                ..candidate.clone()
            })
            .collect::<Vec<_>>();
        let renamed = sample_origins(&renamed_indexes, 3, &config);
        let coordinate_keys = |indexes: &[StationIndex], pool: &[OriginCandidate]| {
            indexes
                .iter()
                .filter_map(|index| pool.iter().find(|candidate| candidate.index == *index))
                .map(stable_candidate_key)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            coordinate_keys(&first, &candidates),
            coordinate_keys(&renamed, &renamed_indexes)
        );
    }

    #[test]
    fn label_rows_carry_the_policy_fingerprint() {
        let config = LabelGenerationConfig::default();
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
        let labels =
            generate_line_removal_labels(&router, "snapshot", &[StationIndex(0)], &[90], &config);
        assert_eq!(
            labels[0].policy_fingerprint,
            label_policy_fingerprint(&config)
        );
        assert_eq!(labels[0].router_algorithm_version, ROUTER_ALGORITHM_VERSION);
    }

    #[test]
    fn persisted_routing_artifacts_carry_router_version_metadata() {
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
        let directory = tempdir().unwrap();
        let baseline = build_routing_baseline(&router, "snapshot", &[StationIndex(0)], &[90]);
        let baseline_path = directory.path().join("routing-baseline.json");
        save_routing_baseline(&baseline_path, &baseline).unwrap();
        let baseline_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&baseline_path).unwrap()).unwrap();
        assert_eq!(
            baseline_json["router_algorithm_version"],
            ROUTER_ALGORITHM_VERSION
        );

        let config = LabelGenerationConfig::default();
        let labels = generate_selected_line_removal_labels_from_baseline(
            &router,
            "snapshot",
            &config,
            &[LineIndex(0)],
            &baseline,
        );
        let labels_path = directory.path().join("labels.jsonl");
        save_jsonl(&labels_path, &labels).unwrap();
        let row: serde_json::Value = serde_json::from_str(
            fs::read_to_string(&labels_path)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(row["router_algorithm_version"], ROUTER_ALGORITHM_VERSION);

        save_label_manifest_with_metadata(
            &labels_path,
            &config,
            baseline.origins.len(),
            "snapshot",
            &baseline.departures,
        )
        .unwrap();
        let manifest: LabelManifest = serde_json::from_slice(
            &fs::read(directory.path().join("labels.manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.schema_version, LABEL_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.router_algorithm_version, ROUTER_ALGORITHM_VERSION);
        assert_eq!(
            manifest.policy_fingerprint,
            label_policy_fingerprint(&config)
        );

        let batch = LabelBatchManifest {
            schema_version: LABEL_BATCH_SCHEMA_VERSION.into(),
            router_algorithm_version: ROUTER_ALGORITHM_VERSION.into(),
            snapshot_id: "snapshot".into(),
            policy_fingerprint: label_policy_fingerprint(&config),
            config,
            origins: baseline.origins.clone(),
            departure_times_seconds: baseline.departures.clone(),
            router_config: baseline.router_config.clone(),
            baseline_fingerprint: baseline.fingerprint.clone(),
            line_count: router.data.line_count,
            completed_lines: vec![LineIndex(0)],
            status: "committed".into(),
        };
        save_label_batch_manifest(&labels_path, &batch).unwrap();
        assert_eq!(
            load_label_batch_manifest(&labels_path)
                .unwrap()
                .router_algorithm_version,
            ROUTER_ALGORITHM_VERSION
        );
    }

    #[test]
    fn legacy_label_rows_fail_identity_validation() {
        let legacy: LineImpactLabel = serde_json::from_value(serde_json::json!({
            "snapshot": "snapshot",
            "line": 0,
            "accessibility_auc_loss": 0.0,
            "unreachable_share": 0.0,
            "mean_delay_reachable_seconds": 0.0,
            "p95_delay_reachable_seconds": 0.0,
            "mean_extra_transfers": 0.0,
            "stations_losing_all_service_share": 0.0,
            "query_count": 1
        }))
        .unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("legacy-labels.jsonl");
        fs::write(&path, serde_json::to_string(&legacy).unwrap() + "\n").unwrap();
        let error = load_jsonl(&path).unwrap_err();
        assert!(error.to_string().contains("router"));
    }

    #[test]
    fn selected_label_generation_does_not_run_unselected_lines() {
        let router = Router::new(
            RoutingData {
                station_count: 2,
                line_count: 2,
                patterns: vec![
                    RoutingPattern {
                        line: LineIndex(0),
                        stops: vec![StationIndex(0), StationIndex(1)],
                        trips: vec![trip()],
                    },
                    RoutingPattern {
                        line: LineIndex(1),
                        stops: vec![StationIndex(0), StationIndex(1)],
                        trips: vec![trip()],
                    },
                ],
                transfers: Vec::new(),
            },
            RouterConfig::default(),
        );
        let labels = generate_selected_line_removal_labels(
            &router,
            "snapshot",
            &[StationIndex(0)],
            &[90],
            &LabelGenerationConfig {
                maximum_origins: 1,
                ..LabelGenerationConfig::default()
            },
            &[LineIndex(1)],
        );
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].line, LineIndex(1));
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
