//! Resumable LibTorch training sessions.
//!
//! `tch::nn::Optimizer` deliberately is not used here.  The wrapper in
//! `tch` 0.18 owns a private C++ optimizer state dictionary and cannot save it
//! through the public Rust API.  [`transit_model::tch_backend::TchOptimizer`]
//! owns Adam/AdamW moments in Rust, which gives this session a complete,
//! verifiable split-resume boundary.

use crate::checkpoint::{
    BestMetricState, CheckpointFile, CheckpointStatus, RngState, SamplerState, ScalerState,
    SchedulerState, TrainingCheckpointManifest, TrainingCursor,
};
use crate::control::{ControlDirective, TrainingControl};
use crate::runtime::RuntimeConfig;
use crate::session::{CheckpointMetadata, CheckpointPolicy};
use crate::{MaskSelection, NoopTrainingObserver, PretrainingConfig, TrainingObserver};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tch::{Device, Kind, Reduction, Tensor};
use transit_graph::GraphTensor;
use transit_labels::LineImpactLabel;
use transit_model::tch_backend::{
    PreparedGraph, TchEmbeddings, TchModelArtifactMetadata, TchOptimizer, TchOptimizerMetadata,
    TchRelationalAutoencoder, TchTaskOutputs,
};
use transit_model::CRITICALITY_OUTPUTS;

pub const TCH_CHECKPOINT_BACKEND: &str = "libtorch-adam-pretraining";
pub const TCH_MULTITASK_CHECKPOINT_BACKEND: &str = "libtorch-adam-multitask";
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static THREAD_CONFIGURATION: OnceLock<(usize, usize)> = OnceLock::new();

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TchTrainingReport {
    pub backend: String,
    pub steps: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
}

/// Phase-local progress persisted in a LibTorch multi-task checkpoint. The
/// graph examples and triplet plan are immutable dataset-derived state and
/// are rebuilt from the same graph fingerprints on resume; these fields keep
/// the human-facing report and phase cursor stable across process attempts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TchMultiTaskPhaseState {
    #[serde(rename = "pretrainingReport", default)]
    pub pretraining_report: Option<TchTrainingReport>,
    #[serde(rename = "metricInitialLoss", default)]
    pub metric_initial_loss: Option<f64>,
    #[serde(rename = "metricFinalLoss", default)]
    pub metric_final_loss: Option<f64>,
    #[serde(rename = "metricTriplets", default)]
    pub metric_triplets: usize,
    #[serde(rename = "criticalityReport", default)]
    pub criticality_report: Option<TchTrainingReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TchCheckpointState {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "runId")]
    pub run_id: String,
    #[serde(rename = "attemptId")]
    pub attempt_id: Option<String>,
    pub optimizer: TchOptimizerMetadata,
    pub scheduler: SchedulerState,
    pub scaler: ScalerState,
    pub rng: RngState,
    pub cursor: TrainingCursor,
    pub sampler: SamplerState,
    #[serde(rename = "bestMetrics")]
    pub best_metrics: BestMetricState,
    #[serde(rename = "datasetFingerprint")]
    pub dataset_fingerprint: String,
    #[serde(rename = "configFingerprint")]
    pub config_fingerprint: String,
    #[serde(rename = "codeCommit")]
    pub code_commit: String,
    pub backend: String,
    #[serde(rename = "backendVersion")]
    pub backend_version: String,
    #[serde(rename = "deviceType")]
    pub device_type: String,
    pub report: TchTrainingReport,
    #[serde(rename = "multiTaskPhase", default)]
    pub multi_task_phase: Option<TchMultiTaskPhaseState>,
}

impl TchCheckpointState {
    fn validate(&self) -> Result<()> {
        if self.schema_version != crate::TRAINING_CHECKPOINT_SCHEMA_VERSION {
            bail!(
                "unsupported LibTorch checkpoint schema {}; expected {}",
                self.schema_version,
                crate::TRAINING_CHECKPOINT_SCHEMA_VERSION
            );
        }
        if self.run_id.trim().is_empty()
            || self.dataset_fingerprint.trim().is_empty()
            || self.config_fingerprint.trim().is_empty()
        {
            bail!("LibTorch checkpoint identity fields cannot be blank");
        }
        if self.backend != TCH_CHECKPOINT_BACKEND
            && self.backend != TCH_MULTITASK_CHECKPOINT_BACKEND
        {
            bail!("unsupported LibTorch checkpoint backend {}", self.backend);
        }
        self.cursor.validate()?;
        if self.cursor.global_step != self.optimizer.step {
            bail!(
                "LibTorch checkpoint cursor step {} does not match optimizer step {}",
                self.cursor.global_step,
                self.optimizer.step
            );
        }
        if self.report.steps != self.cursor.global_step as usize {
            bail!(
                "LibTorch checkpoint report steps {} do not match cursor step {}",
                self.report.steps,
                self.cursor.global_step
            );
        }
        if self.backend == TCH_MULTITASK_CHECKPOINT_BACKEND && self.multi_task_phase.is_none() {
            bail!("LibTorch multi-task checkpoint is missing phase state");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TchCheckpointLoad {
    pub directory: PathBuf,
    pub state: TchCheckpointState,
    pub manifest: TrainingCheckpointManifest,
}

#[derive(Clone, Debug)]
pub enum TchTrainingOutcome {
    Completed { checkpoint_path: PathBuf },
    Paused { checkpoint_path: PathBuf },
    TimeSliceExpired { checkpoint_path: PathBuf },
    Cancelled,
}

/// One process-local LibTorch session.  The prepared graph tensors are kept
/// alive for the whole attempt; model weights and optimizer moments are the
/// only state that crosses process boundaries.
pub struct TchTrainingSession {
    pub model: TchRelationalAutoencoder,
    pub optimizer: TchOptimizer,
    pub cursor: TrainingCursor,
    pub scheduler: SchedulerState,
    pub scaler: ScalerState,
    pub rng: RngState,
    pub sampler: SamplerState,
    pub best_metrics: BestMetricState,
    pub report: TchTrainingReport,
    prepared: Vec<PreparedGraph>,
}

impl std::fmt::Debug for TchTrainingSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TchTrainingSession")
            .field("cursor", &self.cursor)
            .field("scheduler", &self.scheduler)
            .field("sampler", &self.sampler)
            .field("report", &self.report)
            .field("prepared_graphs", &self.prepared.len())
            .finish_non_exhaustive()
    }
}

impl TchTrainingSession {
    pub fn new(graphs: &[&GraphTensor], config: &PretrainingConfig) -> Result<Self> {
        let first = graphs.first().context("no graph datasets were provided")?;
        validate_graphs(graphs, first)?;
        configure_runtime(&config.runtime)?;
        let device = tch_device(&config.runtime.device)?;
        let model = TchRelationalAutoencoder::new(device, first, &config.model);
        let optimizer = TchOptimizer::adam(
            &model.var_store,
            f64::from(config.learning_rate),
            f64::from(config.weight_decay),
        )?;
        let prepared = graphs
            .iter()
            .map(|graph| model.prepare_graph(graph))
            .collect::<Result<Vec<_>>>()?;
        let graph_order = graphs
            .iter()
            .map(|graph| graph.manifest.snapshot_id.clone())
            .collect::<Vec<_>>();
        let mut sampler = SamplerState {
            seed: config.seed,
            graph_order,
            ..SamplerState::default()
        };
        sampler.current_graph = 0;
        Ok(Self {
            model,
            optimizer,
            cursor: TrainingCursor::default(),
            scheduler: SchedulerState {
                last_learning_rate: f64::from(config.learning_rate),
                ..SchedulerState::default()
            },
            scaler: ScalerState::default(),
            rng: RngState {
                cpu_seed: config.seed,
                gpu_seed: match config.runtime.device {
                    crate::runtime::DeviceKind::Cuda { index } => Some(config.seed ^ index as u64),
                    crate::runtime::DeviceKind::Cpu => None,
                },
                ..RngState::default()
            },
            sampler,
            best_metrics: BestMetricState::default(),
            report: TchTrainingReport {
                backend: TCH_CHECKPOINT_BACKEND.into(),
                steps: 0,
                initial_loss: 0.0,
                final_loss: 0.0,
            },
            prepared,
        })
    }

    pub fn resume(
        graphs: &[&GraphTensor],
        config: &PretrainingConfig,
        checkpoint: &Path,
        metadata: &CheckpointMetadata,
    ) -> Result<Self> {
        let loaded = load_tch_checkpoint(checkpoint)?;
        validate_state_identity(&loaded.state, metadata)?;
        let mut session = Self::new(graphs, config)?;
        if loaded.state.sampler.graph_order != session.sampler.graph_order {
            bail!("LibTorch checkpoint graph order does not match the requested dataset");
        }
        if loaded.state.cursor.phase != "pretraining" {
            bail!(
                "LibTorch pretraining cannot resume phase {}",
                loaded.state.cursor.phase
            );
        }
        if loaded.state.cursor.global_step > config.steps as u64 {
            bail!(
                "LibTorch checkpoint step {} exceeds configured steps {}",
                loaded.state.cursor.global_step,
                config.steps
            );
        }
        session
            .model
            .load_weights(&loaded.directory.join("model.ot"))?;
        session.optimizer.load_state(
            &loaded.directory.join("optimizer.ot"),
            &loaded.state.optimizer,
        )?;
        session.cursor = loaded.state.cursor;
        session.scheduler = loaded.state.scheduler;
        session.scaler = loaded.state.scaler;
        session.rng = loaded.state.rng;
        session.sampler = loaded.state.sampler;
        session.best_metrics = loaded.state.best_metrics;
        session.report = loaded.state.report;
        Ok(session)
    }

    /// Forking restores model weights but intentionally starts a new optimizer,
    /// sampler, and training cursor under the new run identity.
    pub fn fork(
        graphs: &[&GraphTensor],
        config: &PretrainingConfig,
        checkpoint: &Path,
    ) -> Result<Self> {
        let loaded = load_tch_checkpoint(checkpoint)?;
        let mut session = Self::new(graphs, config)?;
        session
            .model
            .load_weights(&loaded.directory.join("model.ot"))?;
        Ok(session)
    }

    fn state(&self, metadata: &CheckpointMetadata) -> TchCheckpointState {
        TchCheckpointState {
            schema_version: crate::TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: metadata.run_id.clone(),
            attempt_id: metadata.attempt_id.clone(),
            optimizer: self.optimizer.metadata(),
            scheduler: self.scheduler.clone(),
            scaler: self.scaler.clone(),
            rng: self.rng.clone(),
            cursor: self.cursor.clone(),
            sampler: self.sampler.clone(),
            best_metrics: self.best_metrics.clone(),
            dataset_fingerprint: metadata.dataset_fingerprint.clone(),
            config_fingerprint: metadata.config_fingerprint.clone(),
            code_commit: metadata.code_commit.clone(),
            backend: TCH_CHECKPOINT_BACKEND.into(),
            backend_version: metadata.backend_version.clone(),
            device_type: metadata.device_type.clone(),
            report: self.report.clone(),
            multi_task_phase: None,
        }
    }

    fn accumulation_cycle(
        &mut self,
        graphs: &[&GraphTensor],
        config: &PretrainingConfig,
    ) -> Result<f64> {
        let accumulation = config.runtime.gradient_accumulation.max(1);
        self.optimizer.zero_grad();
        let mut total_loss = 0.0_f64;
        for _ in 0..accumulation {
            let graph_index = self.sampler.current_graph % graphs.len();
            let graph = graphs[graph_index];
            let prepared = &self.prepared[graph_index];
            let mask = MaskSelection::sample(
                graph,
                &config.mask,
                config
                    .seed
                    .wrapping_add(self.cursor.examples_seen)
                    .wrapping_add(graph_index as u64 * 7_919),
            );
            let reconstruction = self
                .model
                .forward_prepared_unchecked(graph, prepared, &mask, true)?;
            let station_loss = masked_mse(
                &reconstruction.station_features,
                &prepared.station_features,
                &mask.station_rows,
            );
            let line_loss = masked_mse(
                &reconstruction.line_features,
                &prepared.line_features,
                &mask.line_rows,
            );
            let loss = station_loss + line_loss;
            total_loss += loss.double_value(&[]);
            (loss / accumulation as f64).backward();
            self.sampler.current_graph = (graph_index + 1) % graphs.len();
            self.sampler.current_example = self.sampler.current_example.saturating_add(1);
            self.cursor.examples_seen = self
                .cursor
                .examples_seen
                .saturating_add((graph.manifest.station_count + graph.manifest.line_count) as u64);
        }
        self.optimizer.step_gradients()?;
        Ok(total_loss / accumulation as f64)
    }

    fn checkpoint(
        &self,
        root: &Path,
        metadata: &CheckpointMetadata,
        observer: &mut dyn TrainingObserver,
    ) -> Result<PathBuf> {
        observer.checkpoint_started("pretraining", self.cursor.global_step as usize);
        let path = save_tch_checkpoint(root, self, metadata)?;
        observer.checkpoint_committed("pretraining", self.cursor.global_step as usize, &path);
        Ok(path)
    }
}

pub fn run_tch_pretraining_with_policy_options(
    graphs: &[&GraphTensor],
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    allow_fork: bool,
    observer: &mut dyn TrainingObserver,
) -> Result<(TchTrainingSession, TchTrainingOutcome)> {
    config.runtime.validate()?;
    if graphs.is_empty() {
        bail!("no graph datasets were provided");
    }
    let mut session = match resume {
        Some(path) if allow_fork => TchTrainingSession::fork(graphs, config, path)?,
        Some(path) => TchTrainingSession::resume(graphs, config, path, metadata)?,
        None => TchTrainingSession::new(graphs, config)?,
    };
    let mut last_checkpoint_path = resume.map(Path::to_path_buf);
    let mut last_saved_step = session.cursor.global_step;
    let mut last_checkpoint_at = Instant::now();
    observer.phase_started("pretraining", Some(config.steps));
    observer.learning_rate_changed(
        "pretraining",
        session.cursor.global_step as usize,
        config.learning_rate,
    );

    while session.cursor.global_step < config.steps as u64 {
        observer.epoch_started(
            "pretraining",
            session.cursor.global_step as usize + 1,
            config.steps,
        );
        let loss = session.accumulation_cycle(graphs, config)?;
        let step = session.cursor.global_step;
        if step == 0 {
            session.report.initial_loss = loss;
        }
        session.report.final_loss = loss;
        session.cursor.global_step = step.saturating_add(1);
        session.cursor.epoch = session.cursor.global_step;
        session.cursor.batch = session.cursor.global_step;
        session.report.steps = session.cursor.global_step as usize;
        session.scheduler.last_step = session.cursor.global_step;
        session.scheduler.last_learning_rate = session.optimizer.learning_rate();
        session
            .best_metrics
            .values
            .insert("reconstruction_loss".into(), loss);
        session.best_metrics.steps_without_improvement = 0;
        observer.metric(
            "pretraining",
            session.cursor.global_step as usize,
            session.cursor.global_step as usize,
            "reconstruction_loss",
            loss as f32,
        );
        if session.cursor.global_step % 10 == 0 || session.cursor.global_step == config.steps as u64
        {
            observer.heartbeat("pretraining", session.cursor.global_step as usize);
        }

        let directive = control.directive()?;
        let force = session.cursor.global_step == config.steps as u64;
        let due = force
            || matches!(
                directive,
                ControlDirective::Checkpoint | ControlDirective::Pause
            )
            || checkpoint_policy
                .every_steps
                .filter(|value| *value > 0)
                .is_some_and(|value| session.cursor.global_step % value as u64 == 0)
            || checkpoint_policy
                .every_seconds
                .filter(|value| *value > 0)
                .is_some_and(|value| last_checkpoint_at.elapsed() >= Duration::from_secs(value));
        if due && session.cursor.global_step > last_saved_step {
            let path = session.checkpoint(checkpoint_root, metadata, observer)?;
            last_checkpoint_path = Some(path);
            last_saved_step = session.cursor.global_step;
            last_checkpoint_at = Instant::now();
        }
        if matches!(directive, ControlDirective::Cancel) {
            observer.phase_completed("pretraining");
            return Ok((session, TchTrainingOutcome::Cancelled));
        }
        if matches!(directive, ControlDirective::Pause)
            || (matches!(directive, ControlDirective::Checkpoint)
                && control.deadline_expired()
                && !force)
        {
            let path = last_checkpoint_path
                .clone()
                .context("pause requested before a LibTorch checkpoint was committed")?;
            observer.phase_completed("pretraining");
            let outcome = if control.deadline_expired() {
                TchTrainingOutcome::TimeSliceExpired {
                    checkpoint_path: path,
                }
            } else {
                TchTrainingOutcome::Paused {
                    checkpoint_path: path,
                }
            };
            return Ok((session, outcome));
        }
    }
    observer.phase_completed("pretraining");
    let checkpoint_path = match last_checkpoint_path {
        Some(path) if last_saved_step == session.cursor.global_step => path,
        _ => session.checkpoint(checkpoint_root, metadata, observer)?,
    };
    Ok((session, TchTrainingOutcome::Completed { checkpoint_path }))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TchMultiTaskTrainingReport {
    pub backend: String,
    pub dataset_count: usize,
    pub line_count: usize,
    pub pretraining: TchTrainingReport,
    #[serde(rename = "metricEpochs")]
    pub metric_epochs: usize,
    #[serde(rename = "metricInitialLoss")]
    pub metric_initial_loss: f64,
    #[serde(rename = "metricFinalLoss")]
    pub metric_final_loss: f64,
    #[serde(rename = "metricTriplets")]
    pub metric_triplets: usize,
    pub criticality: Option<TchTrainingReport>,
}

#[derive(Clone, Copy, Debug)]
enum TchMetricFacet {
    Base,
    General,
    Role,
    Service,
    Geometry,
    Resilience,
}

impl TchMetricFacet {
    const ALL: [Self; 6] = [
        Self::Base,
        Self::General,
        Self::Role,
        Self::Service,
        Self::Geometry,
        Self::Resilience,
    ];
}

#[derive(Clone, Debug)]
struct TchMetricSample {
    graph: usize,
    line: usize,
    snapshot: String,
    network_system_id: String,
    stable_line_identity: Option<String>,
    feature: Vec<f32>,
    criticality: Option<[f32; CRITICALITY_OUTPUTS]>,
}

#[derive(Clone, Copy, Debug)]
struct TchMetricTriplet {
    facet: TchMetricFacet,
    anchor: usize,
    positive: usize,
    negative: usize,
}

#[derive(Clone, Debug)]
struct TchCriticalitySample {
    graph: usize,
    line: usize,
    snapshot: String,
    target: [f32; CRITICALITY_OUTPUTS],
}

/// A complete LibTorch multi-task run. The encoder and task heads share one
/// VarStore, and the optimizer owns moments for that entire parameter layout.
/// During metric and criticality phases the encoder outputs are detached, so
/// only task heads update until a separate finalist fine-tune is requested.
pub struct TchMultiTaskSession {
    pub model: TchRelationalAutoencoder,
    pub optimizer: TchOptimizer,
    pub runtime: RuntimeConfig,
    pub cursor: TrainingCursor,
    pub scheduler: SchedulerState,
    pub scaler: ScalerState,
    pub rng: RngState,
    pub sampler: SamplerState,
    pub best_metrics: BestMetricState,
    pub report: TchTrainingReport,
    pub phase_state: TchMultiTaskPhaseState,
    prepared: Vec<PreparedGraph>,
}

impl std::fmt::Debug for TchMultiTaskSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TchMultiTaskSession")
            .field("cursor", &self.cursor)
            .field("runtime", &self.runtime)
            .field("scheduler", &self.scheduler)
            .field("sampler", &self.sampler)
            .field("report", &self.report)
            .field("phase_state", &self.phase_state)
            .field("prepared_graphs", &self.prepared.len())
            .finish_non_exhaustive()
    }
}

impl TchMultiTaskSession {
    pub fn new(graphs: &[&GraphTensor], config: &crate::MultiTaskTrainingConfig) -> Result<Self> {
        let first = graphs.first().context("no graph datasets were provided")?;
        validate_graphs(graphs, first)?;
        config.pretraining.runtime.validate()?;
        config.runtime.validate()?;
        let runtime = effective_multitask_runtime(config);
        configure_runtime(&runtime)?;
        let device = tch_device(&runtime.device)?;
        let model = TchRelationalAutoencoder::new_with_representation_config(
            device,
            first,
            &config.pretraining.model,
            &config.representation,
        );
        let optimizer = TchOptimizer::adam(
            &model.var_store,
            f64::from(config.pretraining.learning_rate),
            f64::from(config.pretraining.weight_decay),
        )?;
        let prepared = graphs
            .iter()
            .map(|graph| model.prepare_graph(graph))
            .collect::<Result<Vec<_>>>()?;
        let graph_order = graphs
            .iter()
            .map(|graph| graph.manifest.snapshot_id.clone())
            .collect::<Vec<_>>();
        Ok(Self {
            model,
            optimizer,
            runtime: runtime.clone(),
            cursor: TrainingCursor {
                phase: "pretraining".into(),
                ..TrainingCursor::default()
            },
            scheduler: SchedulerState {
                last_learning_rate: f64::from(config.pretraining.learning_rate),
                ..SchedulerState::default()
            },
            scaler: ScalerState::default(),
            rng: RngState {
                cpu_seed: config.pretraining.seed,
                gpu_seed: match runtime.device.clone() {
                    crate::runtime::DeviceKind::Cuda { index } => {
                        Some(config.pretraining.seed ^ index as u64)
                    }
                    crate::runtime::DeviceKind::Cpu => None,
                },
                ..RngState::default()
            },
            sampler: SamplerState {
                seed: config.pretraining.seed,
                graph_order,
                ..SamplerState::default()
            },
            best_metrics: BestMetricState::default(),
            report: TchTrainingReport {
                backend: TCH_MULTITASK_CHECKPOINT_BACKEND.into(),
                steps: 0,
                initial_loss: 0.0,
                final_loss: 0.0,
            },
            phase_state: TchMultiTaskPhaseState::default(),
            prepared,
        })
    }

    pub fn resume(
        graphs: &[&GraphTensor],
        config: &crate::MultiTaskTrainingConfig,
        checkpoint: &Path,
        metadata: &CheckpointMetadata,
    ) -> Result<Self> {
        let loaded = load_tch_checkpoint(checkpoint)?;
        if loaded.state.backend != TCH_MULTITASK_CHECKPOINT_BACKEND {
            bail!(
                "checkpoint backend {} is not LibTorch multi-task",
                loaded.state.backend
            );
        }
        validate_state_identity(&loaded.state, metadata)?;
        let mut session = Self::new(graphs, config)?;
        if loaded.state.sampler.graph_order != session.sampler.graph_order {
            bail!("LibTorch checkpoint graph order does not match the requested dataset");
        }
        if !["pretraining", "metric-learning", "criticality"]
            .contains(&loaded.state.cursor.phase.as_str())
        {
            bail!(
                "unsupported LibTorch multi-task checkpoint phase {}",
                loaded.state.cursor.phase
            );
        }
        session
            .model
            .load_weights(&loaded.directory.join("model.ot"))?;
        session.optimizer.load_state(
            &loaded.directory.join("optimizer.ot"),
            &loaded.state.optimizer,
        )?;
        session.cursor = loaded.state.cursor;
        session.scheduler = loaded.state.scheduler;
        session.scaler = loaded.state.scaler;
        session.rng = loaded.state.rng;
        session.sampler = loaded.state.sampler;
        session.best_metrics = loaded.state.best_metrics;
        session.report = loaded.state.report;
        session.phase_state = loaded
            .state
            .multi_task_phase
            .context("LibTorch multi-task checkpoint has no phase state")?;
        Ok(session)
    }

    pub fn fork(
        graphs: &[&GraphTensor],
        config: &crate::MultiTaskTrainingConfig,
        checkpoint: &Path,
    ) -> Result<Self> {
        let loaded = load_tch_checkpoint(checkpoint)?;
        let mut session = Self::new(graphs, config)?;
        session
            .model
            .load_weights(&loaded.directory.join("model.ot"))?;
        Ok(session)
    }

    fn state(&self, metadata: &CheckpointMetadata) -> TchCheckpointState {
        TchCheckpointState {
            schema_version: crate::TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: metadata.run_id.clone(),
            attempt_id: metadata.attempt_id.clone(),
            optimizer: self.optimizer.metadata(),
            scheduler: self.scheduler.clone(),
            scaler: self.scaler.clone(),
            rng: self.rng.clone(),
            cursor: self.cursor.clone(),
            sampler: self.sampler.clone(),
            best_metrics: self.best_metrics.clone(),
            dataset_fingerprint: metadata.dataset_fingerprint.clone(),
            config_fingerprint: metadata.config_fingerprint.clone(),
            code_commit: metadata.code_commit.clone(),
            backend: TCH_MULTITASK_CHECKPOINT_BACKEND.into(),
            backend_version: metadata.backend_version.clone(),
            device_type: metadata.device_type.clone(),
            report: self.report.clone(),
            multi_task_phase: Some(self.phase_state.clone()),
        }
    }

    fn checkpoint(
        &self,
        root: &Path,
        metadata: &CheckpointMetadata,
        observer: &mut dyn TrainingObserver,
    ) -> Result<PathBuf> {
        observer.checkpoint_started(&self.cursor.phase, self.cursor.global_step as usize);
        let state = self.state(metadata);
        let path = save_tch_checkpoint_state(root, &self.model, &self.optimizer, &state)?;
        observer.checkpoint_committed(&self.cursor.phase, self.cursor.global_step as usize, &path);
        Ok(path)
    }

    fn pretraining_cycle(
        &mut self,
        graphs: &[&GraphTensor],
        config: &PretrainingConfig,
    ) -> Result<f64> {
        let accumulation = self.runtime.gradient_accumulation.max(1);
        self.optimizer.zero_grad();
        let mut total_loss = 0.0_f64;
        for _ in 0..accumulation {
            let graph_index = self.sampler.current_graph % graphs.len();
            let graph = graphs[graph_index];
            let prepared = &self.prepared[graph_index];
            let mask = MaskSelection::sample(
                graph,
                &config.mask,
                config
                    .seed
                    .wrapping_add(self.cursor.examples_seen)
                    .wrapping_add(graph_index as u64 * 7_919),
            );
            let reconstruction = self
                .model
                .forward_prepared_unchecked(graph, prepared, &mask, true)?;
            let station_loss = masked_mse(
                &reconstruction.station_features,
                &prepared.station_features,
                &mask.station_rows,
            );
            let line_loss = masked_mse(
                &reconstruction.line_features,
                &prepared.line_features,
                &mask.line_rows,
            );
            let loss = station_loss + line_loss;
            total_loss += loss.double_value(&[]);
            (loss / accumulation as f64).backward();
            self.sampler.current_graph = (graph_index + 1) % graphs.len();
            self.sampler.current_example = self.sampler.current_example.saturating_add(1);
            self.cursor.examples_seen = self
                .cursor
                .examples_seen
                .saturating_add((graph.manifest.station_count + graph.manifest.line_count) as u64);
        }
        self.optimizer.step_gradients()?;
        Ok(total_loss / accumulation as f64)
    }

    fn clean_task_outputs(&self, graphs: &[&GraphTensor]) -> Result<Vec<TchTaskOutputs>> {
        graphs
            .iter()
            .enumerate()
            .map(|(index, graph)| {
                let reconstruction = tch::no_grad(|| {
                    self.model.forward_prepared_unchecked(
                        graph,
                        &self.prepared[index],
                        &MaskSelection::all_unmasked(graph),
                        false,
                    )
                })?;
                let embeddings = TchEmbeddings {
                    station: reconstruction.embeddings.station.detach(),
                    line: reconstruction.embeddings.line.detach(),
                    city: reconstruction.embeddings.city.detach(),
                };
                self.model
                    .task_outputs(&embeddings, &self.prepared[index].line_features, true)
            })
            .collect()
    }
}

/// Run the complete native LibTorch multi-task pipeline. A graph is the unit
/// of pretraining work, while one metric or criticality epoch is the unit of
/// later-phase work; both are complete optimizer boundaries and therefore
/// safe checkpoint points.
pub fn run_tch_multitask_with_policy_options(
    datasets: &[(&GraphTensor, &[LineImpactLabel])],
    config: &crate::MultiTaskTrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    allow_fork: bool,
    observer: &mut dyn TrainingObserver,
) -> Result<(
    TchMultiTaskSession,
    TchTrainingOutcome,
    TchMultiTaskTrainingReport,
)> {
    let Some((first_graph, _)) = datasets.first() else {
        bail!("no graph datasets were provided");
    };
    let graphs = datasets.iter().map(|(graph, _)| *graph).collect::<Vec<_>>();
    validate_graphs(&graphs, first_graph)?;
    config.pretraining.runtime.validate()?;
    config.runtime.validate()?;

    let mut session = match resume {
        Some(path) if allow_fork => TchMultiTaskSession::fork(&graphs, config, path)?,
        Some(path) => TchMultiTaskSession::resume(&graphs, config, path, metadata)?,
        None => TchMultiTaskSession::new(&graphs, config)?,
    };
    if session.sampler.graph_order
        != graphs
            .iter()
            .map(|graph| graph.manifest.snapshot_id.clone())
            .collect::<Vec<_>>()
    {
        bail!("LibTorch checkpoint graph order does not match the requested dataset");
    }

    let samples = build_tch_metric_samples(datasets)?;
    let metric_plan = build_tch_metric_plan(&samples, config.max_triplets);
    let metric_triplets = metric_plan.len();
    session.phase_state.metric_triplets = metric_triplets;
    let criticality_samples = build_tch_criticality_samples(datasets);
    let mut last_checkpoint_path = resume.map(Path::to_path_buf);
    let mut last_saved_step = session.cursor.global_step;
    let mut last_checkpoint_at = Instant::now();

    if session.cursor.phase == "pretraining"
        && session.cursor.global_step < config.pretraining.steps as u64
    {
        observer.phase_started("pretraining", Some(config.pretraining.steps));
        observer.learning_rate_changed(
            "pretraining",
            session.cursor.global_step as usize,
            config.pretraining.learning_rate,
        );
        while session.cursor.global_step < config.pretraining.steps as u64 {
            observer.epoch_started(
                "pretraining",
                session.cursor.global_step as usize + 1,
                config.pretraining.steps,
            );
            let loss = session.pretraining_cycle(&graphs, &config.pretraining)?;
            if session.cursor.global_step == 0 {
                session.report.initial_loss = loss;
            }
            session.report.final_loss = loss;
            session.cursor.global_step = session.cursor.global_step.saturating_add(1);
            session.cursor.epoch = session.cursor.global_step;
            session.cursor.batch = session.cursor.global_step;
            session.report.steps = session.cursor.global_step as usize;
            session.scheduler.last_step = session.cursor.global_step;
            session.scheduler.last_learning_rate = session.optimizer.learning_rate();
            session
                .best_metrics
                .values
                .insert("reconstruction_loss".into(), loss);
            session.best_metrics.steps_without_improvement = 0;
            session.phase_state.pretraining_report = Some(TchTrainingReport {
                backend: "libtorch-multitask-pretraining".into(),
                steps: session.cursor.global_step as usize,
                initial_loss: session.report.initial_loss,
                final_loss: session.report.final_loss,
            });
            observer.metric(
                "pretraining",
                session.cursor.global_step as usize,
                session.cursor.global_step as usize,
                "reconstruction_loss",
                loss as f32,
            );
            if session.cursor.global_step % 10 == 0
                || session.cursor.global_step == config.pretraining.steps as u64
            {
                observer.heartbeat("pretraining", session.cursor.global_step as usize);
            }

            let directive = control.directive()?;
            let force = session.cursor.global_step == config.pretraining.steps as u64;
            if tch_checkpoint_due(
                checkpoint_policy,
                last_checkpoint_at,
                directive,
                force,
                session.cursor.global_step,
            ) && session.cursor.global_step > last_saved_step
            {
                let path = session.checkpoint(checkpoint_root, metadata, observer)?;
                last_checkpoint_path = Some(path);
                last_saved_step = session.cursor.global_step;
                last_checkpoint_at = Instant::now();
            }
            if matches!(directive, ControlDirective::Cancel) {
                observer.phase_completed("pretraining");
                let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
                return Ok((session, TchTrainingOutcome::Cancelled, report));
            }
            if matches!(directive, ControlDirective::Pause)
                || (matches!(directive, ControlDirective::Checkpoint)
                    && control.deadline_expired()
                    && !force)
            {
                let path = last_checkpoint_path
                    .clone()
                    .context("pause requested before a LibTorch checkpoint was committed")?;
                observer.phase_completed("pretraining");
                let outcome = if control.deadline_expired() {
                    TchTrainingOutcome::TimeSliceExpired {
                        checkpoint_path: path,
                    }
                } else {
                    TchTrainingOutcome::Paused {
                        checkpoint_path: path,
                    }
                };
                let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
                return Ok((session, outcome, report));
            }
        }
        observer.phase_completed("pretraining");
    } else if session.cursor.phase == "pretraining" {
        session
            .phase_state
            .pretraining_report
            .get_or_insert_with(|| TchTrainingReport {
                backend: "libtorch-multitask-pretraining".into(),
                steps: session.cursor.global_step as usize,
                initial_loss: session.report.initial_loss,
                final_loss: session.report.final_loss,
            });
    }

    let resuming_metric = session.cursor.phase == "metric-learning";
    let resuming_criticality = session.cursor.phase == "criticality";
    if !resuming_criticality {
        if session.cursor.phase == "pretraining" {
            session.cursor.phase = "metric-learning".into();
            session.cursor.epoch = 0;
            session.cursor.batch = 0;
        }
        if session.cursor.phase != "metric-learning" {
            bail!(
                "unexpected LibTorch multi-task phase {}",
                session.cursor.phase
            );
        }
        let start_epoch = if resuming_metric {
            session.cursor.epoch as usize
        } else {
            0
        };
        if start_epoch > config.metric_epochs {
            bail!("metric-learning checkpoint epoch exceeds configured epochs");
        }
        session
            .optimizer
            .set_learning_rate(f64::from(config.metric_learning_rate))?;
        session.scheduler.last_learning_rate = session.optimizer.learning_rate();
        observer.phase_started("metric-learning", Some(config.metric_epochs));
        observer.learning_rate_changed(
            "metric-learning",
            session.cursor.global_step as usize,
            config.metric_learning_rate,
        );
        for epoch in start_epoch..config.metric_epochs {
            observer.epoch_started("metric-learning", epoch + 1, config.metric_epochs);
            let loss =
                metric_optimizer_epoch(&mut session, &graphs, &samples, &metric_plan, config)?;
            if epoch == 0 && !resuming_metric {
                session.phase_state.metric_initial_loss = Some(loss);
            }
            if session.phase_state.metric_initial_loss.is_none() {
                session.phase_state.metric_initial_loss = Some(loss);
            }
            session.phase_state.metric_final_loss = Some(loss);
            session.cursor.epoch = (epoch + 1) as u64;
            session.cursor.batch = 0;
            session.cursor.global_step = session.cursor.global_step.saturating_add(1);
            session.report.steps = session.cursor.global_step as usize;
            session.scheduler.last_step = session.cursor.global_step;
            session.scheduler.last_learning_rate = session.optimizer.learning_rate();
            session
                .best_metrics
                .values
                .insert("metric_loss".into(), loss);
            session.best_metrics.steps_without_improvement = 0;
            observer.metric(
                "metric-learning",
                epoch + 1,
                session.cursor.global_step as usize,
                "validation_triplet_loss",
                loss as f32,
            );
            observer.metric(
                "metric-learning",
                epoch + 1,
                session.cursor.global_step as usize,
                "triplets",
                metric_triplets as f32,
            );
            if epoch % 10 == 0 || epoch + 1 == config.metric_epochs {
                observer.heartbeat("metric-learning", session.cursor.global_step as usize);
            }
            let directive = control.directive()?;
            let force = epoch + 1 == config.metric_epochs;
            if tch_checkpoint_due(
                checkpoint_policy,
                last_checkpoint_at,
                directive,
                force,
                session.cursor.global_step,
            ) && session.cursor.global_step > last_saved_step
            {
                let path = session.checkpoint(checkpoint_root, metadata, observer)?;
                last_checkpoint_path = Some(path);
                last_saved_step = session.cursor.global_step;
                last_checkpoint_at = Instant::now();
            }
            if matches!(directive, ControlDirective::Cancel) {
                observer.phase_completed("metric-learning");
                let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
                return Ok((session, TchTrainingOutcome::Cancelled, report));
            }
            if matches!(directive, ControlDirective::Pause)
                || (matches!(directive, ControlDirective::Checkpoint)
                    && control.deadline_expired()
                    && !force)
            {
                let path = last_checkpoint_path
                    .clone()
                    .context("pause requested before a metric checkpoint was committed")?;
                observer.phase_completed("metric-learning");
                let outcome = if control.deadline_expired() {
                    TchTrainingOutcome::TimeSliceExpired {
                        checkpoint_path: path,
                    }
                } else {
                    TchTrainingOutcome::Paused {
                        checkpoint_path: path,
                    }
                };
                let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
                return Ok((session, outcome, report));
            }
        }
        observer.phase_completed("metric-learning");
        session.cursor.phase = "criticality".into();
        session.cursor.epoch = 0;
        session.cursor.batch = 0;
    }

    if criticality_samples.is_empty() {
        // No simulator labels means the metric phase is the terminal training
        // phase. Keep the cursor at the actual last completed phase so a
        // weights artifact can still be inspected as an unsupervised model.
        if config.metric_epochs == 0 {
            session.cursor.phase = "pretraining".into();
        } else {
            session.cursor.phase = "metric-learning".into();
            session.cursor.epoch = config.metric_epochs as u64;
        }
    } else {
        if session.cursor.phase != "criticality" {
            bail!("unexpected LibTorch criticality phase transition");
        }
        let start_epoch = if resuming_criticality {
            session.cursor.epoch as usize
        } else {
            0
        };
        if start_epoch > config.criticality.epochs {
            bail!("criticality checkpoint epoch exceeds configured epochs");
        }
        session
            .optimizer
            .set_learning_rate(f64::from(config.criticality.learning_rate))?;
        session.scheduler.last_learning_rate = session.optimizer.learning_rate();
        observer.phase_started("criticality", Some(config.criticality.epochs));
        observer.learning_rate_changed(
            "criticality",
            session.cursor.global_step as usize,
            config.criticality.learning_rate,
        );
        let mut criticality_initial_loss = session
            .phase_state
            .criticality_report
            .as_ref()
            .map(|report| report.initial_loss)
            .unwrap_or(0.0);
        for epoch in start_epoch..config.criticality.epochs {
            observer.epoch_started("criticality", epoch + 1, config.criticality.epochs);
            let loss =
                criticality_optimizer_epoch(&mut session, &graphs, &criticality_samples, config)?;
            if epoch == 0 && !resuming_criticality {
                criticality_initial_loss = loss;
            }
            let report = TchTrainingReport {
                backend: "libtorch-multitask-criticality".into(),
                steps: epoch + 1,
                initial_loss: criticality_initial_loss,
                final_loss: loss,
            };
            session.phase_state.criticality_report = Some(report.clone());
            session.cursor.epoch = (epoch + 1) as u64;
            session.cursor.batch = 0;
            session.cursor.global_step = session.cursor.global_step.saturating_add(1);
            session.report.steps = session.cursor.global_step as usize;
            session.scheduler.last_step = session.cursor.global_step;
            session
                .best_metrics
                .values
                .insert("criticality_loss".into(), loss);
            session.best_metrics.steps_without_improvement = 0;
            observer.metric(
                "criticality",
                epoch + 1,
                session.cursor.global_step as usize,
                "training_huber_loss",
                loss as f32,
            );
            if epoch % 10 == 0 || epoch + 1 == config.criticality.epochs {
                observer.heartbeat("criticality", session.cursor.global_step as usize);
            }
            let directive = control.directive()?;
            let force = epoch + 1 == config.criticality.epochs;
            if tch_checkpoint_due(
                checkpoint_policy,
                last_checkpoint_at,
                directive,
                force,
                session.cursor.global_step,
            ) && session.cursor.global_step > last_saved_step
            {
                let path = session.checkpoint(checkpoint_root, metadata, observer)?;
                last_checkpoint_path = Some(path);
                last_saved_step = session.cursor.global_step;
                last_checkpoint_at = Instant::now();
            }
            if matches!(directive, ControlDirective::Cancel) {
                observer.phase_completed("criticality");
                let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
                return Ok((session, TchTrainingOutcome::Cancelled, report));
            }
            if matches!(directive, ControlDirective::Pause)
                || (matches!(directive, ControlDirective::Checkpoint)
                    && control.deadline_expired()
                    && !force)
            {
                let path = last_checkpoint_path
                    .clone()
                    .context("pause requested before a criticality checkpoint was committed")?;
                observer.phase_completed("criticality");
                let outcome = if control.deadline_expired() {
                    TchTrainingOutcome::TimeSliceExpired {
                        checkpoint_path: path,
                    }
                } else {
                    TchTrainingOutcome::Paused {
                        checkpoint_path: path,
                    }
                };
                let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
                return Ok((session, outcome, report));
            }
        }
        observer.phase_completed("criticality");
    }

    let checkpoint_path = match last_checkpoint_path {
        Some(path) if last_saved_step == session.cursor.global_step => path,
        _ => session.checkpoint(checkpoint_root, metadata, observer)?,
    };
    let report = tch_multitask_report(&session, datasets.len(), samples.len(), config);
    Ok((
        session,
        TchTrainingOutcome::Completed { checkpoint_path },
        report,
    ))
}

fn tch_checkpoint_due(
    policy: CheckpointPolicy,
    last_checkpoint_at: Instant,
    directive: ControlDirective,
    force: bool,
    current_step: u64,
) -> bool {
    force
        || matches!(
            directive,
            ControlDirective::Checkpoint | ControlDirective::Pause
        )
        || policy
            .every_steps
            .filter(|value| *value > 0)
            .is_some_and(|value| current_step % value as u64 == 0)
        || policy
            .every_seconds
            .filter(|value| *value > 0)
            .is_some_and(|value| last_checkpoint_at.elapsed() >= Duration::from_secs(value))
}

fn tch_multitask_report(
    session: &TchMultiTaskSession,
    dataset_count: usize,
    line_count: usize,
    config: &crate::MultiTaskTrainingConfig,
) -> TchMultiTaskTrainingReport {
    TchMultiTaskTrainingReport {
        backend: "libtorch-multitask".into(),
        dataset_count,
        line_count,
        pretraining: session
            .phase_state
            .pretraining_report
            .clone()
            .unwrap_or_else(|| TchTrainingReport {
                backend: "libtorch-multitask-pretraining".into(),
                steps: config.pretraining.steps,
                initial_loss: session.report.initial_loss,
                final_loss: session.report.final_loss,
            }),
        metric_epochs: config.metric_epochs,
        metric_initial_loss: session.phase_state.metric_initial_loss.unwrap_or(0.0),
        metric_final_loss: session.phase_state.metric_final_loss.unwrap_or(0.0),
        metric_triplets: session.phase_state.metric_triplets,
        criticality: session.phase_state.criticality_report.clone(),
    }
}

fn build_tch_metric_samples(
    datasets: &[(&GraphTensor, &[LineImpactLabel])],
) -> Result<Vec<TchMetricSample>> {
    let mut samples = Vec::new();
    for (graph_index, (graph, labels)) in datasets.iter().enumerate() {
        let mut label_by_line = std::collections::HashMap::<u32, [f32; CRITICALITY_OUTPUTS]>::new();
        for label in labels
            .iter()
            .filter(|label| label.snapshot == graph.manifest.snapshot_id)
        {
            let target = label_targets(label);
            if target.iter().all(|value| value.is_finite()) {
                label_by_line.insert(label.line.0, target);
            }
        }
        for line in 0..graph.manifest.line_count {
            samples.push(TchMetricSample {
                graph: graph_index,
                line,
                snapshot: graph.manifest.snapshot_id.clone(),
                network_system_id: graph.manifest.network_system_id.clone(),
                stable_line_identity: graph
                    .line_identities
                    .get(line)
                    .cloned()
                    .filter(|identity| !identity.trim().is_empty()),
                feature: graph.line_features.row(line).to_vec(),
                criticality: label_by_line.get(&(line as u32)).copied(),
            });
        }
    }
    if samples.is_empty() {
        bail!("cannot train LibTorch task heads without line samples");
    }
    Ok(samples)
}

fn build_tch_criticality_samples(
    datasets: &[(&GraphTensor, &[LineImpactLabel])],
) -> Vec<TchCriticalitySample> {
    datasets
        .iter()
        .enumerate()
        .flat_map(|(graph_index, (graph, labels))| {
            labels
                .iter()
                .filter(|label| {
                    label.snapshot == graph.manifest.snapshot_id
                        && (label.line.0 as usize) < graph.manifest.line_count
                })
                .filter_map(move |label| {
                    let target = label_targets(label);
                    target
                        .iter()
                        .all(|value| value.is_finite())
                        .then_some(TchCriticalitySample {
                            graph: graph_index,
                            line: label.line.0 as usize,
                            snapshot: label.snapshot.clone(),
                            target,
                        })
                })
        })
        .collect()
}

fn label_targets(label: &LineImpactLabel) -> [f32; CRITICALITY_OUTPUTS] {
    transit_model::normalize_criticality_targets([
        label.accessibility_auc_loss,
        label.unreachable_share,
        label.mean_delay_reachable_seconds,
        label.p95_delay_reachable_seconds,
        label.mean_extra_transfers,
        label.stations_losing_all_service_share,
    ])
}

fn build_tch_metric_plan(
    samples: &[TchMetricSample],
    maximum_per_facet: usize,
) -> Vec<TchMetricTriplet> {
    if samples.len() < 3 || maximum_per_facet == 0 {
        return Vec::new();
    }
    let mut plan = Vec::new();
    for facet in TchMetricFacet::ALL {
        for anchor in 0..samples.len() {
            let mut candidates = (0..samples.len())
                .filter(|candidate| *candidate != anchor)
                .map(|candidate| {
                    (
                        candidate,
                        tch_sample_distance(&samples[anchor], &samples[candidate], facet),
                    )
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let positive = candidates
                .iter()
                .find(|(candidate, _)| {
                    tch_identity_positive(&samples[anchor], &samples[*candidate])
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
                plan.push(TchMetricTriplet {
                    facet,
                    anchor,
                    positive,
                    negative,
                });
                if plan
                    .iter()
                    .filter(|triplet| facet_equal(triplet.facet, facet))
                    .count()
                    >= maximum_per_facet
                {
                    break;
                }
            }
        }
    }
    plan
}

fn facet_equal(left: TchMetricFacet, right: TchMetricFacet) -> bool {
    std::mem::discriminant(&left) == std::mem::discriminant(&right)
}

fn tch_identity_positive(anchor: &TchMetricSample, candidate: &TchMetricSample) -> bool {
    anchor.network_system_id == candidate.network_system_id
        && !anchor.network_system_id.is_empty()
        && anchor.stable_line_identity.is_some()
        && anchor.stable_line_identity == candidate.stable_line_identity
        && anchor.snapshot != candidate.snapshot
}

fn tch_sample_distance(
    left: &TchMetricSample,
    right: &TchMetricSample,
    facet: TchMetricFacet,
) -> f32 {
    let values = match facet {
        TchMetricFacet::Resilience => match (left.criticality, right.criticality) {
            (Some(left), Some(_right)) => left.to_vec(),
            _ => left.feature.clone(),
        },
        _ => left.feature.clone(),
    };
    let other = match facet {
        TchMetricFacet::Resilience => match (left.criticality, right.criticality) {
            (Some(_), Some(right)) => right.to_vec(),
            _ => right.feature.clone(),
        },
        _ => right.feature.clone(),
    };
    if values.is_empty() || values.len() != other.len() {
        return 1.0;
    }
    let stride = (values.len() / 256).max(1);
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for index in (0..values.len()).step_by(stride) {
        let left = bounded(values[index]);
        let right = bounded(other[index]);
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

fn bounded(value: f32) -> f32 {
    if value.is_finite() {
        value / (1.0 + value.abs())
    } else {
        0.0
    }
}

fn metric_optimizer_epoch(
    session: &mut TchMultiTaskSession,
    graphs: &[&GraphTensor],
    samples: &[TchMetricSample],
    plan: &[TchMetricTriplet],
    config: &crate::MultiTaskTrainingConfig,
) -> Result<f64> {
    let outputs = session.clean_task_outputs(graphs)?;
    session.optimizer.zero_grad();
    let device = session.model.device();
    let mut loss = Tensor::zeros([], (Kind::Float, device)).set_requires_grad(true);
    for triplet in plan {
        let anchor_sample = &samples[triplet.anchor];
        let positive_sample = &samples[triplet.positive];
        let negative_sample = &samples[triplet.negative];
        let anchor = task_row(
            &outputs[anchor_sample.graph],
            triplet.facet,
            anchor_sample.line,
        );
        let positive = task_row(
            &outputs[positive_sample.graph],
            triplet.facet,
            positive_sample.line,
        );
        let negative = task_row(
            &outputs[negative_sample.graph],
            triplet.facet,
            negative_sample.line,
        );
        let positive_distance = (&anchor - &positive)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float);
        let negative_distance = (&anchor - &negative)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float);
        loss = loss + (config.metric_margin as f64 + positive_distance - negative_distance).relu();
    }
    let count = plan.len();
    if count > 0 {
        loss = loss / count as f64;
    }
    loss.backward();
    session.optimizer.step_gradients()?;
    Ok(loss.double_value(&[]))
}

fn task_row(outputs: &TchTaskOutputs, facet: TchMetricFacet, line: usize) -> Tensor {
    match facet {
        TchMetricFacet::Base => outputs.base.get(line as i64),
        TchMetricFacet::General => outputs.general.get(line as i64),
        TchMetricFacet::Role => outputs.role.get(line as i64),
        TchMetricFacet::Service => outputs.service.get(line as i64),
        TchMetricFacet::Geometry => outputs.geometry.get(line as i64),
        TchMetricFacet::Resilience => outputs.resilience.get(line as i64),
    }
}

fn criticality_optimizer_epoch(
    session: &mut TchMultiTaskSession,
    graphs: &[&GraphTensor],
    samples: &[TchCriticalitySample],
    config: &crate::MultiTaskTrainingConfig,
) -> Result<f64> {
    let outputs = session.clean_task_outputs(graphs)?;
    session.optimizer.zero_grad();
    let device = session.model.device();
    let mut loss = Tensor::zeros([], (Kind::Float, device)).set_requires_grad(true);
    let mut snapshot_counts = std::collections::HashMap::<&str, usize>::new();
    for sample in samples {
        *snapshot_counts.entry(sample.snapshot.as_str()).or_default() += 1;
    }
    let mut weight_sum = 0.0_f64;
    for sample in samples {
        let prediction = outputs[sample.graph].criticality.get(sample.line as i64);
        let target = Tensor::from_slice(&sample.target).to_device(device);
        let weight = 1.0 / snapshot_counts[sample.snapshot.as_str()].max(1) as f64;
        loss = loss + prediction.huber_loss(&target, Reduction::Mean, 1.0) * weight;
        weight_sum += weight;
    }
    if config.criticality.ranking_weight > 0.0 {
        let mut pair_count = 0usize;
        'outer: for left in 0..samples.len() {
            for right in (left + 1)..samples.len() {
                if samples[left].snapshot != samples[right].snapshot {
                    continue;
                }
                if pair_count >= config.criticality.max_ranking_pairs {
                    break 'outer;
                }
                let direction = (samples[left].target[0] - samples[right].target[0]).signum();
                if direction == 0.0 {
                    continue;
                }
                let left_prediction = outputs[samples[left].graph]
                    .criticality
                    .get(samples[left].line as i64)
                    .get(0);
                let right_prediction = outputs[samples[right].graph]
                    .criticality
                    .get(samples[right].line as i64)
                    .get(0);
                let margin = (left_prediction - right_prediction) * direction as f64;
                loss = loss
                    + (-margin).softplus() * f64::from(config.criticality.ranking_weight)
                        / config.criticality.max_ranking_pairs.max(1) as f64;
                pair_count += 1;
            }
        }
    }
    if weight_sum > 0.0 {
        loss = loss / weight_sum;
    }
    loss.backward();
    session.optimizer.step_gradients()?;
    Ok(loss.double_value(&[]))
}

/// Compatibility helper for the original weights-only API.  New callers
/// should use [`run_tch_pretraining_with_policy_options`] and retain its
/// committed checkpoint directory.
pub fn train_tch_autoencoder(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    device: Device,
    checkpoint: Option<&Path>,
) -> Result<TchTrainingReport> {
    let mut config = config.clone();
    config.runtime.device = match device {
        Device::Cpu => crate::runtime::DeviceKind::Cpu,
        Device::Cuda(index) => crate::runtime::DeviceKind::Cuda { index },
        other => bail!("unsupported LibTorch device {other:?}"),
    };
    config.runtime.dtype = crate::runtime::DTypeKind::F32;
    let temporary_root = std::env::temp_dir().join(format!(
        "transit-lab-tch-{}-{}",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let metadata = CheckpointMetadata {
        device_type: config.runtime.device.to_string(),
        ..CheckpointMetadata::default()
    };
    let graphs = [graph];
    let control = TrainingControl::new(None, None);
    let result = (|| {
        let (session, outcome) = run_tch_pretraining_with_policy_options(
            &graphs,
            &config,
            &temporary_root,
            None,
            &control,
            CheckpointPolicy::default(),
            &metadata,
            false,
            &mut NoopTrainingObserver,
        )?;
        match outcome {
            TchTrainingOutcome::Completed { checkpoint_path } => {
                if let Some(path) = checkpoint {
                    session.model.save_weights(path)?;
                } else {
                    let _ = checkpoint_path;
                }
                Ok(session.report)
            }
            TchTrainingOutcome::Cancelled => bail!("LibTorch training was cancelled"),
            TchTrainingOutcome::Paused { .. } | TchTrainingOutcome::TimeSliceExpired { .. } => {
                bail!("unexpected control outcome in compatibility training helper")
            }
        }
    })();
    let _ = fs::remove_dir_all(&temporary_root);
    result
}

pub fn save_tch_checkpoint(
    root: &Path,
    session: &TchTrainingSession,
    metadata: &CheckpointMetadata,
) -> Result<PathBuf> {
    let state = session.state(metadata);
    save_tch_checkpoint_state(root, &session.model, &session.optimizer, &state)
}

/// Commit a native LibTorch checkpoint for either the reconstruction-only or
/// multi-task session. The model VarStore already contains the task heads, so
/// this common writer preserves one atomic artifact format for both paths.
fn save_tch_checkpoint_state(
    root: &Path,
    model: &TchRelationalAutoencoder,
    optimizer: &TchOptimizer,
    state: &TchCheckpointState,
) -> Result<PathBuf> {
    state.validate()?;
    fs::create_dir_all(root)
        .with_context(|| format!("creating LibTorch checkpoint root {}", root.display()))?;
    let final_directory = root.join(format!("step-{:012}", state.cursor.global_step));
    if final_directory.exists() {
        let existing = load_tch_checkpoint(&final_directory)?;
        if existing.state.run_id == state.run_id
            && existing.state.dataset_fingerprint == state.dataset_fingerprint
            && existing.state.config_fingerprint == state.config_fingerprint
            && existing.state.cursor.global_step == state.cursor.global_step
        {
            return Ok(final_directory);
        }
        bail!(
            "LibTorch checkpoint step {} already exists with incompatible state",
            state.cursor.global_step
        );
    }
    let temporary_directory = root.join(format!(
        ".tmp-tch-{}-{}",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&temporary_directory).with_context(|| {
        format!(
            "creating temporary LibTorch checkpoint {}",
            temporary_directory.display()
        )
    })?;
    let result = (|| -> Result<()> {
        let model_path = temporary_directory.join("model.ot");
        let optimizer_path = temporary_directory.join("optimizer.ot");
        model.save_weights(&model_path)?;
        optimizer.save_state(&optimizer_path)?;
        sync_file(&model_path)?;
        sync_file(&optimizer_path)?;
        write_json(
            &temporary_directory.join("optimizer.json"),
            &state.optimizer,
        )?;
        write_json(
            &temporary_directory.join("scheduler.json"),
            &state.scheduler,
        )?;
        write_json(&temporary_directory.join("scaler.json"), &state.scaler)?;
        write_json(&temporary_directory.join("rng.json"), &state.rng)?;
        write_json(&temporary_directory.join("cursor.json"), &state.cursor)?;
        write_json(&temporary_directory.join("sampler.json"), &state.sampler)?;
        write_json(
            &temporary_directory.join("best-metrics.json"),
            &state.best_metrics,
        )?;
        write_json(&temporary_directory.join("report.json"), &state.report)?;
        if let Some(phase) = &state.multi_task_phase {
            write_json(&temporary_directory.join("multi-task-phase.json"), phase)?;
        }

        let mut names = vec![
            "model.ot",
            "optimizer.ot",
            "optimizer.json",
            "scheduler.json",
            "scaler.json",
            "rng.json",
            "cursor.json",
            "sampler.json",
            "best-metrics.json",
            "report.json",
        ];
        if state.multi_task_phase.is_some() {
            names.push("multi-task-phase.json");
        }
        let mut files = names
            .into_iter()
            .map(|name| descriptor(&temporary_directory.join(name), name))
            .collect::<Result<Vec<_>>>()?;
        files.sort_by(|left, right| left.path.cmp(&right.path));
        let manifest = TrainingCheckpointManifest {
            schema_version: crate::TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: state.run_id.clone(),
            attempt_id: state.attempt_id.clone(),
            global_step: state.cursor.global_step,
            phase: state.cursor.phase.clone(),
            dataset_fingerprint: state.dataset_fingerprint.clone(),
            config_fingerprint: state.config_fingerprint.clone(),
            code_commit: state.code_commit.clone(),
            backend: state.backend.clone(),
            backend_version: state.backend_version.clone(),
            device_type: state.device_type.clone(),
            status: CheckpointStatus::Committed,
            checkpoint_fingerprint: fingerprint_files(&files),
            files,
        };
        write_json(&temporary_directory.join("manifest.json"), &manifest)?;
        sync_directory(&temporary_directory)?;
        fs::rename(&temporary_directory, &final_directory).with_context(|| {
            format!(
                "committing LibTorch checkpoint {} as {}",
                temporary_directory.display(),
                final_directory.display()
            )
        })?;
        sync_directory(root)?;
        write_latest_pointer(root, &final_directory, &manifest)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary_directory);
    }
    result.map(|()| final_directory)
}

pub fn load_tch_checkpoint(path: &Path) -> Result<TchCheckpointLoad> {
    let directory = resolve_checkpoint_directory(path)?;
    let manifest: TrainingCheckpointManifest = read_json(&directory.join("manifest.json"))?;
    validate_manifest(&manifest)?;
    for file in &manifest.files {
        let payload_path = directory.join(&file.path);
        let bytes = fs::read(&payload_path).with_context(|| {
            format!(
                "reading LibTorch checkpoint payload {}",
                payload_path.display()
            )
        })?;
        if bytes.len() as u64 != file.size_bytes || sha256_hex(&bytes) != file.sha256.to_lowercase()
        {
            bail!(
                "LibTorch checkpoint payload hash or size mismatch for {}",
                file.path
            );
        }
    }
    if fingerprint_files(&manifest.files) != manifest.checkpoint_fingerprint.to_lowercase() {
        bail!("LibTorch checkpoint manifest fingerprint does not match its payloads");
    }
    let state = TchCheckpointState {
        schema_version: manifest.schema_version,
        run_id: manifest.run_id.clone(),
        attempt_id: manifest.attempt_id.clone(),
        optimizer: read_json(&directory.join("optimizer.json"))?,
        scheduler: read_json(&directory.join("scheduler.json"))?,
        scaler: read_json(&directory.join("scaler.json"))?,
        rng: read_json(&directory.join("rng.json"))?,
        cursor: read_json(&directory.join("cursor.json"))?,
        sampler: read_json(&directory.join("sampler.json"))?,
        best_metrics: read_json(&directory.join("best-metrics.json"))?,
        dataset_fingerprint: manifest.dataset_fingerprint.clone(),
        config_fingerprint: manifest.config_fingerprint.clone(),
        code_commit: manifest.code_commit.clone(),
        backend: manifest.backend.clone(),
        backend_version: manifest.backend_version.clone(),
        device_type: manifest.device_type.clone(),
        report: read_json(&directory.join("report.json"))?,
        multi_task_phase: if directory.join("multi-task-phase.json").is_file() {
            Some(read_json(&directory.join("multi-task-phase.json"))?)
        } else {
            None
        },
    };
    state.validate()?;
    Ok(TchCheckpointLoad {
        directory,
        state,
        manifest,
    })
}

pub fn list_tch_training_checkpoints(root: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {}", root.display())),
    };
    let mut candidates = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| valid_checkpoint_directory_name(name))
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|path| checkpoint_sort_key(path));
    Ok(candidates
        .into_iter()
        .filter(|path| load_tch_checkpoint(path).is_ok())
        .collect())
}

/// A line-level native LibTorch inference row.  The CLI converts this into
/// the versioned `PredictionFile` contract; keeping tensor extraction here
/// means callers do not need to depend directly on `tch`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TchLineInference {
    pub base: Vec<f32>,
    pub general: Vec<f32>,
    pub role: Vec<f32>,
    pub service: Vec<f32>,
    pub geometry: Vec<f32>,
    pub resilience: Vec<f32>,
    pub criticality: Vec<f32>,
}

/// Export the complete native model as a JSON descriptor plus a sibling
/// VarStore archive.  The descriptor is intentionally separate from the
/// resumable directory checkpoint: it is the stable model/inference contract,
/// while the checkpoint also contains optimizer and cursor state.
pub fn save_tch_model_artifact(
    output: &Path,
    weights_path: &Path,
    session: &TchMultiTaskSession,
    graphs: &[&GraphTensor],
    config: &crate::MultiTaskTrainingConfig,
    metadata: &CheckpointMetadata,
) -> Result<TchModelArtifactMetadata> {
    let first = graphs.first().context("no graph datasets were provided")?;
    validate_graphs(graphs, first)?;
    let output_parent = output.parent().unwrap_or_else(|| Path::new("."));
    let weights_parent = weights_path.parent().unwrap_or_else(|| Path::new("."));
    if output_parent != weights_parent {
        bail!("native LibTorch metadata and weights must share a directory");
    }
    let weights_name = weights_path
        .file_name()
        .and_then(|value| value.to_str())
        .context("native LibTorch weights path has a non-UTF-8 filename")?;
    let mut descriptor = TchModelArtifactMetadata::for_graph(
        first,
        config.pretraining.model.clone(),
        config.representation.clone(),
        weights_name,
    );
    descriptor.device_type = session.runtime.device.to_string();
    descriptor.snapshot_ids = graphs
        .iter()
        .map(|graph| graph.manifest.snapshot_id.clone())
        .collect();
    descriptor.model_id = std::env::var("TRANSIT_MODEL_ID").ok();
    descriptor.training_run_id = Some(metadata.run_id.clone());
    descriptor.dataset_fingerprint = Some(metadata.dataset_fingerprint.clone());
    descriptor.config_fingerprint = Some(metadata.config_fingerprint.clone());
    descriptor.validate_for_graph(first)?;

    if output.exists() {
        let existing: TchModelArtifactMetadata = read_json(output).with_context(|| {
            format!(
                "decoding existing native LibTorch model metadata {}",
                output.display()
            )
        })?;
        if existing != descriptor {
            bail!(
                "refusing to overwrite immutable native LibTorch model metadata {}",
                output.display()
            );
        }
        if !weights_path.is_file() {
            bail!(
                "native LibTorch model metadata exists but weights are missing at {}",
                weights_path.display()
            );
        }
        return Ok(descriptor);
    }
    if weights_path.exists() {
        bail!(
            "native LibTorch weights already exist without matching metadata at {}",
            weights_path.display()
        );
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating native model directory {}", parent.display()))?;
    }

    let temporary_weights = output_parent.join(format!(
        ".tmp-tch-model-{}-{}.ot",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary_metadata = output_parent.join(format!(
        ".tmp-tch-model-{}-{}.json",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        session.model.save_weights(&temporary_weights)?;
        sync_file(&temporary_weights)?;
        let bytes = serde_json::to_vec_pretty(&descriptor)
            .context("encoding native LibTorch model metadata")?;
        write_synced_file(&temporary_metadata, &bytes)?;
        fs::rename(&temporary_weights, weights_path).with_context(|| {
            format!(
                "committing native LibTorch weights as {}",
                weights_path.display()
            )
        })?;
        fs::rename(&temporary_metadata, output).with_context(|| {
            format!(
                "committing native LibTorch model metadata as {}",
                output.display()
            )
        })?;
        sync_directory(output_parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_weights);
        let _ = fs::remove_file(&temporary_metadata);
        if !output.exists() {
            let _ = fs::remove_file(weights_path);
        }
    }
    result.map(|()| descriptor)
}

/// Run the native LibTorch model on CPU and return line-level task outputs.
/// We intentionally load onto CPU even when the model was trained on CUDA;
/// VarStore remaps the archive tensors to the requested device and this keeps
/// the CLI inference contract usable on workers without a GPU.
pub fn predict_tch_model(
    metadata_path: &Path,
    graph: &GraphTensor,
) -> Result<Vec<TchLineInference>> {
    let metadata: TchModelArtifactMetadata = read_json(metadata_path).with_context(|| {
        format!(
            "decoding native LibTorch model metadata {}",
            metadata_path.display()
        )
    })?;
    metadata.validate_for_graph(graph)?;
    let weights_path = metadata.resolve_weights_path(metadata_path)?;
    let runtime = RuntimeConfig::default();
    configure_runtime(&runtime)?;
    let device = Device::Cpu;
    let mut model = TchRelationalAutoencoder::new_with_representation_config(
        device,
        graph,
        &metadata.model_config,
        &metadata.representation_config,
    );
    model.load_weights(&weights_path)?;
    let prepared = model.prepare_graph(graph)?;
    let outputs = tch::no_grad(|| -> Result<TchTaskOutputs> {
        let reconstruction = model.forward_prepared_unchecked(
            graph,
            &prepared,
            &MaskSelection::all_unmasked(graph),
            false,
        )?;
        let embeddings = TchEmbeddings {
            station: reconstruction.embeddings.station,
            line: reconstruction.embeddings.line,
            city: reconstruction.embeddings.city,
        };
        model.task_outputs(&embeddings, &prepared.line_features, false)
    })?;
    (0..graph.manifest.line_count)
        .map(|line| {
            Ok(TchLineInference {
                base: tensor_values(&outputs.base.get(line as i64))?,
                general: tensor_values(&outputs.general.get(line as i64))?,
                role: tensor_values(&outputs.role.get(line as i64))?,
                service: tensor_values(&outputs.service.get(line as i64))?,
                geometry: tensor_values(&outputs.geometry.get(line as i64))?,
                resilience: tensor_values(&outputs.resilience.get(line as i64))?,
                criticality: tensor_values(&outputs.criticality.get(line as i64))?,
            })
        })
        .collect()
}

fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
    let count = tensor.numel();
    let mut values = vec![0.0_f32; count];
    tensor
        .f_copy_data(&mut values, count)
        .context("copying native LibTorch inference output")?;
    Ok(values)
}

fn validate_graphs(graphs: &[&GraphTensor], first: &GraphTensor) -> Result<()> {
    for graph in graphs {
        graph.validate()?;
        if graph.station_features.cols != first.station_features.cols
            || graph.line_features.cols != first.line_features.cols
            || graph.station_temporal.cols != first.station_temporal.cols
            || graph.line_temporal.cols != first.line_temporal.cols
            || graph.manifest.schema_version != first.manifest.schema_version
        {
            bail!("graph datasets have incompatible feature schemas");
        }
    }
    Ok(())
}

fn effective_multitask_runtime(config: &crate::MultiTaskTrainingConfig) -> RuntimeConfig {
    let defaults = RuntimeConfig::default();
    let mut runtime = config.pretraining.runtime.clone();
    let top_level = &config.runtime;
    if top_level.device != defaults.device {
        runtime.device = top_level.device.clone();
    }
    if top_level.dtype != defaults.dtype {
        runtime.dtype = top_level.dtype.clone();
    }
    if top_level.intraop_threads != defaults.intraop_threads {
        runtime.intraop_threads = top_level.intraop_threads;
    }
    if top_level.interop_threads != defaults.interop_threads {
        runtime.interop_threads = top_level.interop_threads;
    }
    if top_level.rayon_threads != defaults.rayon_threads {
        runtime.rayon_threads = top_level.rayon_threads;
    }
    if top_level.concurrent_training_jobs != defaults.concurrent_training_jobs {
        runtime.concurrent_training_jobs = top_level.concurrent_training_jobs;
    }
    if top_level.gradient_accumulation != defaults.gradient_accumulation {
        runtime.gradient_accumulation = top_level.gradient_accumulation;
    }
    runtime
}

fn configure_runtime(runtime: &RuntimeConfig) -> Result<()> {
    runtime.validate()?;
    if !matches!(runtime.dtype, crate::runtime::DTypeKind::F32) {
        bail!("the resumable LibTorch path currently supports f32 only");
    }
    let requested = (runtime.intraop_threads, runtime.interop_threads);
    if let Some(existing) = THREAD_CONFIGURATION.get() {
        if *existing != requested {
            bail!(
                "LibTorch thread configuration is already fixed at intra-op {}, inter-op {}; requested intra-op {}, inter-op {}",
                existing.0,
                existing.1,
                requested.0,
                requested.1
            );
        }
        return Ok(());
    }
    tch::set_num_threads(requested.0 as i32);
    tch::set_num_interop_threads(requested.1 as i32);
    let _ = THREAD_CONFIGURATION.set(requested);
    Ok(())
}

fn tch_device(device: &crate::runtime::DeviceKind) -> Result<Device> {
    Ok(match device {
        crate::runtime::DeviceKind::Cpu => Device::Cpu,
        crate::runtime::DeviceKind::Cuda { index } => Device::Cuda(*index),
    })
}

fn masked_mse(prediction: &Tensor, target: &Tensor, rows: &[bool]) -> Tensor {
    let row_mask = Tensor::from_slice(
        &rows
            .iter()
            .map(|masked| if *masked { 1.0_f32 } else { 0.0_f32 })
            .collect::<Vec<_>>(),
    )
    .to_device(prediction.device())
    .unsqueeze(1);
    let difference = (prediction - target) * &row_mask;
    (&difference * &difference).sum(Kind::Float) / row_mask.sum(Kind::Float).clamp_min(1.0)
}

fn validate_state_identity(
    state: &TchCheckpointState,
    metadata: &CheckpointMetadata,
) -> Result<()> {
    if state.run_id != metadata.run_id {
        bail!(
            "LibTorch checkpoint run ID {:?} is incompatible with {:?}",
            state.run_id,
            metadata.run_id
        );
    }
    if state.dataset_fingerprint != metadata.dataset_fingerprint {
        bail!("LibTorch checkpoint dataset fingerprint is incompatible with this run");
    }
    if state.config_fingerprint != metadata.config_fingerprint {
        bail!("LibTorch checkpoint configuration fingerprint is incompatible with this run");
    }
    Ok(())
}

fn descriptor(path: &Path, name: &str) -> Result<CheckpointFile> {
    let bytes =
        fs::read(path).with_context(|| format!("reading checkpoint file {}", path.display()))?;
    Ok(CheckpointFile {
        path: name.into(),
        sha256: sha256_hex(&bytes),
        size_bytes: bytes.len() as u64,
    })
}

fn validate_manifest(manifest: &TrainingCheckpointManifest) -> Result<()> {
    if manifest.schema_version != crate::TRAINING_CHECKPOINT_SCHEMA_VERSION {
        bail!("unsupported LibTorch checkpoint manifest schema");
    }
    if manifest.status != CheckpointStatus::Committed {
        bail!("LibTorch checkpoint manifest is not committed");
    }
    if manifest.backend != TCH_CHECKPOINT_BACKEND
        && manifest.backend != TCH_MULTITASK_CHECKPOINT_BACKEND
    {
        bail!("checkpoint is not a LibTorch resumable checkpoint");
    }
    let mut names = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    if names.len() != manifest.files.len()
        || manifest.files.iter().any(|file| {
            file.path.is_empty()
                || file.path.contains('/')
                || file.path.contains('\\')
                || file.path == "."
                || file.path == ".."
                || file.path == "manifest.json"
                || file.path.contains(':')
                || file.sha256.len() != 64
                || !file.sha256.chars().all(|value| value.is_ascii_hexdigit())
        })
    {
        bail!("LibTorch checkpoint manifest contains invalid payload entries");
    }
    for required in [
        "model.ot",
        "optimizer.ot",
        "optimizer.json",
        "scheduler.json",
        "scaler.json",
        "rng.json",
        "cursor.json",
        "sampler.json",
        "best-metrics.json",
        "report.json",
    ] {
        if !manifest.files.iter().any(|file| file.path == required) {
            bail!("LibTorch checkpoint is missing {required}");
        }
    }
    Ok(())
}

fn resolve_checkpoint_directory(path: &Path) -> Result<PathBuf> {
    if path.join("manifest.json").is_file() {
        return Ok(path.to_path_buf());
    }
    if path.join("latest.json").is_file() {
        let pointer: LatestPointer = read_json(&path.join("latest.json"))?;
        if pointer.schema_version == crate::TRAINING_CHECKPOINT_SCHEMA_VERSION
            && valid_checkpoint_directory_name(&pointer.directory)
        {
            let candidate = path.join(pointer.directory);
            if let Ok(loaded) = load_tch_checkpoint(&candidate) {
                if loaded.manifest.global_step == pointer.global_step
                    && loaded.manifest.checkpoint_fingerprint == pointer.checkpoint_fingerprint
                {
                    return Ok(candidate);
                }
            }
        }
    }
    list_tch_training_checkpoints(path)?
        .last()
        .cloned()
        .with_context(|| {
            format!(
                "no committed LibTorch checkpoints found under {}",
                path.display()
            )
        })
}

fn valid_checkpoint_directory_name(name: &str) -> bool {
    name.starts_with("step-")
        && name.len() > "step-".len()
        && !name.contains('/')
        && !name.contains('\\')
        && name["step-".len()..]
            .chars()
            .all(|value| value.is_ascii_digit())
}

fn checkpoint_sort_key(path: &Path) -> u64 {
    path.file_name()
        .and_then(|value| value.to_str())
        .and_then(|value| value.strip_prefix("step-"))
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LatestPointer {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    directory: String,
    #[serde(rename = "globalStep")]
    global_step: u64,
    #[serde(rename = "checkpointFingerprint")]
    checkpoint_fingerprint: String,
}

fn write_latest_pointer(
    root: &Path,
    directory: &Path,
    manifest: &TrainingCheckpointManifest,
) -> Result<()> {
    let name = directory
        .file_name()
        .and_then(|value| value.to_str())
        .context("LibTorch checkpoint directory has a non-UTF-8 name")?;
    let pointer = LatestPointer {
        schema_version: crate::TRAINING_CHECKPOINT_SCHEMA_VERSION,
        directory: name.into(),
        global_step: manifest.global_step,
        checkpoint_fingerprint: manifest.checkpoint_fingerprint.clone(),
    };
    let temporary = root.join(format!(
        ".latest-tch-{}-{}",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    write_json(&temporary, &pointer)?;
    fs::rename(&temporary, root.join("latest.json")).with_context(|| {
        format!(
            "updating LibTorch latest checkpoint pointer in {}",
            root.display()
        )
    })?;
    sync_directory(root)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("encoding LibTorch checkpoint JSON")?;
    write_synced_file(path, &bytes)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("decoding {}", path.display()))
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing {}", path.display()))?;
    Ok(())
}

fn sync_file(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("opening {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("syncing {}", path.display()))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("opening directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("syncing directory {}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn fingerprint_files(files: &[CheckpointFile]) -> String {
    let mut sorted = files.to_vec();
    sorted.sort_by(|left, right| left.path.cmp(&right.path));
    sha256_hex(&serde_json::to_vec(&sorted).expect("checkpoint descriptors are serializable"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_directory_names_are_strict() {
        assert!(valid_checkpoint_directory_name("step-000000000001"));
        assert!(!valid_checkpoint_directory_name("step-"));
        assert!(!valid_checkpoint_directory_name("step-1/escape"));
        assert!(!valid_checkpoint_directory_name(".tmp-step-1"));
    }
}
