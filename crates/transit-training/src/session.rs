use crate::checkpoint::{
    BestMetricState, CheckpointCompatibility, MultiTaskPhaseState, OptimizerState, RngState,
    SamplerState, ScalerState, SchedulerState, TrainingCheckpointV1, TrainingCursor,
};
use crate::control::{ControlDirective, TrainingControl};
use crate::{
    MaskSelection, MultiTaskTrainingConfig, MultiTaskTrainingReport, NoopTrainingObserver,
    PretrainingConfig, ReferenceCheckpoint, ReferenceRelationalAutoencoder, TrainingObserver,
    TrainingReport,
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use transit_graph::GraphTensor;
use transit_labels::LineImpactLabel;
use transit_model::{
    DecoderGradients, Embeddings, ReferenceLineRepresentationEncoder,
    TrainableLineRepresentationModel,
};

#[derive(Clone, Debug)]
pub struct CheckpointMetadata {
    pub run_id: String,
    pub attempt_id: Option<String>,
    pub dataset_fingerprint: String,
    pub config_fingerprint: String,
    pub code_commit: String,
    pub backend_version: String,
    pub device_type: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CheckpointPolicy {
    pub every_steps: Option<usize>,
    pub every_seconds: Option<u64>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            run_id: "local-run".into(),
            attempt_id: None,
            dataset_fingerprint: "local-dataset".into(),
            config_fingerprint: "local-config".into(),
            code_commit: "working-tree".into(),
            backend_version: env!("CARGO_PKG_VERSION").into(),
            device_type: "cpu".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceTrainingSession {
    pub model: ReferenceRelationalAutoencoder,
    pub cursor: TrainingCursor,
    pub optimizer: OptimizerState,
    pub scheduler: SchedulerState,
    pub scaler: ScalerState,
    pub rng: RngState,
    pub sampler: SamplerState,
    pub best_metrics: BestMetricState,
    pub report: TrainingReport,
    pub pending_decoder_gradients: Option<DecoderGradients>,
}

impl ReferenceTrainingSession {
    pub fn new(config: &PretrainingConfig) -> Self {
        Self {
            model: ReferenceRelationalAutoencoder::new(config.model.clone()),
            cursor: TrainingCursor::default(),
            optimizer: OptimizerState::reference(0, config.learning_rate, config.weight_decay),
            scheduler: SchedulerState {
                last_learning_rate: f64::from(config.learning_rate),
                ..SchedulerState::default()
            },
            scaler: ScalerState::default(),
            rng: RngState {
                cpu_seed: config.seed,
                ..RngState::default()
            },
            sampler: SamplerState {
                seed: config.seed,
                ..SamplerState::default()
            },
            best_metrics: BestMetricState::default(),
            report: TrainingReport {
                backend: "reference-cpu-decoder".into(),
                steps: 0,
                initial_loss: 0.0,
                final_loss: 0.0,
            },
            pending_decoder_gradients: None,
        }
    }

    pub fn from_checkpoint(
        checkpoint: TrainingCheckpointV1,
        config: &PretrainingConfig,
        compatibility: CheckpointCompatibility<'_>,
    ) -> Result<Self> {
        crate::validate_checkpoint_compatibility(&checkpoint, compatibility)?;
        let expected = serde_json::to_value(&config.model).context("encoding model config")?;
        let actual = serde_json::to_value(&checkpoint.model.encoder.config)
            .context("encoding checkpoint model config")?;
        if expected != actual {
            anyhow::bail!(
                "checkpoint model architecture is incompatible with the requested config"
            );
        }
        let report = checkpoint.report.unwrap_or(TrainingReport {
            backend: "reference-cpu-decoder".into(),
            steps: checkpoint.cursor.global_step as usize,
            initial_loss: 0.0,
            final_loss: 0.0,
        });
        Ok(Self {
            model: checkpoint.model.encoder,
            cursor: checkpoint.cursor,
            optimizer: checkpoint.optimizer,
            scheduler: checkpoint.scheduler,
            scaler: checkpoint.scaler,
            rng: checkpoint.rng,
            sampler: checkpoint.sampler,
            best_metrics: checkpoint.best_metrics,
            report,
            pending_decoder_gradients: checkpoint.decoder_gradients,
        })
    }

    /// Restore model weights for a new logical run. A fork intentionally does
    /// not inherit the source run's optimizer, scheduler, sampler identity, or
    /// experiment fingerprints, but it still validates the checkpoint payload
    /// and requires the model architecture to match so a silent partial load is
    /// impossible.
    pub fn from_fork_checkpoint(
        checkpoint: TrainingCheckpointV1,
        config: &PretrainingConfig,
    ) -> Result<Self> {
        let expected = serde_json::to_value(&config.model).context("encoding model config")?;
        let actual = serde_json::to_value(&checkpoint.model.encoder.config)
            .context("encoding checkpoint model config")?;
        if expected != actual {
            anyhow::bail!(
                "fork checkpoint model architecture is incompatible with the requested config"
            );
        }
        checkpoint.validate()?;
        let mut session = Self::from_checkpoint(
            checkpoint,
            config,
            CheckpointCompatibility {
                run_id: None,
                dataset_fingerprint: None,
                config_fingerprint: None,
                backend: Some("reference-cpu-decoder"),
                device_type: Some("cpu"),
            },
        )?;
        session.optimizer = OptimizerState::reference(
            session.cursor.global_step,
            config.learning_rate,
            config.weight_decay,
        );
        session.scheduler = SchedulerState {
            last_step: session.cursor.global_step,
            last_learning_rate: f64::from(config.learning_rate),
            ..SchedulerState::default()
        };
        session.rng.cpu_seed = config.seed;
        session.sampler.seed = config.seed;
        session.sampler.current_graph = 0;
        session.sampler.current_example = 0;
        session.sampler.graph_order.clear();
        session.cursor = TrainingCursor::default();
        session.report.steps = 0;
        session.report.initial_loss = 0.0;
        session.report.final_loss = 0.0;
        session.pending_decoder_gradients = None;
        session.report.backend = "reference-cpu-decoder-fork".into();
        Ok(session)
    }

    pub fn optimizer_step(
        &mut self,
        graph: &GraphTensor,
        config: &PretrainingConfig,
        observer: &mut dyn TrainingObserver,
    ) -> Result<f32> {
        self.optimizer_step_on_graph(graph, 0, 1, config, observer)
    }

    /// Execute one deterministic graph unit and record which city must be
    /// selected next. The graph index is checkpointed so a resumed multi-city
    /// run continues the same balanced round-robin order instead of silently
    /// restarting at the first snapshot.
    pub fn optimizer_step_on_graph(
        &mut self,
        graph: &GraphTensor,
        graph_index: usize,
        graph_count: usize,
        config: &PretrainingConfig,
        observer: &mut dyn TrainingObserver,
    ) -> Result<f32> {
        let step = self.cursor.global_step;
        let mask = MaskSelection::sample(
            graph,
            &config.mask,
            config
                .seed
                .wrapping_add(step)
                .wrapping_add(graph_index as u64 * 7919),
        );
        let loss = self
            .model
            .train_decoder_step(graph, &mask, config.learning_rate)?;
        if self.cursor.global_step == 0 {
            self.report.initial_loss = loss;
        }
        self.report.final_loss = loss;
        self.report.steps = (step + 1) as usize;
        self.cursor.global_step += 1;
        self.cursor.epoch = self.cursor.global_step;
        self.cursor.batch = self.cursor.global_step;
        self.cursor.examples_seen +=
            (graph.manifest.station_count + graph.manifest.line_count) as u64;
        self.cursor.gradient_accumulation_position = 0;
        self.optimizer.step = self.cursor.global_step;
        self.scheduler.last_step = self.cursor.global_step;
        self.scheduler.last_learning_rate = f64::from(config.learning_rate);
        self.sampler.current_graph = if graph_count == 0 {
            0
        } else {
            (graph_index + 1) % graph_count
        };
        self.sampler.current_example = self.cursor.global_step as usize;
        self.best_metrics
            .values
            .insert("training_loss".into(), f64::from(loss));
        self.best_metrics.steps_without_improvement = 0;
        observer.metric(
            "pretraining",
            self.cursor.global_step as usize,
            self.cursor.global_step as usize,
            "reconstruction_loss",
            loss,
        );
        Ok(loss)
    }

    /// Add one graph unit to the current accumulation cycle.  No optimizer
    /// state or model weights change until the caller reaches the configured
    /// accumulation boundary, so checkpoints can only observe a complete or
    /// explicitly pending cycle.
    pub fn accumulate_graph_unit(
        &mut self,
        graph: &GraphTensor,
        graph_index: usize,
        graph_count: usize,
        config: &PretrainingConfig,
    ) -> Result<(f32, bool)> {
        let mask = MaskSelection::sample(
            graph,
            &config.mask,
            config
                .seed
                .wrapping_add(self.cursor.examples_seen)
                .wrapping_add(graph_index as u64 * 7919),
        );
        let (gradients, loss) = self.model.decoder_gradients(graph, &mask)?;
        if let Some(pending) = &mut self.pending_decoder_gradients {
            pending.add_assign(&gradients)?;
        } else {
            self.pending_decoder_gradients = Some(gradients);
        }
        self.cursor.gradient_accumulation_position =
            self.cursor.gradient_accumulation_position.saturating_add(1);
        self.cursor.examples_seen = self
            .cursor
            .examples_seen
            .saturating_add((graph.manifest.station_count + graph.manifest.line_count) as u64);
        self.sampler.current_graph = if graph_count == 0 {
            0
        } else {
            (graph_index + 1) % graph_count
        };
        self.sampler.current_example = self.sampler.current_example.saturating_add(1);

        let accumulation = config.runtime.gradient_accumulation.max(1) as u64;
        if self.cursor.gradient_accumulation_position < accumulation {
            return Ok((loss, false));
        }
        let pending = self
            .pending_decoder_gradients
            .take()
            .context("decoder accumulation state disappeared before optimizer step")?;
        self.model.apply_decoder_gradients(
            &pending,
            config.learning_rate,
            config.weight_decay,
            pending.target_count,
        )?;
        let step = self.cursor.global_step;
        let mean_loss = loss;
        if step == 0 {
            self.report.initial_loss = mean_loss;
        }
        self.report.final_loss = mean_loss;
        self.report.steps = (step + 1) as usize;
        self.cursor.global_step += 1;
        self.cursor.epoch = self.cursor.global_step;
        self.cursor.batch = self.cursor.global_step;
        self.cursor.gradient_accumulation_position = 0;
        self.optimizer.step = self.cursor.global_step;
        self.optimizer.learning_rate = f64::from(config.learning_rate);
        self.optimizer.weight_decay = f64::from(config.weight_decay);
        for group in &mut self.optimizer.parameter_groups {
            group.learning_rate = f64::from(config.learning_rate);
            group.weight_decay = f64::from(config.weight_decay);
        }
        self.scheduler.last_step = self.cursor.global_step;
        self.scheduler.last_learning_rate = f64::from(config.learning_rate);
        self.best_metrics
            .values
            .insert("training_loss".into(), f64::from(mean_loss));
        self.best_metrics.steps_without_improvement = 0;
        Ok((mean_loss, true))
    }

    pub fn checkpoint(&self, metadata: &CheckpointMetadata) -> TrainingCheckpointV1 {
        TrainingCheckpointV1 {
            schema_version: crate::TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: metadata.run_id.clone(),
            attempt_id: metadata.attempt_id.clone(),
            model: ReferenceCheckpoint {
                encoder: self.model.clone(),
                head: None,
                report: Some(self.report.clone()),
                representation: None,
                config_fingerprint: Some(metadata.config_fingerprint.clone()),
                seed: Some(self.rng.cpu_seed),
                training_run_id: Some(metadata.run_id.clone()),
                dataset_fingerprint: Some(metadata.dataset_fingerprint.clone()),
                model_id: None,
            },
            optimizer: self.optimizer.clone(),
            scheduler: self.scheduler.clone(),
            scaler: self.scaler.clone(),
            rng: self.rng.clone(),
            cursor: self.cursor.clone(),
            sampler: self.sampler.clone(),
            best_metrics: self.best_metrics.clone(),
            dataset_fingerprint: metadata.dataset_fingerprint.clone(),
            config_fingerprint: metadata.config_fingerprint.clone(),
            code_commit: metadata.code_commit.clone(),
            backend: "reference-cpu-decoder".into(),
            backend_version: metadata.backend_version.clone(),
            device_type: metadata.device_type.clone(),
            report: Some(self.report.clone()),
            decoder_gradients: self.pending_decoder_gradients.clone(),
            multi_task_phase: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ReferenceTrainingOutcome {
    Completed { checkpoint_path: PathBuf },
    Paused { checkpoint_path: PathBuf },
    TimeSliceExpired { checkpoint_path: PathBuf },
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct ResumableMultiTaskResult {
    pub checkpoint: ReferenceCheckpoint,
    pub report: MultiTaskTrainingReport,
}

fn update_phase_optimizer(session: &mut ReferenceTrainingSession, learning_rate: f32) {
    session.optimizer.learning_rate = f64::from(learning_rate);
    for group in &mut session.optimizer.parameter_groups {
        group.learning_rate = f64::from(learning_rate);
    }
    session.scheduler.last_step = session.cursor.global_step;
    session.scheduler.last_learning_rate = f64::from(learning_rate);
}

fn phase_model(
    session: &ReferenceTrainingSession,
    representation: Option<&TrainableLineRepresentationModel>,
    head: Option<&transit_model::CriticalityHead>,
    report: Option<TrainingReport>,
    metadata: &CheckpointMetadata,
) -> ReferenceCheckpoint {
    ReferenceCheckpoint {
        encoder: session.model.clone(),
        head: head.cloned(),
        report,
        representation: representation.cloned(),
        config_fingerprint: Some(metadata.config_fingerprint.clone()),
        seed: Some(session.rng.cpu_seed),
        training_run_id: Some(metadata.run_id.clone()),
        dataset_fingerprint: Some(metadata.dataset_fingerprint.clone()),
        model_id: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn save_multitask_phase_checkpoint(
    session: &ReferenceTrainingSession,
    checkpoint_root: &Path,
    metadata: &CheckpointMetadata,
    phase: &str,
    model: ReferenceCheckpoint,
    phase_state: MultiTaskPhaseState,
    report: TrainingReport,
    observer: &mut dyn TrainingObserver,
) -> Result<PathBuf> {
    observer.checkpoint_started(phase, session.cursor.global_step as usize);
    let mut checkpoint = session.checkpoint(metadata);
    checkpoint.cursor.phase = phase.into();
    checkpoint.model = model;
    checkpoint.report = Some(report);
    checkpoint.multi_task_phase = Some(phase_state);
    let path = crate::save_training_checkpoint(checkpoint_root, &checkpoint)?;
    observer.checkpoint_committed(phase, session.cursor.global_step as usize, &path);
    Ok(path)
}

fn checkpoint_due(
    session: &ReferenceTrainingSession,
    policy: CheckpointPolicy,
    last_checkpoint_at: Instant,
    directive: ControlDirective,
    force: bool,
) -> bool {
    force
        || matches!(
            directive,
            ControlDirective::Checkpoint | ControlDirective::Pause
        )
        || policy
            .every_steps
            .filter(|value| *value > 0)
            .is_some_and(|value| session.cursor.global_step % value as u64 == 0)
        || policy
            .every_seconds
            .filter(|value| *value > 0)
            .is_some_and(|value| last_checkpoint_at.elapsed() >= Duration::from_secs(value))
}

#[allow(clippy::too_many_arguments)]
fn multitask_report(
    pretraining: &TrainingReport,
    dataset_count: usize,
    line_count: usize,
    config: &MultiTaskTrainingConfig,
    metric_initial_loss: f32,
    metric_final_loss: f32,
    metric_triplets: usize,
    criticality: Option<TrainingReport>,
) -> MultiTaskTrainingReport {
    MultiTaskTrainingReport {
        backend: "reference-cpu-multitask".into(),
        dataset_count,
        line_count,
        pretraining: pretraining.clone(),
        metric_epochs: config.metric_epochs,
        metric_initial_loss,
        metric_final_loss,
        metric_triplets,
        criticality,
    }
}

/// Run all reference-backend multi-task phases as one resumable logical run.
/// Pretraining checkpoints contain the encoder; metric checkpoints add the
/// learned facet projections; criticality checkpoints add the final head.
/// Every later-phase checkpoint is committed after a complete epoch, so a
/// resumed process starts at the next epoch with the same deterministic
/// examples and optimizer metadata.
#[allow(clippy::too_many_arguments)]
pub fn run_reference_multitask_with_policy_options(
    datasets: &[(&GraphTensor, &[LineImpactLabel])],
    config: &MultiTaskTrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    allow_fork: bool,
    observer: &mut dyn TrainingObserver,
) -> Result<(ResumableMultiTaskResult, ReferenceTrainingOutcome)> {
    let Some((first_graph, _)) = datasets.first() else {
        anyhow::bail!("no graph datasets were provided");
    };
    let graphs = datasets.iter().map(|(graph, _)| *graph).collect::<Vec<_>>();
    for graph in &graphs {
        if graph.station_features.cols != first_graph.station_features.cols
            || graph.line_features.cols != first_graph.line_features.cols
            || graph.station_temporal.cols != first_graph.station_temporal.cols
            || graph.line_temporal.cols != first_graph.line_temporal.cols
        {
            anyhow::bail!("graph datasets have incompatible feature schemas");
        }
        graph.validate()?;
    }
    config.pretraining.runtime.validate()?;
    config.runtime.validate()?;

    let expected_order = graphs
        .iter()
        .map(|graph| graph.manifest.snapshot_id.clone())
        .collect::<Vec<_>>();
    let mut phase_state = MultiTaskPhaseState::default();
    let mut last_checkpoint_path: Option<PathBuf>;
    let mut session;
    let phase_from_checkpoint;
    let mut restored_representation = None;

    if let Some(path) = resume.filter(|_| !allow_fork) {
        let (checkpoint, _) = crate::load_training_checkpoint(path)?;
        phase_from_checkpoint = checkpoint.cursor.phase.clone();
        if !["pretraining", "metric-learning", "criticality"]
            .contains(&phase_from_checkpoint.as_str())
        {
            anyhow::bail!(
                "unsupported training checkpoint phase {}",
                phase_from_checkpoint
            );
        }
        if phase_from_checkpoint == "pretraining"
            && checkpoint.cursor.global_step > config.pretraining.steps as u64
        {
            anyhow::bail!(
                "pretraining checkpoint step {} exceeds configured pretraining steps {}",
                checkpoint.cursor.global_step,
                config.pretraining.steps
            );
        }
        phase_state = checkpoint.multi_task_phase.clone().unwrap_or_default();
        restored_representation = checkpoint.model.representation.clone();
        let representation = restored_representation.is_some();
        let compatibility = CheckpointCompatibility {
            run_id: Some(metadata.run_id.as_str()),
            dataset_fingerprint: Some(metadata.dataset_fingerprint.as_str()),
            config_fingerprint: Some(metadata.config_fingerprint.as_str()),
            backend: Some("reference-cpu-decoder"),
            device_type: Some(metadata.device_type.as_str()),
        };
        session = ReferenceTrainingSession::from_checkpoint(
            checkpoint,
            &config.pretraining,
            compatibility,
        )?;
        if session.sampler.graph_order != expected_order {
            anyhow::bail!("checkpoint graph order does not match the requested dataset");
        }
        if phase_from_checkpoint != "pretraining" && !representation {
            anyhow::bail!(
                "{} checkpoint has no learned representation",
                phase_from_checkpoint
            );
        }
        if phase_from_checkpoint == "pretraining"
            && session.cursor.global_step < config.pretraining.steps as u64
        {
            // A multi-task run can be interrupted during pretraining.  Do
            // not treat the existence of a valid checkpoint as proof that
            // the phase is complete: continue the phase through the normal
            // resumable pretraining runner before constructing embeddings.
            let (pretraining_session, outcome) =
                run_reference_pretraining_multi_with_policy_options(
                    &graphs,
                    &config.pretraining,
                    checkpoint_root,
                    Some(path),
                    control,
                    checkpoint_policy,
                    metadata,
                    false,
                    observer,
                )?;
            session = pretraining_session;
            match outcome {
                ReferenceTrainingOutcome::Completed { checkpoint_path } => {
                    last_checkpoint_path = Some(checkpoint_path);
                }
                other => {
                    let pretraining_report = session.report.clone();
                    phase_state.pretraining_report = Some(pretraining_report.clone());
                    let model = phase_model(
                        &session,
                        None,
                        None,
                        Some(pretraining_report.clone()),
                        metadata,
                    );
                    let report = multitask_report(
                        &pretraining_report,
                        datasets.len(),
                        0,
                        config,
                        0.0,
                        0.0,
                        0,
                        None,
                    );
                    return Ok((
                        ResumableMultiTaskResult {
                            checkpoint: model,
                            report,
                        },
                        other,
                    ));
                }
            }
        } else {
            last_checkpoint_path = Some(path.to_path_buf());
        }
    } else {
        let (pretraining_session, outcome) = run_reference_pretraining_multi_with_policy_options(
            &graphs,
            &config.pretraining,
            checkpoint_root,
            resume,
            control,
            checkpoint_policy,
            metadata,
            allow_fork,
            observer,
        )?;
        session = pretraining_session;
        match outcome {
            ReferenceTrainingOutcome::Completed { checkpoint_path } => {
                last_checkpoint_path = Some(checkpoint_path);
            }
            other => {
                let pretraining_report = session.report.clone();
                phase_state.pretraining_report = Some(pretraining_report.clone());
                let model = phase_model(
                    &session,
                    None,
                    None,
                    Some(pretraining_report.clone()),
                    metadata,
                );
                let report = multitask_report(
                    &pretraining_report,
                    datasets.len(),
                    0,
                    config,
                    0.0,
                    0.0,
                    0,
                    None,
                );
                return Ok((
                    ResumableMultiTaskResult {
                        checkpoint: model,
                        report,
                    },
                    other,
                ));
            }
        }
        phase_from_checkpoint = "pretraining".to_owned();
    }

    let pretraining_report = phase_state
        .pretraining_report
        .clone()
        .unwrap_or_else(|| session.report.clone());
    phase_state.pretraining_report = Some(pretraining_report.clone());
    let mut embeddings = Vec::<Embeddings>::with_capacity(graphs.len());
    for graph in &graphs {
        embeddings.push(
            session
                .model
                .encode(graph, &MaskSelection::all_unmasked(graph))?,
        );
    }
    let mut representation =
        if phase_from_checkpoint == "metric-learning" || phase_from_checkpoint == "criticality" {
            restored_representation
        } else {
            None
        };
    if representation.is_none() {
        let extractor = ReferenceLineRepresentationEncoder::new(config.representation.clone());
        let raw = extractor.raw_features(first_graph, &embeddings[0])?;
        representation = Some(TrainableLineRepresentationModel::from_raw_features(
            &raw,
            config.representation.clone(),
        )?);
    }
    let mut representation = representation.expect("representation initialized");
    if serde_json::to_value(&representation.config)?
        != serde_json::to_value(&config.representation)?
    {
        anyhow::bail!(
            "checkpoint representation configuration is incompatible with the requested experiment"
        );
    }

    let representation_inputs = datasets
        .iter()
        .zip(&embeddings)
        .map(|((graph, labels), embedding)| (*graph, embedding, *labels))
        .collect::<Vec<_>>();
    let samples = crate::collect_representation_samples(&representation_inputs, &representation)?;
    let metric_plan = crate::build_metric_training_plan(&samples, config.max_triplets);
    let metric_triplets = metric_plan
        .iter()
        .map(|(_, triplets)| triplets.len())
        .sum::<usize>();
    phase_state.metric_triplets = metric_triplets;

    let mut last_saved_step = session.cursor.global_step;
    let mut last_checkpoint_at = Instant::now();
    let mut metric_initial_loss = phase_state.metric_initial_loss.unwrap_or(0.0);
    let mut metric_final_loss = phase_state.metric_final_loss.unwrap_or(0.0);
    let metric_start_epoch = if phase_from_checkpoint == "metric-learning" {
        session.cursor.epoch as usize
    } else if phase_from_checkpoint == "criticality" {
        config.metric_epochs
    } else {
        0
    };
    if metric_start_epoch > config.metric_epochs {
        anyhow::bail!("metric-learning checkpoint epoch exceeds configured epochs");
    }
    session.cursor.phase = "metric-learning".into();
    session.cursor.batch = 0;
    update_phase_optimizer(&mut session, config.metric_learning_rate);
    observer.phase_started("metric-learning", Some(config.metric_epochs));
    observer.learning_rate_changed(
        "metric-learning",
        session.cursor.global_step as usize,
        config.metric_learning_rate,
    );
    for epoch in metric_start_epoch..config.metric_epochs {
        observer.epoch_started("metric-learning", epoch + 1, config.metric_epochs);
        let (loss, _) =
            crate::fit_metric_epoch(&mut representation, &samples, &metric_plan, config)?;
        if (epoch == 0 && phase_from_checkpoint != "metric-learning")
            || (phase_state.metric_initial_loss.is_none() && epoch == metric_start_epoch)
        {
            metric_initial_loss = loss;
        }
        metric_final_loss = loss;
        session.cursor.epoch = (epoch + 1) as u64;
        session.cursor.batch = 0;
        session.cursor.global_step = session.cursor.global_step.saturating_add(1);
        session.optimizer.step = session.cursor.global_step;
        session.scheduler.last_step = session.cursor.global_step;
        session
            .best_metrics
            .values
            .insert("metric_loss".into(), f64::from(loss));
        session.best_metrics.steps_without_improvement = 0;
        observer.metric(
            "metric-learning",
            epoch + 1,
            session.cursor.global_step as usize,
            "validation_triplet_loss",
            loss,
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
        phase_state.metric_initial_loss = Some(metric_initial_loss);
        phase_state.metric_final_loss = Some(metric_final_loss);
        let directive = control.directive()?;
        let force = epoch + 1 == config.metric_epochs;
        if checkpoint_due(
            &session,
            checkpoint_policy,
            last_checkpoint_at,
            directive,
            force,
        ) && session.cursor.global_step > last_saved_step
        {
            let report = TrainingReport {
                backend: "reference-cpu-metric-learning".into(),
                steps: epoch + 1,
                initial_loss: metric_initial_loss,
                final_loss: metric_final_loss,
            };
            let model = phase_model(
                &session,
                Some(&representation),
                None,
                Some(report.clone()),
                metadata,
            );
            let path = save_multitask_phase_checkpoint(
                &session,
                checkpoint_root,
                metadata,
                "metric-learning",
                model,
                phase_state.clone(),
                report,
                observer,
            )?;
            last_checkpoint_path = Some(path);
            last_saved_step = session.cursor.global_step;
            last_checkpoint_at = Instant::now();
        }
        if matches!(directive, ControlDirective::Cancel) {
            observer.phase_completed("metric-learning");
            let report = TrainingReport {
                backend: "reference-cpu-metric-learning".into(),
                steps: epoch + 1,
                initial_loss: metric_initial_loss,
                final_loss: metric_final_loss,
            };
            let checkpoint = phase_model(
                &session,
                Some(&representation),
                None,
                Some(report.clone()),
                metadata,
            );
            return Ok((
                ResumableMultiTaskResult {
                    checkpoint,
                    report: multitask_report(
                        &pretraining_report,
                        datasets.len(),
                        samples.len(),
                        config,
                        metric_initial_loss,
                        metric_final_loss,
                        metric_triplets,
                        None,
                    ),
                },
                ReferenceTrainingOutcome::Cancelled,
            ));
        }
        if matches!(directive, ControlDirective::Pause)
            || (matches!(directive, ControlDirective::Checkpoint)
                && control.deadline_expired()
                && epoch + 1 < config.metric_epochs)
        {
            let path = last_checkpoint_path
                .clone()
                .context("pause requested before a metric checkpoint was committed")?;
            observer.phase_completed("metric-learning");
            let checkpoint = phase_model(
                &session,
                Some(&representation),
                None,
                Some(TrainingReport {
                    backend: "reference-cpu-metric-learning".into(),
                    steps: epoch + 1,
                    initial_loss: metric_initial_loss,
                    final_loss: metric_final_loss,
                }),
                metadata,
            );
            let report = multitask_report(
                &pretraining_report,
                datasets.len(),
                samples.len(),
                config,
                metric_initial_loss,
                metric_final_loss,
                metric_triplets,
                None,
            );
            let outcome = if control.deadline_expired() {
                ReferenceTrainingOutcome::TimeSliceExpired {
                    checkpoint_path: path,
                }
            } else {
                ReferenceTrainingOutcome::Paused {
                    checkpoint_path: path,
                }
            };
            return Ok((ResumableMultiTaskResult { checkpoint, report }, outcome));
        }
    }
    observer.phase_completed("metric-learning");

    let has_criticality = datasets.iter().any(|(_, labels)| !labels.is_empty());
    if has_criticality && control.deadline_expired() {
        let path = last_checkpoint_path
            .clone()
            .context("deadline reached before a metric checkpoint was committed")?;
        let report = TrainingReport {
            backend: "reference-cpu-metric-learning".into(),
            steps: config.metric_epochs,
            initial_loss: metric_initial_loss,
            final_loss: metric_final_loss,
        };
        let checkpoint = phase_model(
            &session,
            Some(&representation),
            None,
            Some(report),
            metadata,
        );
        let report = multitask_report(
            &pretraining_report,
            datasets.len(),
            samples.len(),
            config,
            metric_initial_loss,
            metric_final_loss,
            metric_triplets,
            None,
        );
        return Ok((
            ResumableMultiTaskResult { checkpoint, report },
            ReferenceTrainingOutcome::TimeSliceExpired {
                checkpoint_path: path,
            },
        ));
    }
    let mut criticality_head = None;
    let mut criticality_report = phase_state.criticality_report.clone();
    if has_criticality {
        let (input_dimension, examples) = crate::build_representation_criticality_examples(
            &representation,
            &representation_inputs,
        )?;
        if phase_from_checkpoint == "criticality" {
            let (checkpoint, _) = crate::load_training_checkpoint(
                resume.context("criticality resume requires its checkpoint path")?,
            )?;
            criticality_head = checkpoint.model.head;
            if criticality_head.is_none() {
                anyhow::bail!("criticality checkpoint has no criticality head");
            }
        } else {
            criticality_head = Some(transit_model::CriticalityHead::new(
                input_dimension,
                transit_model::CRITICALITY_OUTPUTS,
                config.criticality.seed,
            ));
        }
        if criticality_head
            .as_ref()
            .is_some_and(|head| head.input_dimension != input_dimension)
        {
            anyhow::bail!(
                "criticality checkpoint input width is incompatible with the requested dataset"
            );
        }
        let mut head = criticality_head
            .take()
            .expect("criticality head initialized");
        let criticality_start_epoch = if phase_from_checkpoint == "criticality" {
            session.cursor.epoch as usize
        } else {
            0
        };
        if criticality_start_epoch > config.criticality.epochs {
            anyhow::bail!("criticality checkpoint epoch exceeds configured epochs");
        }
        session.cursor.phase = "criticality".into();
        session.cursor.batch = 0;
        update_phase_optimizer(&mut session, config.criticality.learning_rate);
        observer.phase_started("criticality", Some(config.criticality.epochs));
        observer.learning_rate_changed(
            "criticality",
            session.cursor.global_step as usize,
            config.criticality.learning_rate,
        );
        let mut criticality_initial_loss = criticality_report
            .as_ref()
            .map(|report| report.initial_loss)
            .unwrap_or(0.0);
        let mut criticality_final_loss;
        for epoch in criticality_start_epoch..config.criticality.epochs {
            observer.epoch_started("criticality", epoch + 1, config.criticality.epochs);
            let loss = crate::fit_criticality_epoch(&mut head, &examples, &config.criticality)?;
            if (epoch == 0 && phase_from_checkpoint != "criticality")
                || (criticality_report.is_none() && epoch == criticality_start_epoch)
            {
                criticality_initial_loss = loss;
            }
            criticality_final_loss = loss;
            session.cursor.epoch = (epoch + 1) as u64;
            session.cursor.batch = 0;
            session.cursor.global_step = session.cursor.global_step.saturating_add(1);
            session.optimizer.step = session.cursor.global_step;
            session.scheduler.last_step = session.cursor.global_step;
            session
                .best_metrics
                .values
                .insert("criticality_loss".into(), f64::from(loss));
            session.best_metrics.steps_without_improvement = 0;
            observer.metric(
                "criticality",
                epoch + 1,
                session.cursor.global_step as usize,
                "training_huber_loss",
                loss,
            );
            if epoch % 10 == 0 || epoch + 1 == config.criticality.epochs {
                observer.heartbeat("criticality", session.cursor.global_step as usize);
            }
            let report = TrainingReport {
                backend: "reference-cpu-representation-head".into(),
                steps: epoch + 1,
                initial_loss: criticality_initial_loss,
                final_loss: criticality_final_loss,
            };
            criticality_report = Some(report.clone());
            phase_state.criticality_report = Some(report.clone());
            let directive = control.directive()?;
            let force = epoch + 1 == config.criticality.epochs;
            if checkpoint_due(
                &session,
                checkpoint_policy,
                last_checkpoint_at,
                directive,
                force,
            ) && session.cursor.global_step > last_saved_step
            {
                let model = phase_model(
                    &session,
                    Some(&representation),
                    Some(&head),
                    Some(report.clone()),
                    metadata,
                );
                let path = save_multitask_phase_checkpoint(
                    &session,
                    checkpoint_root,
                    metadata,
                    "criticality",
                    model,
                    phase_state.clone(),
                    report,
                    observer,
                )?;
                last_checkpoint_path = Some(path);
                last_saved_step = session.cursor.global_step;
                last_checkpoint_at = Instant::now();
            }
            if matches!(directive, ControlDirective::Cancel) {
                observer.phase_completed("criticality");
                let report = criticality_report.clone();
                let checkpoint = phase_model(
                    &session,
                    Some(&representation),
                    Some(&head),
                    report.clone(),
                    metadata,
                );
                return Ok((
                    ResumableMultiTaskResult {
                        checkpoint,
                        report: multitask_report(
                            &pretraining_report,
                            datasets.len(),
                            samples.len(),
                            config,
                            metric_initial_loss,
                            metric_final_loss,
                            metric_triplets,
                            report,
                        ),
                    },
                    ReferenceTrainingOutcome::Cancelled,
                ));
            }
            if matches!(directive, ControlDirective::Pause)
                || (matches!(directive, ControlDirective::Checkpoint)
                    && control.deadline_expired()
                    && epoch + 1 < config.criticality.epochs)
            {
                let path = last_checkpoint_path
                    .clone()
                    .context("pause requested before a criticality checkpoint was committed")?;
                observer.phase_completed("criticality");
                let checkpoint = phase_model(
                    &session,
                    Some(&representation),
                    Some(&head),
                    criticality_report.clone(),
                    metadata,
                );
                let report = multitask_report(
                    &pretraining_report,
                    datasets.len(),
                    samples.len(),
                    config,
                    metric_initial_loss,
                    metric_final_loss,
                    metric_triplets,
                    criticality_report.clone(),
                );
                let outcome = if control.deadline_expired() {
                    ReferenceTrainingOutcome::TimeSliceExpired {
                        checkpoint_path: path,
                    }
                } else {
                    ReferenceTrainingOutcome::Paused {
                        checkpoint_path: path,
                    }
                };
                return Ok((ResumableMultiTaskResult { checkpoint, report }, outcome));
            }
        }
        observer.phase_completed("criticality");
        criticality_head = Some(head);
    }

    let final_report = criticality_report.clone().or_else(|| {
        phase_state
            .metric_final_loss
            .map(|final_loss| TrainingReport {
                backend: "reference-cpu-metric-learning".into(),
                steps: config.metric_epochs,
                initial_loss: metric_initial_loss,
                final_loss,
            })
    });
    let checkpoint = phase_model(
        &session,
        Some(&representation),
        criticality_head.as_ref(),
        final_report.clone(),
        metadata,
    );
    let report = multitask_report(
        &pretraining_report,
        datasets.len(),
        samples.len(),
        config,
        metric_initial_loss,
        metric_final_loss,
        metric_triplets,
        criticality_report,
    );
    let checkpoint_path = last_checkpoint_path
        .context("multi-task completion did not produce a committed checkpoint")?;
    Ok((
        ResumableMultiTaskResult { checkpoint, report },
        ReferenceTrainingOutcome::Completed { checkpoint_path },
    ))
}

/// Run the reference pretraining phase as a resumable logical execution. The
/// process can exit after this function returns; all model and cursor state is
/// already in the committed checkpoint directory.
#[allow(clippy::too_many_arguments)]
pub fn run_reference_pretraining(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_every_steps: Option<usize>,
    metadata: &CheckpointMetadata,
    observer: &mut dyn TrainingObserver,
) -> Result<(ReferenceTrainingSession, ReferenceTrainingOutcome)> {
    run_reference_pretraining_with_policy(
        graph,
        config,
        checkpoint_root,
        resume,
        control,
        CheckpointPolicy {
            every_steps: checkpoint_every_steps,
            every_seconds: None,
        },
        metadata,
        observer,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_reference_pretraining_with_policy(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    observer: &mut dyn TrainingObserver,
) -> Result<(ReferenceTrainingSession, ReferenceTrainingOutcome)> {
    run_reference_pretraining_with_policy_options(
        graph,
        config,
        checkpoint_root,
        resume,
        control,
        checkpoint_policy,
        metadata,
        false,
        observer,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_reference_pretraining_with_policy_options(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    allow_fork: bool,
    observer: &mut dyn TrainingObserver,
) -> Result<(ReferenceTrainingSession, ReferenceTrainingOutcome)> {
    run_reference_pretraining_multi_with_policy_options(
        &[graph],
        config,
        checkpoint_root,
        resume,
        control,
        checkpoint_policy,
        metadata,
        allow_fork,
        observer,
    )
}

/// Run resumable pretraining over a balanced round-robin sequence of graphs.
/// A checkpoint stores the graph order and next graph cursor, making a pause
/// or worker restart transparent even when the dataset contains several city
/// systems.
#[allow(clippy::too_many_arguments)]
pub fn run_reference_pretraining_multi_with_policy_options(
    graphs: &[&GraphTensor],
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    allow_fork: bool,
    observer: &mut dyn TrainingObserver,
) -> Result<(ReferenceTrainingSession, ReferenceTrainingOutcome)> {
    let Some(first_graph) = graphs.first() else {
        anyhow::bail!("no graph datasets were provided");
    };
    for graph in graphs {
        if graph.station_features.cols != first_graph.station_features.cols
            || graph.line_features.cols != first_graph.line_features.cols
            || graph.station_temporal.cols != first_graph.station_temporal.cols
            || graph.line_temporal.cols != first_graph.line_temporal.cols
        {
            anyhow::bail!("graph datasets have incompatible feature schemas");
        }
        graph.validate()?;
    }
    let mut session = if let Some(path) = resume {
        let (checkpoint, _) = crate::load_training_checkpoint(path)?;
        if allow_fork {
            ReferenceTrainingSession::from_fork_checkpoint(checkpoint, config)?
        } else {
            ReferenceTrainingSession::from_checkpoint(
                checkpoint,
                config,
                CheckpointCompatibility {
                    run_id: Some(metadata.run_id.as_str()),
                    dataset_fingerprint: Some(metadata.dataset_fingerprint.as_str()),
                    config_fingerprint: Some(metadata.config_fingerprint.as_str()),
                    backend: Some("reference-cpu-decoder"),
                    device_type: Some(metadata.device_type.as_str()),
                },
            )?
        }
    } else {
        ReferenceTrainingSession::new(config)
    };
    let graph_order = graphs
        .iter()
        .map(|graph| graph.manifest.snapshot_id.clone())
        .collect::<Vec<_>>();
    if session.sampler.graph_order.is_empty() {
        session.sampler.graph_order = graph_order.clone();
        session.sampler.current_graph %= graphs.len();
    } else if session.sampler.graph_order != graph_order {
        anyhow::bail!("checkpoint graph order does not match the requested dataset");
    } else {
        session.sampler.current_graph %= graphs.len();
    }
    observer.phase_started("pretraining", Some(config.steps));
    observer.learning_rate_changed(
        "pretraining",
        session.cursor.global_step as usize,
        config.learning_rate,
    );
    let checkpoint_interval = checkpoint_policy
        .every_steps
        .filter(|interval| *interval > 0);
    let checkpoint_seconds = checkpoint_policy
        .every_seconds
        .filter(|interval| *interval > 0)
        .map(Duration::from_secs);
    let started_at = Instant::now();
    let mut last_checkpoint_at = started_at;
    while session.cursor.global_step < config.steps as u64 {
        let mut last_loss = 0.0_f32;
        let mut completed_optimizer_step = false;
        while !completed_optimizer_step {
            let graph_index = session.sampler.current_graph % graphs.len();
            let graph = graphs[graph_index];
            let (loss, completed) =
                session.accumulate_graph_unit(graph, graph_index, graphs.len(), config)?;
            last_loss = loss;
            completed_optimizer_step = completed;
        }
        observer.epoch_started(
            "pretraining",
            session.cursor.global_step as usize,
            config.steps,
        );
        observer.metric(
            "pretraining",
            session.cursor.global_step as usize,
            session.cursor.global_step as usize,
            "reconstruction_loss",
            last_loss,
        );
        if session.cursor.global_step % 10 == 0 || session.cursor.global_step == config.steps as u64
        {
            observer.heartbeat("pretraining", session.cursor.global_step as usize);
        }
        let directive = control.directive()?;
        let periodic_checkpoint = session.cursor.global_step < config.steps as u64
            && checkpoint_interval
                .is_some_and(|interval| session.cursor.global_step % interval as u64 == 0);
        let periodic_time_checkpoint = session.cursor.global_step < config.steps as u64
            && checkpoint_seconds.is_some_and(|interval| last_checkpoint_at.elapsed() >= interval);
        if periodic_checkpoint
            || periodic_time_checkpoint
            || matches!(
                directive,
                ControlDirective::Checkpoint | ControlDirective::Pause
            )
        {
            let path = save_session_checkpoint(&session, checkpoint_root, metadata, observer)?;
            last_checkpoint_at = Instant::now();
            if matches!(directive, ControlDirective::Pause) {
                observer.phase_completed("pretraining");
                return Ok((
                    session,
                    ReferenceTrainingOutcome::Paused {
                        checkpoint_path: path,
                    },
                ));
            }
            if matches!(directive, ControlDirective::Checkpoint)
                && control.deadline_expired()
                && session.cursor.global_step < config.steps as u64
            {
                observer.phase_completed("pretraining");
                return Ok((
                    session,
                    ReferenceTrainingOutcome::TimeSliceExpired {
                        checkpoint_path: path,
                    },
                ));
            }
        }
        if matches!(directive, ControlDirective::Cancel) {
            observer.phase_completed("pretraining");
            return Ok((session, ReferenceTrainingOutcome::Cancelled));
        }
    }
    let path = save_session_checkpoint(&session, checkpoint_root, metadata, observer)?;
    observer.phase_completed("pretraining");
    Ok((
        session,
        ReferenceTrainingOutcome::Completed {
            checkpoint_path: path,
        },
    ))
}

/// Quiet convenience wrapper for balanced multi-city pretraining.
#[allow(clippy::too_many_arguments)]
pub fn run_reference_pretraining_multi_with_policy(
    graphs: &[&GraphTensor],
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_policy: CheckpointPolicy,
    metadata: &CheckpointMetadata,
    observer: &mut dyn TrainingObserver,
) -> Result<(ReferenceTrainingSession, ReferenceTrainingOutcome)> {
    run_reference_pretraining_multi_with_policy_options(
        graphs,
        config,
        checkpoint_root,
        resume,
        control,
        checkpoint_policy,
        metadata,
        false,
        observer,
    )
}

fn save_session_checkpoint(
    session: &ReferenceTrainingSession,
    checkpoint_root: &Path,
    metadata: &CheckpointMetadata,
    observer: &mut dyn TrainingObserver,
) -> Result<PathBuf> {
    observer.checkpoint_started("pretraining", session.cursor.global_step as usize);
    let path = crate::save_training_checkpoint(checkpoint_root, &session.checkpoint(metadata))?;
    observer.checkpoint_committed("pretraining", session.cursor.global_step as usize, &path);
    Ok(path)
}

/// Convenience entry point for callers that do not need a custom observer.
pub fn run_reference_pretraining_quiet(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    checkpoint_root: &Path,
    resume: Option<&Path>,
    control: &TrainingControl,
    checkpoint_every_steps: Option<usize>,
    metadata: &CheckpointMetadata,
) -> Result<(ReferenceTrainingSession, ReferenceTrainingOutcome)> {
    let mut observer = NoopTrainingObserver;
    run_reference_pretraining(
        graph,
        config,
        checkpoint_root,
        resume,
        control,
        checkpoint_every_steps,
        metadata,
        &mut observer,
    )
}

pub fn max_wall_time(seconds: Option<u64>) -> Option<Duration> {
    seconds.map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{list_training_checkpoints, DesiredTrainingState, TrainingControlFile};
    use chrono::NaiveDate;
    use gtfs_compile::{compile, CompileOptions};
    use gtfs_ingest::GtfsFeed;
    use std::fs;
    use tempfile::tempdir;
    use transit_domain::LineIndex;

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

    fn config(steps: usize) -> PretrainingConfig {
        PretrainingConfig {
            model: crate::ModelConfig {
                hidden_dimension: 8,
                graph_layers: 1,
                dropout: 0.0,
                ..crate::ModelConfig::default()
            },
            steps,
            learning_rate: 0.00001,
            ..PretrainingConfig::default()
        }
    }

    fn metadata() -> CheckpointMetadata {
        CheckpointMetadata {
            run_id: "run-equivalence".into(),
            dataset_fingerprint: "dataset-equivalence".into(),
            config_fingerprint: "config-equivalence".into(),
            ..CheckpointMetadata::default()
        }
    }

    fn multitask_config(
        pretraining_steps: usize,
        metric_epochs: usize,
        criticality_epochs: usize,
        gradient_accumulation: usize,
    ) -> MultiTaskTrainingConfig {
        let mut pretraining = config(pretraining_steps);
        pretraining.runtime.gradient_accumulation = gradient_accumulation;
        MultiTaskTrainingConfig {
            pretraining,
            representation: transit_model::RepresentationConfig {
                base_dimension: 12,
                city_dimension: 8,
                general_dimension: 8,
                role_dimension: 8,
                service_dimension: 8,
                geometry_dimension: 8,
                resilience_dimension: 8,
                seed: 5,
            },
            metric_epochs,
            metric_learning_rate: 0.0001,
            metric_margin: 0.25,
            metric_weight_decay: 0.00001,
            max_triplets: 32,
            criticality: crate::CriticalityTrainingConfig {
                epochs: criticality_epochs,
                learning_rate: 0.0001,
                ranking_weight: 0.25,
                seed: 19,
                max_ranking_pairs: 32,
            },
            runtime: crate::RuntimeConfig::default(),
        }
    }

    fn labels(graph: &GraphTensor) -> Vec<LineImpactLabel> {
        (0..graph.manifest.line_count)
            .map(|line| LineImpactLabel {
                snapshot: graph.manifest.snapshot_id.clone(),
                line: LineIndex(line as u32),
                accessibility_auc_loss: 0.01 + line as f32 * 0.02,
                unreachable_share: 0.02 + line as f32 * 0.01,
                mean_delay_reachable_seconds: 20.0 + line as f32,
                p95_delay_reachable_seconds: 40.0 + line as f32 * 2.0,
                mean_extra_transfers: 0.1 + line as f32 * 0.03,
                stations_losing_all_service_share: 0.05 + line as f32 * 0.01,
                query_count: 8,
                router_algorithm_version: transit_labels::ROUTER_ALGORITHM_VERSION.into(),
                policy_fingerprint: "test-policy".into(),
            })
            .collect()
    }

    fn write_control(path: &std::path::Path, desired_state: DesiredTrainingState) {
        fs::write(
            path,
            serde_json::to_vec(&TrainingControlFile {
                desired_state,
                ..TrainingControlFile::default()
            })
            .unwrap(),
        )
        .unwrap();
    }

    struct PauseOnEpoch {
        path: std::path::PathBuf,
        phase: &'static str,
        epoch: usize,
        requested: bool,
    }

    impl TrainingObserver for PauseOnEpoch {
        fn epoch_started(&mut self, phase: &str, epoch: usize, _total: usize) {
            if !self.requested && phase == self.phase && epoch == self.epoch {
                write_control(&self.path, DesiredTrainingState::Paused);
                self.requested = true;
            }
        }
    }

    #[test]
    fn continuous_and_split_training_match_after_resume() {
        let graph = graph();
        let continuous_root = tempdir().unwrap();
        let split_root = tempdir().unwrap();
        let full_config = config(12);
        let control = TrainingControl::new(None, None);
        let (continuous, ReferenceTrainingOutcome::Completed { .. }) = run_reference_pretraining(
            &graph,
            &full_config,
            continuous_root.path(),
            None,
            &control,
            Some(12),
            &metadata(),
            &mut NoopTrainingObserver,
        )
        .unwrap() else {
            panic!("continuous run did not complete")
        };

        let first_config = config(5);
        let (_, ReferenceTrainingOutcome::Completed { checkpoint_path }) =
            run_reference_pretraining(
                &graph,
                &first_config,
                split_root.path(),
                None,
                &control,
                Some(5),
                &metadata(),
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("first split run did not complete")
        };
        let (split, ReferenceTrainingOutcome::Completed { .. }) = run_reference_pretraining(
            &graph,
            &full_config,
            split_root.path(),
            Some(&checkpoint_path),
            &control,
            Some(12),
            &metadata(),
            &mut NoopTrainingObserver,
        )
        .unwrap() else {
            panic!("resumed run did not complete")
        };

        assert_eq!(continuous.cursor.global_step, split.cursor.global_step);
        assert_eq!(
            serde_json::to_vec(&continuous.model).unwrap(),
            serde_json::to_vec(&split.model).unwrap()
        );
        assert_eq!(continuous.optimizer.step, split.optimizer.step);
        assert_eq!(
            continuous.sampler.current_example,
            split.sampler.current_example
        );
    }

    #[test]
    fn multi_city_resume_preserves_round_robin_graph_cursor() {
        let first_graph = graph();
        let mut second_graph = first_graph.clone();
        second_graph.manifest.snapshot_id = "snapshot-second-city".into();
        let graphs = vec![&first_graph, &second_graph];
        let continuous_root = tempdir().unwrap();
        let split_root = tempdir().unwrap();
        let full_config = config(8);
        let metadata = CheckpointMetadata {
            dataset_fingerprint: "snapshot-first,snapshot-second-city".into(),
            ..metadata()
        };
        let control = TrainingControl::new(None, None);
        let (
            continuous,
            ReferenceTrainingOutcome::Completed {
                checkpoint_path: continuous_checkpoint_path,
            },
        ) = run_reference_pretraining_multi_with_policy_options(
            &graphs,
            &full_config,
            continuous_root.path(),
            None,
            &control,
            CheckpointPolicy {
                every_steps: Some(8),
                every_seconds: None,
            },
            &metadata,
            false,
            &mut NoopTrainingObserver,
        )
        .unwrap()
        else {
            panic!("multi-city continuous run did not complete")
        };
        let first_config = config(3);
        let (_, ReferenceTrainingOutcome::Completed { checkpoint_path }) =
            run_reference_pretraining_multi_with_policy_options(
                &graphs,
                &first_config,
                split_root.path(),
                None,
                &control,
                CheckpointPolicy {
                    every_steps: Some(3),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("multi-city first split did not complete")
        };
        let (split, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_pretraining_multi_with_policy_options(
                &graphs,
                &full_config,
                split_root.path(),
                Some(&checkpoint_path),
                &control,
                CheckpointPolicy {
                    every_steps: Some(8),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("multi-city resumed run did not complete")
        };
        assert_eq!(
            continuous.sampler.graph_order,
            vec![
                first_graph.manifest.snapshot_id.clone(),
                "snapshot-second-city".into()
            ]
        );
        let (persisted, _) = crate::load_training_checkpoint(&continuous_checkpoint_path).unwrap();
        assert_eq!(
            persisted.sampler.graph_order,
            continuous.sampler.graph_order
        );
        assert_eq!(
            continuous.sampler.current_graph,
            split.sampler.current_graph
        );
        assert_eq!(continuous.cursor.global_step, split.cursor.global_step);
        assert_eq!(
            serde_json::to_vec(&continuous.model).unwrap(),
            serde_json::to_vec(&split.model).unwrap()
        );
    }

    #[test]
    fn multitask_resume_continues_an_intermediate_pretraining_checkpoint() {
        let graph = graph();
        let config = multitask_config(5, 2, 0, 1);
        let empty_labels = Vec::<LineImpactLabel>::new();
        let datasets = [(&graph, empty_labels.as_slice())];
        let full_root = tempdir().unwrap();
        let split_root = tempdir().unwrap();
        let full_control = TrainingControl::new(None, None);
        let (full, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                full_root.path(),
                None,
                &full_control,
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata(),
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("full multi-task run did not complete")
        };

        let control_path = split_root.path().join("control.json");
        let split_control = TrainingControl::from_path(&control_path);
        let mut pause = PauseOnEpoch {
            path: control_path.clone(),
            phase: "pretraining",
            epoch: 2,
            requested: false,
        };
        let (_, ReferenceTrainingOutcome::Paused { checkpoint_path }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                split_root.path(),
                None,
                &split_control,
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata(),
                false,
                &mut pause,
            )
            .unwrap()
        else {
            panic!("pretraining pause did not produce a paused outcome")
        };
        write_control(&control_path, DesiredTrainingState::Running);
        let (resumed, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                split_root.path(),
                Some(&checkpoint_path),
                &TrainingControl::from_path(&control_path),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata(),
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("resumed multi-task run did not complete")
        };

        assert_eq!(
            serde_json::to_vec(&full.checkpoint).unwrap(),
            serde_json::to_vec(&resumed.checkpoint).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&full.report).unwrap(),
            serde_json::to_vec(&resumed.report).unwrap()
        );
    }

    #[test]
    fn multitask_metric_learning_resume_restores_representation_state() {
        let graph = graph();
        let config = multitask_config(1, 6, 0, 1);
        let empty_labels = Vec::<LineImpactLabel>::new();
        let datasets = [(&graph, empty_labels.as_slice())];
        let full_root = tempdir().unwrap();
        let split_root = tempdir().unwrap();
        let metadata = metadata();
        let (full, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                full_root.path(),
                None,
                &TrainingControl::new(None, None),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("full metric run did not complete")
        };
        let control_path = split_root.path().join("control.json");
        let mut pause = PauseOnEpoch {
            path: control_path.clone(),
            phase: "metric-learning",
            epoch: 3,
            requested: false,
        };
        let (_, ReferenceTrainingOutcome::Paused { checkpoint_path }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                split_root.path(),
                None,
                &TrainingControl::from_path(&control_path),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut pause,
            )
            .unwrap()
        else {
            panic!("metric pause did not produce a paused outcome")
        };
        write_control(&control_path, DesiredTrainingState::Running);
        let (resumed, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                split_root.path(),
                Some(&checkpoint_path),
                &TrainingControl::from_path(&control_path),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("resumed metric run did not complete")
        };
        assert_eq!(
            serde_json::to_vec(&full.checkpoint).unwrap(),
            serde_json::to_vec(&resumed.checkpoint).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&full.report).unwrap(),
            serde_json::to_vec(&resumed.report).unwrap()
        );
    }

    #[test]
    fn multitask_criticality_resume_restores_head_and_phase_report() {
        let graph = graph();
        let graph_labels = labels(&graph);
        let datasets = [(&graph, graph_labels.as_slice())];
        let config = multitask_config(1, 2, 5, 1);
        let full_root = tempdir().unwrap();
        let split_root = tempdir().unwrap();
        let metadata = metadata();
        let (full, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                full_root.path(),
                None,
                &TrainingControl::new(None, None),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("full criticality run did not complete")
        };
        let control_path = split_root.path().join("control.json");
        let mut pause = PauseOnEpoch {
            path: control_path.clone(),
            phase: "criticality",
            epoch: 3,
            requested: false,
        };
        let (_, ReferenceTrainingOutcome::Paused { checkpoint_path }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                split_root.path(),
                None,
                &TrainingControl::from_path(&control_path),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut pause,
            )
            .unwrap()
        else {
            panic!("criticality pause did not produce a paused outcome")
        };
        write_control(&control_path, DesiredTrainingState::Running);
        let (resumed, ReferenceTrainingOutcome::Completed { .. }) =
            run_reference_multitask_with_policy_options(
                &datasets,
                &config,
                split_root.path(),
                Some(&checkpoint_path),
                &TrainingControl::from_path(&control_path),
                CheckpointPolicy {
                    every_steps: Some(1),
                    every_seconds: None,
                },
                &metadata,
                false,
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("resumed criticality run did not complete")
        };
        assert_eq!(
            serde_json::to_vec(&full.checkpoint).unwrap(),
            serde_json::to_vec(&resumed.checkpoint).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&full.report).unwrap(),
            serde_json::to_vec(&resumed.report).unwrap()
        );
    }

    #[test]
    fn multitask_deadline_yields_after_a_metric_optimizer_boundary() {
        let graph = graph();
        let config = multitask_config(0, 3, 0, 1);
        let root = tempdir().unwrap();
        let empty_labels = Vec::<LineImpactLabel>::new();
        let datasets = [(&graph, empty_labels.as_slice())];
        let (result, outcome) = run_reference_multitask_with_policy_options(
            &datasets,
            &config,
            root.path(),
            None,
            &TrainingControl::with_policy(None, Some(Duration::ZERO), Some(Duration::ZERO)),
            CheckpointPolicy {
                every_steps: Some(1),
                every_seconds: None,
            },
            &metadata(),
            false,
            &mut NoopTrainingObserver,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            ReferenceTrainingOutcome::TimeSliceExpired { .. }
        ));
        assert!(result.checkpoint.representation.is_some());
        assert_eq!(result.checkpoint.encoder.config.graph_layers, 1);
        assert_eq!(result.report.metric_epochs, config.metric_epochs);
        assert_eq!(list_training_checkpoints(root.path()).unwrap().len(), 2);
    }

    #[test]
    fn gradient_accumulation_resume_matches_continuous_training() {
        let graph = graph();
        let full_config = config(6);
        let mut full_config = full_config;
        full_config.runtime.gradient_accumulation = 2;
        let first_config = PretrainingConfig {
            steps: 3,
            ..full_config.clone()
        };
        let full_root = tempdir().unwrap();
        let split_root = tempdir().unwrap();
        let control = TrainingControl::new(None, None);
        let (full, ReferenceTrainingOutcome::Completed { .. }) = run_reference_pretraining(
            &graph,
            &full_config,
            full_root.path(),
            None,
            &control,
            Some(1),
            &metadata(),
            &mut NoopTrainingObserver,
        )
        .unwrap() else {
            panic!("continuous accumulation run did not complete")
        };
        let (_, ReferenceTrainingOutcome::Completed { checkpoint_path }) =
            run_reference_pretraining(
                &graph,
                &first_config,
                split_root.path(),
                None,
                &control,
                Some(1),
                &metadata(),
                &mut NoopTrainingObserver,
            )
            .unwrap()
        else {
            panic!("first accumulation split did not complete")
        };
        let (resumed, ReferenceTrainingOutcome::Completed { .. }) = run_reference_pretraining(
            &graph,
            &full_config,
            split_root.path(),
            Some(&checkpoint_path),
            &control,
            Some(1),
            &metadata(),
            &mut NoopTrainingObserver,
        )
        .unwrap() else {
            panic!("resumed accumulation run did not complete")
        };
        assert_eq!(
            serde_json::to_vec(&full.model).unwrap(),
            serde_json::to_vec(&resumed.model).unwrap()
        );
        assert_eq!(full.cursor, resumed.cursor);
        assert_eq!(full.optimizer.step, resumed.optimizer.step);
    }
}
