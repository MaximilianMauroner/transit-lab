use crate::{ReferenceCheckpoint, TrainingReport};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use transit_model::DecoderGradients;

pub const TRAINING_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

static TEMPORARY_CHECKPOINT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckpointStatus {
    Committed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingCursor {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    pub phase: String,
    pub epoch: u64,
    pub batch: u64,
    #[serde(rename = "globalStep")]
    pub global_step: u64,
    #[serde(rename = "examplesSeen")]
    pub examples_seen: u64,
    #[serde(rename = "gradientAccumulationPosition")]
    pub gradient_accumulation_position: u64,
}

impl Default for TrainingCursor {
    fn default() -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            phase: "pretraining".into(),
            epoch: 0,
            batch: 0,
            global_step: 0,
            examples_seen: 0,
            gradient_accumulation_position: 0,
        }
    }
}

impl TrainingCursor {
    pub fn at_step(phase: impl Into<String>, step: u64) -> Self {
        Self {
            phase: phase.into(),
            global_step: step,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != checkpoint_schema_version() {
            bail!(
                "unsupported training cursor schema {}; expected {}",
                self.schema_version,
                checkpoint_schema_version()
            );
        }
        if self.phase.trim().is_empty() {
            bail!("training cursor phase cannot be blank");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerParameterGroup {
    pub name: String,
    #[serde(rename = "learningRate")]
    pub learning_rate: f64,
    #[serde(rename = "weightDecay")]
    pub weight_decay: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerState {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    pub kind: String,
    #[serde(rename = "step")]
    pub step: u64,
    #[serde(rename = "learningRate")]
    pub learning_rate: f64,
    #[serde(rename = "weightDecay")]
    pub weight_decay: f64,
    #[serde(rename = "parameterGroups")]
    pub parameter_groups: Vec<OptimizerParameterGroup>,
    /// Backend-specific state is kept in the payload file. The reference
    /// backend has no hidden momentum buffers; LibTorch can add tensor files
    /// without changing the top-level checkpoint contract.
    #[serde(default)]
    pub state: BTreeMap<String, serde_json::Value>,
}

impl OptimizerState {
    pub fn reference(step: u64, learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            kind: "reference-sgd".into(),
            step,
            learning_rate: f64::from(learning_rate),
            weight_decay: f64::from(weight_decay),
            parameter_groups: vec![OptimizerParameterGroup {
                name: "all".into(),
                learning_rate: f64::from(learning_rate),
                weight_decay: f64::from(weight_decay),
            }],
            state: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchedulerState {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    pub kind: String,
    #[serde(rename = "lastStep")]
    pub last_step: u64,
    #[serde(rename = "lastLearningRate")]
    pub last_learning_rate: f64,
    #[serde(default)]
    pub state: BTreeMap<String, serde_json::Value>,
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            kind: "constant".into(),
            last_step: 0,
            last_learning_rate: 0.0,
            state: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalerState {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    pub enabled: bool,
    pub scale: f64,
    #[serde(rename = "growthTracker")]
    pub growth_tracker: u64,
    #[serde(rename = "backoffFactor")]
    pub backoff_factor: f64,
    #[serde(rename = "growthInterval")]
    pub growth_interval: u64,
}

impl Default for ScalerState {
    fn default() -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            enabled: false,
            scale: 1.0,
            growth_tracker: 0,
            backoff_factor: 0.5,
            growth_interval: 2_000,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplerState {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    pub seed: u64,
    #[serde(rename = "graphOrder")]
    pub graph_order: Vec<String>,
    #[serde(rename = "currentGraph")]
    pub current_graph: usize,
    #[serde(rename = "currentExample")]
    pub current_example: usize,
    #[serde(rename = "permutationEpoch")]
    pub permutation_epoch: u64,
}

impl Default for SamplerState {
    fn default() -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            seed: 0,
            graph_order: Vec::new(),
            current_graph: 0,
            current_example: 0,
            permutation_epoch: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RngState {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    pub algorithm: String,
    #[serde(rename = "cpuSeed")]
    pub cpu_seed: u64,
    #[serde(rename = "gpuSeed")]
    pub gpu_seed: Option<u64>,
    #[serde(rename = "stateBytes", default)]
    pub state_bytes: Vec<u8>,
}

impl Default for RngState {
    fn default() -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            algorithm: "derived-seed-v1".into(),
            cpu_seed: 0,
            gpu_seed: None,
            state_bytes: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BestMetricState {
    #[serde(rename = "schemaVersion", default = "checkpoint_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub values: BTreeMap<String, f64>,
    #[serde(rename = "stepsWithoutImprovement")]
    pub steps_without_improvement: u64,
}

/// Progress metadata for the multi-task phases.  Model/head parameters live
/// in `TrainingCheckpointV1.model`; this small companion state makes it
/// possible to resume at the exact phase boundary without inferring progress
/// from a human-readable report.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MultiTaskPhaseState {
    #[serde(
        rename = "pretrainingReport",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub pretraining_report: Option<TrainingReport>,
    #[serde(
        rename = "metricInitialLoss",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub metric_initial_loss: Option<f32>,
    #[serde(
        rename = "metricFinalLoss",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub metric_final_loss: Option<f32>,
    #[serde(rename = "metricTriplets", default)]
    pub metric_triplets: usize,
    #[serde(
        rename = "criticalityReport",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub criticality_report: Option<TrainingReport>,
}

impl Default for BestMetricState {
    fn default() -> Self {
        Self {
            schema_version: checkpoint_schema_version(),
            values: BTreeMap::new(),
            steps_without_improvement: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointFile {
    pub path: String,
    #[serde(rename = "sha256")]
    pub sha256: String,
    #[serde(rename = "sizeBytes")]
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingCheckpointManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "runId")]
    pub run_id: String,
    #[serde(rename = "attemptId")]
    pub attempt_id: Option<String>,
    #[serde(rename = "globalStep")]
    pub global_step: u64,
    pub phase: String,
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
    pub status: CheckpointStatus,
    #[serde(rename = "checkpointFingerprint")]
    pub checkpoint_fingerprint: String,
    pub files: Vec<CheckpointFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingCheckpointV1 {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "runId")]
    pub run_id: String,
    #[serde(rename = "attemptId")]
    pub attempt_id: Option<String>,
    pub model: ReferenceCheckpoint,
    pub optimizer: OptimizerState,
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
    #[serde(default)]
    pub report: Option<TrainingReport>,
    #[serde(
        rename = "multiTaskPhase",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub multi_task_phase: Option<MultiTaskPhaseState>,
    /// Pending reference-backend gradients when a graph accumulation cycle
    /// spans more than one graph unit.  LibTorch checkpoints may omit this
    /// field and carry their backend-native optimizer state instead.
    #[serde(
        rename = "decoderGradients",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub decoder_gradients: Option<DecoderGradients>,
}

impl TrainingCheckpointV1 {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION {
            bail!(
                "unsupported training checkpoint schema {}; expected {}",
                self.schema_version,
                TRAINING_CHECKPOINT_SCHEMA_VERSION
            );
        }
        if self.run_id.trim().is_empty() {
            bail!("training checkpoint run ID cannot be blank");
        }
        if self.dataset_fingerprint.trim().is_empty() {
            bail!("training checkpoint dataset fingerprint cannot be blank");
        }
        if self.config_fingerprint.trim().is_empty() {
            bail!("training checkpoint config fingerprint cannot be blank");
        }
        if self.backend.trim().is_empty() || self.device_type.trim().is_empty() {
            bail!("training checkpoint backend and device type are required");
        }
        self.cursor.validate()?;
        if self.optimizer.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION
            || self.scheduler.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION
            || self.scaler.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION
            || self.sampler.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION
            || self.rng.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION
            || self.best_metrics.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION
        {
            bail!("training checkpoint contains an unsupported state schema");
        }
        if self.cursor.global_step != self.optimizer.step {
            bail!(
                "checkpoint cursor step {} does not match optimizer step {}",
                self.cursor.global_step,
                self.optimizer.step
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CheckpointCompatibility<'a> {
    pub run_id: Option<&'a str>,
    pub dataset_fingerprint: Option<&'a str>,
    pub config_fingerprint: Option<&'a str>,
    pub backend: Option<&'a str>,
    pub device_type: Option<&'a str>,
}

pub fn validate_checkpoint_compatibility(
    checkpoint: &TrainingCheckpointV1,
    expected: CheckpointCompatibility<'_>,
) -> Result<()> {
    checkpoint.validate()?;
    for (label, actual, wanted) in [
        ("run ID", checkpoint.run_id.as_str(), expected.run_id),
        (
            "dataset fingerprint",
            checkpoint.dataset_fingerprint.as_str(),
            expected.dataset_fingerprint,
        ),
        (
            "configuration fingerprint",
            checkpoint.config_fingerprint.as_str(),
            expected.config_fingerprint,
        ),
        ("backend", checkpoint.backend.as_str(), expected.backend),
        (
            "device type",
            checkpoint.device_type.as_str(),
            expected.device_type,
        ),
    ] {
        if let Some(wanted) = wanted {
            if actual != wanted {
                bail!("checkpoint {label} {actual:?} is incompatible with {wanted:?}");
            }
        }
    }
    Ok(())
}

pub fn checkpoint_schema_version() -> u32 {
    TRAINING_CHECKPOINT_SCHEMA_VERSION
}

pub fn save_training_checkpoint(root: &Path, checkpoint: &TrainingCheckpointV1) -> Result<PathBuf> {
    checkpoint.validate()?;
    fs::create_dir_all(root)
        .with_context(|| format!("creating checkpoint root {}", root.display()))?;
    let final_directory = root.join(format!("step-{:012}", checkpoint.cursor.global_step));
    if final_directory.exists() {
        let existing = load_training_checkpoint(&final_directory)?;
        if existing.0.checkpoint_fingerprint() == checkpoint.checkpoint_fingerprint() {
            return Ok(final_directory);
        }
        bail!(
            "checkpoint step {} already exists with different state",
            checkpoint.cursor.global_step
        );
    }

    let temporary_directory = root.join(format!(
        ".tmp-{}-{}-{}",
        std::process::id(),
        unix_timestamp_nanos(),
        TEMPORARY_CHECKPOINT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&temporary_directory).with_context(|| {
        format!(
            "creating temporary checkpoint {}",
            temporary_directory.display()
        )
    })?;

    let result = (|| {
        let payloads = checkpoint_payloads(checkpoint)?;
        let mut files = Vec::with_capacity(payloads.len());
        for (name, bytes) in payloads {
            let path = temporary_directory.join(name);
            write_synced_file(&path, &bytes)?;
            files.push(CheckpointFile {
                path: path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .context("checkpoint payload has a non-UTF-8 name")?
                    .into(),
                sha256: sha256_hex(&bytes),
                size_bytes: bytes.len() as u64,
            });
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));
        let checkpoint_fingerprint = fingerprint_files(&files);
        let manifest = TrainingCheckpointManifest {
            schema_version: TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: checkpoint.run_id.clone(),
            attempt_id: checkpoint.attempt_id.clone(),
            global_step: checkpoint.cursor.global_step,
            phase: checkpoint.cursor.phase.clone(),
            dataset_fingerprint: checkpoint.dataset_fingerprint.clone(),
            config_fingerprint: checkpoint.config_fingerprint.clone(),
            code_commit: checkpoint.code_commit.clone(),
            backend: checkpoint.backend.clone(),
            backend_version: checkpoint.backend_version.clone(),
            device_type: checkpoint.device_type.clone(),
            status: CheckpointStatus::Committed,
            checkpoint_fingerprint,
            files,
        };
        let manifest_bytes =
            serde_json::to_vec_pretty(&manifest).context("encoding checkpoint manifest")?;
        // The manifest is deliberately the last file written. A directory is
        // resumable only after this committed marker exists and its payload
        // hashes validate.
        write_synced_file(&temporary_directory.join("manifest.json"), &manifest_bytes)?;
        sync_directory(&temporary_directory)?;
        fs::rename(&temporary_directory, &final_directory).with_context(|| {
            format!(
                "committing checkpoint {} as {}",
                temporary_directory.display(),
                final_directory.display()
            )
        })?;
        sync_directory(root)?;
        write_latest_pointer(root, &final_directory, &manifest)?;
        Ok::<(), anyhow::Error>(())
    })();

    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary_directory);
    }
    result.map(|()| final_directory)
}

/// Returns the checkpoint and its committed manifest. The tuple keeps the
/// manifest available to the worker without making it re-read the directory.
pub fn load_training_checkpoint(
    path: &Path,
) -> Result<(TrainingCheckpointV1, TrainingCheckpointManifest)> {
    let directory = resolve_checkpoint_directory(path)?;
    let manifest_path = directory.join("manifest.json");
    let manifest: TrainingCheckpointManifest = read_json(&manifest_path)
        .with_context(|| format!("reading checkpoint manifest {}", manifest_path.display()))?;
    validate_manifest(&manifest)?;
    for file in &manifest.files {
        let payload_path = directory.join(&file.path);
        let bytes = fs::read(&payload_path)
            .with_context(|| format!("reading checkpoint payload {}", payload_path.display()))?;
        if bytes.len() as u64 != file.size_bytes || sha256_hex(&bytes) != file.sha256 {
            bail!("checkpoint payload hash or size mismatch for {}", file.path);
        }
    }
    let model: ReferenceCheckpoint = read_json(&directory.join("model.ot"))?;
    let optimizer: OptimizerState = read_json(&directory.join("optimizer.ot"))?;
    let scheduler: SchedulerState = read_json(&directory.join("scheduler.json"))?;
    let scaler: ScalerState = read_json(&directory.join("scaler.ot"))?;
    let rng: RngState = read_json(&directory.join("rng.ot"))?;
    let cursor: TrainingCursor = read_json(&directory.join("cursor.json"))?;
    let sampler: SamplerState = read_json(&directory.join("sampler.json"))?;
    let best_metrics: BestMetricState = read_json(&directory.join("best-metrics.json"))?;
    let decoder_gradients: Option<DecoderGradients> =
        if directory.join("decoder-gradients.json").is_file() {
            Some(read_json(&directory.join("decoder-gradients.json"))?)
        } else {
            None
        };
    let report: Option<TrainingReport> = if directory.join("report.json").is_file() {
        Some(read_json(&directory.join("report.json"))?)
    } else {
        None
    };
    let multi_task_phase: Option<MultiTaskPhaseState> =
        if directory.join("multi-task-phase.json").is_file() {
            Some(read_json(&directory.join("multi-task-phase.json"))?)
        } else {
            None
        };
    let checkpoint = TrainingCheckpointV1 {
        schema_version: manifest.schema_version,
        run_id: manifest.run_id.clone(),
        attempt_id: manifest.attempt_id.clone(),
        model,
        optimizer,
        scheduler,
        scaler,
        rng,
        cursor,
        sampler,
        best_metrics,
        dataset_fingerprint: manifest.dataset_fingerprint.clone(),
        config_fingerprint: manifest.config_fingerprint.clone(),
        code_commit: manifest.code_commit.clone(),
        backend: manifest.backend.clone(),
        backend_version: manifest.backend_version.clone(),
        device_type: manifest.device_type.clone(),
        report,
        decoder_gradients,
        multi_task_phase,
    };
    checkpoint.validate()?;
    // Validate the fingerprint against the bytes that were committed, not a
    // freshly re-encoded Rust value. JSON number round-trips can choose a
    // different but equivalent decimal spelling for an f32-derived metric;
    // the manifest must remain an integrity check for the actual files.
    if fingerprint_files(&manifest.files) != manifest.checkpoint_fingerprint {
        bail!("checkpoint manifest fingerprint does not match its payloads");
    }
    Ok((checkpoint, manifest))
}

pub fn load_latest_training_checkpoint(
    root: &Path,
) -> Result<(TrainingCheckpointV1, TrainingCheckpointManifest)> {
    load_training_checkpoint(root)
}

pub fn list_training_checkpoints(root: &Path) -> Result<Vec<PathBuf>> {
    let mut candidates = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading checkpoint root {}", root.display()))
        }
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("step-") || !entry.file_type()?.is_dir() {
            continue;
        }
        candidates.push(entry.path());
    }
    // Directory enumeration order is not stable across filesystems. Sort the
    // candidates before validating them so reconciliation and fallback from a
    // stale latest pointer always choose the highest committed training step.
    candidates.sort_by(|left, right| checkpoint_sort_key(left).cmp(&checkpoint_sort_key(right)));
    let mut checkpoints = Vec::new();
    for path in candidates {
        if load_training_checkpoint(&path).is_ok() {
            checkpoints.push(path);
        }
    }
    Ok(checkpoints)
}

impl TrainingCheckpointV1 {
    pub fn checkpoint_fingerprint(&self) -> String {
        let payloads = checkpoint_payloads(self).unwrap_or_default();
        let mut files = payloads
            .iter()
            .map(|(path, bytes)| CheckpointFile {
                path: (*path).into(),
                sha256: sha256_hex(bytes),
                size_bytes: bytes.len() as u64,
            })
            .collect::<Vec<_>>();
        files.sort_by(|left, right| left.path.cmp(&right.path));
        fingerprint_files(&files)
    }
}

fn checkpoint_payloads(checkpoint: &TrainingCheckpointV1) -> Result<Vec<(&'static str, Vec<u8>)>> {
    let mut payloads = vec![
        ("model.ot", encode_json(&checkpoint.model)?),
        ("optimizer.ot", encode_json(&checkpoint.optimizer)?),
        ("scheduler.json", encode_json(&checkpoint.scheduler)?),
        ("scaler.ot", encode_json(&checkpoint.scaler)?),
        ("rng.ot", encode_json(&checkpoint.rng)?),
        ("cursor.json", encode_json(&checkpoint.cursor)?),
        ("sampler.json", encode_json(&checkpoint.sampler)?),
        ("best-metrics.json", encode_json(&checkpoint.best_metrics)?),
    ];
    if let Some(gradients) = &checkpoint.decoder_gradients {
        payloads.push(("decoder-gradients.json", encode_json(gradients)?));
    }
    if let Some(report) = &checkpoint.report {
        payloads.push(("report.json", encode_json(report)?));
    }
    if let Some(phase) = &checkpoint.multi_task_phase {
        payloads.push(("multi-task-phase.json", encode_json(phase)?));
    }
    Ok(payloads)
}

fn encode_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(value).context("encoding checkpoint JSON")?)
}

fn validate_manifest(manifest: &TrainingCheckpointManifest) -> Result<()> {
    if manifest.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION {
        bail!(
            "unsupported checkpoint manifest schema {}; expected {}",
            manifest.schema_version,
            TRAINING_CHECKPOINT_SCHEMA_VERSION
        );
    }
    if manifest.status != CheckpointStatus::Committed {
        bail!("checkpoint manifest is not committed");
    }
    if manifest.files.is_empty() {
        bail!("checkpoint manifest has no payload files");
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
                || !file
                    .sha256
                    .chars()
                    .all(|character| character.is_ascii_hexdigit())
        })
    {
        bail!("checkpoint manifest contains invalid payload entries");
    }
    Ok(())
}

fn resolve_checkpoint_directory(path: &Path) -> Result<PathBuf> {
    if path.join("manifest.json").is_file() {
        return Ok(path.to_path_buf());
    }
    let latest_path = path.join("latest.json");
    if latest_path.is_file() {
        let pointer: LatestPointer = read_json(&latest_path)?;
        if pointer.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION {
            bail!("unsupported latest checkpoint pointer schema");
        }
        if valid_checkpoint_directory_name(&pointer.directory) {
            let directory = path.join(&pointer.directory);
            if directory.join("manifest.json").is_file()
                && load_training_checkpoint(&directory)
                    .ok()
                    .is_some_and(|(_, manifest)| {
                        manifest.global_step == pointer.global_step
                            && manifest.checkpoint_fingerprint == pointer.checkpoint_fingerprint
                    })
            {
                return Ok(directory);
            }
        }
    }
    let checkpoints = list_training_checkpoints(path)?;
    checkpoints
        .last()
        .cloned()
        .with_context(|| format!("no committed checkpoints found under {}", path.display()))
}

fn valid_checkpoint_directory_name(name: &str) -> bool {
    name.starts_with("step-")
        && name.len() > "step-".len()
        && name["step-".len()..]
            .chars()
            .all(|value| value.is_ascii_digit())
        && !name.contains('/')
        && !name.contains('\\')
}

fn checkpoint_sort_key(path: &Path) -> (u8, u64, String) {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let step = name
        .strip_prefix("step-")
        .and_then(|value| value.parse::<u64>().ok());
    match step {
        Some(step) => (0, step, name.to_owned()),
        None => (1, u64::MAX, name.to_owned()),
    }
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
        .context("checkpoint directory has a non-UTF-8 name")?;
    let pointer = LatestPointer {
        schema_version: TRAINING_CHECKPOINT_SCHEMA_VERSION,
        directory: name.into(),
        global_step: manifest.global_step,
        checkpoint_fingerprint: manifest.checkpoint_fingerprint.clone(),
    };
    let temporary = root.join(format!(
        ".latest-{}-{}",
        std::process::id(),
        TEMPORARY_CHECKPOINT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    write_synced_file(&temporary, &encode_json(&pointer)?)?;
    fs::rename(&temporary, root.join("latest.json"))
        .with_context(|| format!("updating latest checkpoint pointer in {}", root.display()))?;
    sync_directory(root)
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("creating checkpoint file {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing checkpoint file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing checkpoint file {}", path.display()))?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("opening directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("syncing directory {}", path.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("decoding {}", path.display()))
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
    let encoded = serde_json::to_vec(files).expect("checkpoint files are serializable");
    sha256_hex(&encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReferenceRelationalAutoencoder;
    use tempfile::tempdir;

    fn checkpoint() -> TrainingCheckpointV1 {
        let mut checkpoint = TrainingCheckpointV1 {
            schema_version: TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: "run-test".into(),
            attempt_id: Some("attempt-1".into()),
            model: ReferenceCheckpoint {
                encoder: ReferenceRelationalAutoencoder::new(Default::default()),
                head: None,
                report: None,
                representation: None,
                config_fingerprint: Some("config-test".into()),
                seed: Some(7),
                training_run_id: None,
                dataset_fingerprint: None,
                model_id: None,
            },
            optimizer: OptimizerState::reference(4, 0.001, 0.00001),
            scheduler: SchedulerState {
                last_step: 4,
                last_learning_rate: 0.001,
                ..SchedulerState::default()
            },
            scaler: ScalerState::default(),
            rng: RngState {
                cpu_seed: 7,
                ..RngState::default()
            },
            cursor: TrainingCursor::at_step("pretraining", 4),
            sampler: SamplerState {
                seed: 7,
                ..SamplerState::default()
            },
            best_metrics: BestMetricState::default(),
            dataset_fingerprint: "dataset-test".into(),
            config_fingerprint: "config-test".into(),
            code_commit: "test".into(),
            backend: "reference-cpu-decoder".into(),
            backend_version: "test".into(),
            device_type: "cpu".into(),
            report: None,
            decoder_gradients: None,
            multi_task_phase: None,
        };
        checkpoint.optimizer.step = checkpoint.cursor.global_step;
        checkpoint
    }

    #[test]
    fn checkpoint_round_trip_uses_a_committed_directory() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("checkpoints");
        let expected = checkpoint();
        let path = save_training_checkpoint(&root, &expected).unwrap();

        assert!(path.join("manifest.json").is_file());
        assert!(root.join("latest.json").is_file());
        assert!(list_training_checkpoints(&root).unwrap().len() == 1);
        let (actual, manifest) = load_training_checkpoint(&root).unwrap();
        assert_eq!(actual.cursor.global_step, 4);
        assert_eq!(manifest.status, CheckpointStatus::Committed);
        assert_eq!(
            manifest.checkpoint_fingerprint,
            expected.checkpoint_fingerprint()
        );
    }

    #[test]
    fn corrupt_payloads_are_rejected_and_temporary_directories_are_ignored() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("checkpoints");
        let path = save_training_checkpoint(&root, &checkpoint()).unwrap();
        fs::create_dir(root.join(".tmp-interrupted")).unwrap();
        fs::write(path.join("model.ot"), b"corrupt").unwrap();
        assert!(load_training_checkpoint(&root).is_err());
        assert!(list_training_checkpoints(&root).unwrap().is_empty());
    }

    #[test]
    fn latest_pointer_falls_back_to_the_newest_valid_committed_directory() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("checkpoints");
        let first = checkpoint();
        save_training_checkpoint(&root, &first).unwrap();
        let mut second = checkpoint();
        second.cursor.global_step = 8;
        second.optimizer.step = 8;
        save_training_checkpoint(&root, &second).unwrap();
        fs::write(root.join("step-000000000008/model.ot"), b"corrupt").unwrap();

        let (loaded, _) = load_training_checkpoint(&root).unwrap();
        assert_eq!(loaded.cursor.global_step, 4);
        assert_eq!(
            list_training_checkpoints(&root).unwrap(),
            vec![root.join("step-000000000004")]
        );
    }

    #[test]
    fn incompatible_checkpoint_metadata_is_rejected() {
        let checkpoint = checkpoint();
        let error = validate_checkpoint_compatibility(
            &checkpoint,
            CheckpointCompatibility {
                run_id: Some("another-run"),
                dataset_fingerprint: None,
                config_fingerprint: Some("config-test"),
                backend: Some("reference-cpu-decoder"),
                device_type: Some("cpu"),
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("run ID"));
    }
}

fn unix_timestamp_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}
