//! LibTorch implementation of the masked relational encoder.
//!
//! This module is intentionally feature-gated. Building the default workspace
//! does not require a system LibTorch installation, while Linux/NVIDIA users
//! can enable `tch-backend` and train the same graph schema on a GPU.
//!
//! Export boundary: [`TchRelationalAutoencoder::save`] and
//! [`TchRelationalAutoencoder::load`] persist only the [`tch::nn::VarStore`]
//! tensors. `tch` 0.18 does not expose the state held by its optimizer wrapper,
//! so these methods produce inference/model artifacts, not resumable training
//! checkpoints. The resumable LibTorch session in `transit-training` uses the
//! explicit [`TchOptimizer`] below, whose moments are serializable. The
//! capability declaration and the explicit optimizer-state error below still
//! describe the standard `tch::nn::Optimizer` wrapper until a supported bridge
//! exists. A C++ bridge cannot safely be added at this layer: the optimizer
//! pointer is hidden inside `tch::COptimizer`, and `torch-sys` only exposes
//! construction, parameter registration, hyperparameter setters, step, and
//! free operations for it.

use crate::{MaskSelection, ModelConfig, RepresentationConfig, CRITICALITY_OUTPUTS};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use tch::{nn, Device, Kind, Tensor};
use transit_domain::SERVICE_DAY_BINS;
use transit_graph::{FeatureMatrix, GraphTensor, EDGE_FEATURES};

/// Capabilities of checkpoint artifacts produced by this backend.
///
/// This is deliberately explicit rather than inferred from a file extension:
/// a `VarStore` file contains model tensors only. In particular, loading one
/// after rebuilding an Adam/AdamW optimizer starts that optimizer with empty
/// moment buffers and therefore cannot provide exact split-resume semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TchCheckpointCapabilities {
    /// Whether model parameters and buffers can be saved and loaded.
    pub model_weights: bool,
    /// Whether optimizer state can be saved and loaded.
    pub optimizer_state: bool,
    /// Whether a run can be split at an optimizer step and resumed exactly.
    pub exact_split_resume: bool,
}

impl TchCheckpointCapabilities {
    /// The capability available through the standard `tch` 0.18 optimizer
    /// wrapper; the resumable training session has a separate Rust optimizer.
    pub const WEIGHTS_ONLY: Self = Self {
        model_weights: true,
        optimizer_state: false,
        exact_split_resume: false,
    };
}

/// Capability declaration for the optional LibTorch backend.
pub const TCH_CHECKPOINT_CAPABILITIES: TchCheckpointCapabilities =
    TchCheckpointCapabilities::WEIGHTS_ONLY;

/// Returns the checkpoint capabilities without requiring a model instance.
pub const fn checkpoint_capabilities() -> TchCheckpointCapabilities {
    TCH_CHECKPOINT_CAPABILITIES
}

/// Schema for the JSON descriptor that accompanies a native LibTorch weights
/// file.  A VarStore archive does not contain enough information to rebuild
/// the module safely (feature widths and task-head dimensions are part of the
/// architecture), so production model artifacts carry this descriptor next
/// to `weights_path`.
pub const TCH_MODEL_ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TchModelArtifactMetadata {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub backend: String,
    #[serde(rename = "weightsPath")]
    pub weights_path: String,
    #[serde(rename = "deviceType", default)]
    pub device_type: String,
    #[serde(rename = "modelConfig")]
    pub model_config: ModelConfig,
    #[serde(rename = "representationConfig")]
    pub representation_config: RepresentationConfig,
    #[serde(rename = "stationFeatureWidth")]
    pub station_feature_width: usize,
    #[serde(rename = "lineFeatureWidth")]
    pub line_feature_width: usize,
    #[serde(rename = "graphSchemaVersion")]
    pub graph_schema_version: String,
    #[serde(rename = "snapshotIds", default)]
    pub snapshot_ids: Vec<String>,
    #[serde(rename = "modelId", default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(
        rename = "trainingRunId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub training_run_id: Option<String>,
    #[serde(
        rename = "datasetFingerprint",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub dataset_fingerprint: Option<String>,
    #[serde(
        rename = "configFingerprint",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub config_fingerprint: Option<String>,
    #[serde(rename = "supportedHeads", default)]
    pub supported_heads: Vec<String>,
    #[serde(rename = "embeddingDimensions", default)]
    pub embedding_dimensions: BTreeMap<String, usize>,
}

impl TchModelArtifactMetadata {
    pub fn for_graph(
        graph: &GraphTensor,
        model_config: ModelConfig,
        representation_config: RepresentationConfig,
        weights_path: impl Into<String>,
    ) -> Self {
        let embedding_dimensions = [
            ("base", representation_config.base_dimension),
            ("general", representation_config.general_dimension),
            ("role", representation_config.role_dimension),
            ("service", representation_config.service_dimension),
            ("geometry", representation_config.geometry_dimension),
            ("resilience", representation_config.resilience_dimension),
        ]
        .into_iter()
        .map(|(name, dimension)| (name.to_owned(), dimension))
        .collect();
        Self {
            schema_version: TCH_MODEL_ARTIFACT_SCHEMA_VERSION,
            backend: "libtorch".into(),
            weights_path: weights_path.into(),
            device_type: "cpu".into(),
            model_config,
            representation_config,
            station_feature_width: graph.station_features.cols,
            line_feature_width: graph.line_features.cols,
            graph_schema_version: graph.manifest.schema_version.clone(),
            snapshot_ids: Vec::new(),
            model_id: None,
            training_run_id: None,
            dataset_fingerprint: None,
            config_fingerprint: None,
            supported_heads: vec![
                "criticality".into(),
                "reconstruction".into(),
                "similarity-preview".into(),
            ],
            embedding_dimensions,
        }
    }

    pub fn validate_for_graph(&self, graph: &GraphTensor) -> Result<()> {
        if self.schema_version != TCH_MODEL_ARTIFACT_SCHEMA_VERSION {
            anyhow::bail!(
                "unsupported LibTorch model artifact schema {}; expected {}",
                self.schema_version,
                TCH_MODEL_ARTIFACT_SCHEMA_VERSION
            );
        }
        if self.backend != "libtorch" {
            anyhow::bail!("unsupported native model backend {}", self.backend);
        }
        validate_safe_relative_path(&self.weights_path)?;
        if self.station_feature_width != graph.station_features.cols
            || self.line_feature_width != graph.line_features.cols
        {
            anyhow::bail!(
                "LibTorch model feature schema does not match graph: station {} vs {}, line {} vs {}",
                self.station_feature_width,
                graph.station_features.cols,
                self.line_feature_width,
                graph.line_features.cols
            );
        }
        if self.graph_schema_version != graph.manifest.schema_version {
            anyhow::bail!(
                "LibTorch model graph schema {} does not match graph {}",
                self.graph_schema_version,
                graph.manifest.schema_version
            );
        }
        if self.model_config.hidden_dimension == 0
            || self.model_config.temporal_dimension == 0
            || self.model_config.graph_layers == 0
            || self.representation_config.base_dimension == 0
            || self.representation_config.general_dimension == 0
            || self.representation_config.role_dimension == 0
            || self.representation_config.service_dimension == 0
            || self.representation_config.geometry_dimension == 0
            || self.representation_config.resilience_dimension == 0
        {
            anyhow::bail!("LibTorch model artifact contains zero-sized architecture dimensions");
        }
        Ok(())
    }

    /// Resolve the sibling weights archive without allowing traversal or a
    /// symlink to escape the model-artifact directory.  The lexical check in
    /// [`Self::validate_for_graph`] protects the metadata contract; this
    /// filesystem check protects the actual load operation.
    pub fn resolve_weights_path(&self, metadata_path: &Path) -> Result<PathBuf> {
        validate_safe_relative_path(&self.weights_path)?;
        let parent = metadata_path.parent().unwrap_or_else(|| Path::new("."));
        let canonical_parent = fs::canonicalize(parent).with_context(|| {
            format!(
                "resolving native LibTorch model directory {}",
                parent.display()
            )
        })?;
        let candidate = parent.join(&self.weights_path);
        let canonical_candidate = fs::canonicalize(&candidate).with_context(|| {
            format!(
                "resolving native LibTorch model weights {}",
                candidate.display()
            )
        })?;
        if !canonical_candidate.starts_with(&canonical_parent) {
            anyhow::bail!(
                "native LibTorch model weights resolve outside {}",
                canonical_parent.display()
            );
        }
        if !canonical_candidate.is_file() {
            anyhow::bail!(
                "native LibTorch model weights are not a file at {}",
                canonical_candidate.display()
            );
        }
        Ok(canonical_candidate)
    }
}

fn validate_safe_relative_path(value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    let drive_prefixed = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    if value.is_empty()
        || value.trim() != value
        || value.starts_with('/')
        || value.starts_with('\\')
        || drive_prefixed
        || value.contains('\0')
        || value.split(['/', '\\']).any(|segment| {
            segment.is_empty() || segment == "." || segment == ".." || segment.contains(':')
        })
    {
        anyhow::bail!("LibTorch model weights path must be a safe relative path");
    }
    Ok(())
}

/// Stable error returned when a caller asks this backend to persist optimizer
/// state. Keeping this as a typed error prevents a missing bridge from being
/// mistaken for a successful weights-only save.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TchOptimizerCheckpointError {
    /// `tch` 0.18.1 does not expose the wrapped optimizer state.
    StateSerializationUnavailable,
}

impl Display for TchOptimizerCheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateSerializationUnavailable => formatter.write_str("tch 0.18.1 does not expose optimizer-state serialization; this backend supports model weights only and cannot provide exact split resume"),
        }
    }
}

impl std::error::Error for TchOptimizerCheckpointError {}

/// Attempt to save an optimizer state for a resumable LibTorch checkpoint.
///
/// This function intentionally fails. `tch::nn::Optimizer` contains a private
/// `COptimizer`, and `torch-sys` 0.18.1 exports no state save/load functions.
/// A caller must not silently continue with [`TchRelationalAutoencoder::save`]
/// because that would discard Adam/AdamW moments.
pub fn save_optimizer_state(_optimizer: &nn::Optimizer, _path: &std::path::Path) -> Result<()> {
    Err(TchOptimizerCheckpointError::StateSerializationUnavailable.into())
}

/// Attempt to load an optimizer state for a resumable LibTorch checkpoint.
///
/// See [`save_optimizer_state`] for why this is an explicit error rather than
/// a best-effort or model-only fallback.
pub fn load_optimizer_state(_optimizer: &mut nn::Optimizer, _path: &std::path::Path) -> Result<()> {
    Err(TchOptimizerCheckpointError::StateSerializationUnavailable.into())
}

/// Metadata for the optimizer state owned by [`TchOptimizer`].  The standard
/// `tch::nn::Optimizer` wrapper does not expose its C++ state dictionary, so a
/// resumable training session uses this small Rust-owned optimizer instead.
/// Tensor moments are stored in the companion `.ot` archive and this metadata
/// validates that the archive is being loaded into the same parameter layout.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TchOptimizerMetadata {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub kind: String,
    pub step: u64,
    #[serde(rename = "learningRate")]
    pub learning_rate: f64,
    #[serde(rename = "weightDecay")]
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    #[serde(rename = "parameterNames")]
    pub parameter_names: Vec<String>,
}

/// A resumable Adam/AdamW optimizer for LibTorch tensors.
///
/// This deliberately does not wrap `tch::nn::Optimizer`: that type owns a
/// private C++ optimizer pointer and `tch` 0.18 exposes no save/load methods
/// for its moment buffers. The parameters are shallow clones of the tensors in
/// a `VarStore`, so updates still mutate the model in place while the moment
/// tensors remain under explicit Rust ownership and can be serialized.
#[derive(Debug)]
pub struct TchOptimizer {
    parameters: Vec<TchOptimizerParameter>,
    step: u64,
    learning_rate: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    decoupled_weight_decay: bool,
    kind: String,
    device: Device,
}

#[derive(Debug)]
struct TchOptimizerParameter {
    name: String,
    tensor: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

impl TchOptimizer {
    /// Construct a resumable Adam optimizer over all trainable VarStore
    /// parameters. Parameter names are sorted to make checkpoint layout
    /// independent of HashMap iteration order.
    pub fn adam(var_store: &nn::VarStore, learning_rate: f64, weight_decay: f64) -> Result<Self> {
        Self::new(var_store, learning_rate, weight_decay, false)
    }

    /// Construct a resumable AdamW optimizer over all trainable VarStore
    /// parameters.
    pub fn adamw(var_store: &nn::VarStore, learning_rate: f64, weight_decay: f64) -> Result<Self> {
        Self::new(var_store, learning_rate, weight_decay, true)
    }

    fn new(
        var_store: &nn::VarStore,
        learning_rate: f64,
        weight_decay: f64,
        decoupled_weight_decay: bool,
    ) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            anyhow::bail!("LibTorch optimizer learning rate must be finite and non-negative");
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            anyhow::bail!("LibTorch optimizer weight decay must be finite and non-negative");
        }
        let mut variables = var_store
            .variables()
            .into_iter()
            .filter(|(_, tensor)| tensor.requires_grad())
            .collect::<Vec<_>>();
        variables.sort_by(|left, right| left.0.cmp(&right.0));
        if variables.is_empty() {
            anyhow::bail!("LibTorch optimizer cannot be built for an empty VarStore");
        }
        let device = var_store.device();
        let parameters = variables
            .into_iter()
            .map(|(name, tensor)| TchOptimizerParameter {
                exp_avg: tensor.zeros_like(),
                exp_avg_sq: tensor.zeros_like(),
                name,
                tensor,
            })
            .collect();
        Ok(Self {
            parameters,
            step: 0,
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decoupled_weight_decay,
            kind: if decoupled_weight_decay {
                "adamw".into()
            } else {
                "adam".into()
            },
            device,
        })
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            anyhow::bail!("LibTorch optimizer learning rate must be finite and non-negative");
        }
        self.learning_rate = learning_rate;
        Ok(())
    }

    pub fn metadata(&self) -> TchOptimizerMetadata {
        TchOptimizerMetadata {
            schema_version: 1,
            kind: self.kind.clone(),
            step: self.step,
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            parameter_names: self
                .parameters
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect(),
        }
    }

    /// Clear gradients on all parameters before the next accumulation cycle.
    pub fn zero_grad(&mut self) {
        for parameter in &mut self.parameters {
            parameter.tensor.zero_grad();
        }
    }

    /// Apply one Adam/AdamW update to the gradients currently attached to the
    /// model. The step is advanced only after all parameter updates succeed.
    pub fn step_gradients(&mut self) -> Result<()> {
        let next_step = self.step.saturating_add(1);
        let bias_correction1 = 1.0 - self.beta1.powf(next_step as f64);
        let bias_correction2 = 1.0 - self.beta2.powf(next_step as f64);
        if !(bias_correction1 > 0.0 && bias_correction2 > 0.0) {
            anyhow::bail!("LibTorch optimizer bias correction became non-positive");
        }
        let step_size = self.learning_rate / bias_correction1;
        let denominator_scale = bias_correction2.sqrt();

        tch::no_grad(|| -> Result<()> {
            for parameter in &mut self.parameters {
                let gradient = parameter.tensor.grad();
                if !gradient.defined() {
                    continue;
                }
                let gradient = gradient.detach();
                parameter.exp_avg =
                    &parameter.exp_avg * self.beta1 + &gradient * (1.0 - self.beta1);
                parameter.exp_avg_sq = &parameter.exp_avg_sq * self.beta2
                    + (&gradient * &gradient) * (1.0 - self.beta2);

                let denominator = parameter.exp_avg_sq.sqrt() / denominator_scale + self.epsilon;
                let mut update = &parameter.exp_avg / denominator;
                if !self.decoupled_weight_decay && self.weight_decay != 0.0 {
                    update = update + &parameter.tensor * self.weight_decay;
                }
                if self.decoupled_weight_decay && self.weight_decay != 0.0 {
                    let _ = parameter
                        .tensor
                        .f_mul_scalar_(1.0 - self.learning_rate * self.weight_decay)
                        .context("applying AdamW decoupled weight decay")?;
                }
                let _ = parameter
                    .tensor
                    .f_add_(&update.g_mul_scalar(-step_size))
                    .with_context(|| format!("updating LibTorch parameter {}", parameter.name))?;
            }
            Ok(())
        })?;
        self.step = next_step;
        Ok(())
    }

    /// Backward, update, and clear the gradients as one optimizer step.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.zero_grad();
        loss.backward();
        self.step_gradients()?;
        Ok(())
    }

    /// Save Adam/AdamW moments as a named tensor archive. Metadata is written
    /// by the caller next to this file so the directory checkpoint can hash
    /// both payloads atomically.
    pub fn save_state(&self, path: &Path) -> Result<()> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .with_context(|| format!("creating optimizer state directory {}", parent.display()))?;
        let temporary = path.with_extension(format!(
            "{}-tmp-{}",
            path.extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("ot"),
            std::process::id()
        ));
        let named = self
            .parameters
            .iter()
            .flat_map(|parameter| {
                [
                    (
                        format!("exp_avg.{}", parameter.name),
                        parameter.exp_avg.shallow_clone(),
                    ),
                    (
                        format!("exp_avg_sq.{}", parameter.name),
                        parameter.exp_avg_sq.shallow_clone(),
                    ),
                ]
            })
            .collect::<Vec<_>>();
        let named_refs = named
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &temporary)
            .with_context(|| format!("saving LibTorch optimizer state {}", path.display()))?;
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error).with_context(|| {
                format!("committing LibTorch optimizer state {}", path.display())
            });
        }
        Ok(())
    }

    /// Load and validate Adam/AdamW moments. The parameter names, optimizer
    /// kind and hyperparameters must match; runtime-only device placement may
    /// differ because tensors are loaded directly onto this optimizer's
    /// device.
    pub fn load_state(&mut self, path: &Path, metadata: &TchOptimizerMetadata) -> Result<()> {
        let expected = self.metadata();
        if metadata.schema_version != expected.schema_version
            || metadata.kind != expected.kind
            || metadata.parameter_names != expected.parameter_names
            || (metadata.beta1 - expected.beta1).abs() > f64::EPSILON
            || (metadata.beta2 - expected.beta2).abs() > f64::EPSILON
            || (metadata.epsilon - expected.epsilon).abs() > f64::EPSILON
            || (metadata.weight_decay - expected.weight_decay).abs() > f64::EPSILON
        {
            anyhow::bail!("LibTorch optimizer checkpoint metadata is incompatible with the model");
        }
        if !metadata.learning_rate.is_finite() || metadata.learning_rate < 0.0 {
            anyhow::bail!("LibTorch optimizer checkpoint learning rate is invalid");
        }
        let loaded = Tensor::load_multi_with_device(path, self.device)
            .with_context(|| format!("loading LibTorch optimizer state {}", path.display()))?;
        let mut tensors = HashMap::with_capacity(loaded.len());
        for (name, tensor) in loaded {
            if tensors.insert(name.clone(), tensor).is_some() {
                anyhow::bail!("LibTorch optimizer checkpoint contains duplicate tensor {name}");
            }
        }
        for parameter in &mut self.parameters {
            let average_name = format!("exp_avg.{}", parameter.name);
            let square_name = format!("exp_avg_sq.{}", parameter.name);
            let average = tensors
                .remove(&average_name)
                .with_context(|| format!("optimizer checkpoint is missing {average_name}"))?;
            let square = tensors
                .remove(&square_name)
                .with_context(|| format!("optimizer checkpoint is missing {square_name}"))?;
            if average.size() != parameter.tensor.size() || square.size() != parameter.tensor.size()
            {
                anyhow::bail!(
                    "optimizer checkpoint tensor shape does not match parameter {}",
                    parameter.name
                );
            }
            parameter.exp_avg = average;
            parameter.exp_avg_sq = square;
        }
        if !tensors.is_empty() {
            anyhow::bail!("LibTorch optimizer checkpoint contains unexpected tensors");
        }
        self.step = metadata.step;
        self.learning_rate = metadata.learning_rate;
        Ok(())
    }
}

#[derive(Debug)]
struct RelationalLayer {
    station_self: nn::Linear,
    line_self: nn::Linear,
    station_to_line: nn::Linear,
    line_to_station: nn::Linear,
    transfer: nn::Linear,
    transit_station: nn::Linear,
    transit_line: nn::Linear,
    transit_edge: nn::Linear,
    interchange: nn::Linear,
    station_norm: nn::LayerNorm,
    line_norm: nn::LayerNorm,
}

impl RelationalLayer {
    fn new(path: nn::Path<'_>, hidden: i64, edge_features: i64) -> Self {
        Self {
            station_self: nn::linear(&path / "station_self", hidden, hidden, Default::default()),
            line_self: nn::linear(&path / "line_self", hidden, hidden, Default::default()),
            station_to_line: nn::linear(
                &path / "station_to_line",
                hidden,
                hidden,
                Default::default(),
            ),
            line_to_station: nn::linear(
                &path / "line_to_station",
                hidden,
                hidden,
                Default::default(),
            ),
            transfer: nn::linear(&path / "transfer", hidden, hidden, Default::default()),
            transit_station: nn::linear(
                &path / "transit_station",
                hidden,
                hidden,
                Default::default(),
            ),
            transit_line: nn::linear(&path / "transit_line", hidden, hidden, Default::default()),
            transit_edge: nn::linear(
                &path / "transit_edge",
                edge_features,
                hidden,
                Default::default(),
            ),
            interchange: nn::linear(&path / "interchange", hidden, hidden, Default::default()),
            station_norm: nn::layer_norm(&path / "station_norm", vec![hidden], Default::default()),
            line_norm: nn::layer_norm(&path / "line_norm", vec![hidden], Default::default()),
        }
    }
}

#[derive(Debug)]
struct GraphIndices {
    serves_src: Tensor,
    serves_dst: Tensor,
    transit_src: Tensor,
    transit_dst: Tensor,
    transit_line: Tensor,
    transfer_src: Tensor,
    transfer_dst: Tensor,
    interchange_src: Tensor,
    interchange_dst: Tensor,
}

/// Immutable graph tensors prepared once for a particular LibTorch device.
/// The Rust `GraphTensor` remains the source-of-truth artifact; this view is a
/// process-local cache and is intentionally not serialized with model state.
#[derive(Debug)]
pub struct PreparedGraph {
    pub station_features: Tensor,
    pub line_features: Tensor,
    pub station_temporal: Tensor,
    pub line_temporal: Tensor,
    pub transit_features: Tensor,
    pub transit_temporal: Tensor,
    pub transfer_features: Tensor,
    pub serves_src: Tensor,
    pub serves_dst: Tensor,
    pub transit_src: Tensor,
    pub transit_dst: Tensor,
    pub transit_line: Tensor,
    pub transfer_src: Tensor,
    pub transfer_dst: Tensor,
    pub interchange_src: Tensor,
    pub interchange_dst: Tensor,
    pub pattern_offsets: Vec<u32>,
    pub pattern_stops: Tensor,
    pub pattern_lines: Tensor,
    pub pattern_trip_counts: Tensor,
    pub pattern_stop_features: Tensor,
    pub pattern_segment_features: Tensor,
    /// Padded sequence views used by the batched pattern encoder.  The flat
    /// arrays above remain part of the graph contract; these views are a
    /// process-local execution cache and avoid constructing one tiny tensor
    /// for every stop on every forward pass.
    pub pattern_token_stops: Tensor,
    pub pattern_token_stop_features: Tensor,
    pub pattern_token_segment_features: Tensor,
    pub pattern_token_mask: Tensor,
    pub pattern_max_length: usize,
    pub station_count: usize,
    pub line_count: usize,
    pub device: Device,
}

impl PreparedGraph {
    pub fn from_graph(graph: &GraphTensor, device: Device) -> Result<Self> {
        graph.validate()?;
        let (
            pattern_token_stops,
            pattern_token_stop_features,
            pattern_token_segment_features,
            pattern_token_mask,
            pattern_max_length,
        ) = padded_pattern_tensors(graph, device);
        Ok(Self {
            station_features: matrix_tensor(&graph.station_features, device)?,
            line_features: matrix_tensor(&graph.line_features, device)?,
            station_temporal: temporal_tensor(
                &graph.station_temporal,
                graph.manifest.station_count,
                device,
            )?,
            line_temporal: temporal_tensor(
                &graph.line_temporal,
                graph.manifest.line_count,
                device,
            )?,
            transit_features: matrix_tensor(&graph.transit_features, device)?,
            transit_temporal: matrix_tensor(&graph.transit_temporal, device)?,
            transfer_features: matrix_tensor(&graph.transfer_features, device)?,
            serves_src: index_tensor(&graph.serves_src, device),
            serves_dst: index_tensor(&graph.serves_dst, device),
            transit_src: index_tensor(&graph.transit_src, device),
            transit_dst: index_tensor(&graph.transit_dst, device),
            transit_line: index_tensor(&graph.transit_line, device),
            transfer_src: index_tensor(&graph.transfer_src, device),
            transfer_dst: index_tensor(&graph.transfer_dst, device),
            interchange_src: index_tensor(&graph.interchange_src, device),
            interchange_dst: index_tensor(&graph.interchange_dst, device),
            pattern_offsets: graph.pattern_offsets.clone(),
            pattern_stops: index_tensor(&graph.pattern_stops, device),
            pattern_lines: index_tensor(&graph.pattern_lines, device),
            pattern_trip_counts: Tensor::from_slice(
                &graph
                    .pattern_trip_counts
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>(),
            )
            .to_device(device),
            pattern_stop_features: matrix_tensor(&graph.pattern_stop_features, device)?,
            pattern_segment_features: matrix_tensor(&graph.pattern_segment_features, device)?,
            pattern_token_stops,
            pattern_token_stop_features,
            pattern_token_segment_features,
            pattern_token_mask,
            pattern_max_length,
            station_count: graph.manifest.station_count,
            line_count: graph.manifest.line_count,
            device,
        })
    }

    fn indices(&self) -> GraphIndices {
        GraphIndices {
            serves_src: self.serves_src.shallow_clone(),
            serves_dst: self.serves_dst.shallow_clone(),
            transit_src: self.transit_src.shallow_clone(),
            transit_dst: self.transit_dst.shallow_clone(),
            transit_line: self.transit_line.shallow_clone(),
            transfer_src: self.transfer_src.shallow_clone(),
            transfer_dst: self.transfer_dst.shallow_clone(),
            interchange_src: self.interchange_src.shallow_clone(),
            interchange_dst: self.interchange_dst.shallow_clone(),
        }
    }

    pub fn matches(&self, graph: &GraphTensor, device: Device) -> bool {
        self.device == device
            && self.station_count == graph.manifest.station_count
            && self.line_count == graph.manifest.line_count
            && self.serves_src.size().first().copied().unwrap_or(0) as usize
                == graph.serves_src.len()
            && self.transit_src.size().first().copied().unwrap_or(0) as usize
                == graph.transit_src.len()
            && self.transfer_src.size().first().copied().unwrap_or(0) as usize
                == graph.transfer_src.len()
            && self.pattern_lines.size().first().copied().unwrap_or(0) as usize
                == graph.manifest.pattern_count
            && self
                .pattern_token_stops
                .size()
                .first()
                .copied()
                .unwrap_or(0) as usize
                == graph.manifest.pattern_count
            && self.pattern_token_stops.size().get(1).copied().unwrap_or(0) as usize
                == self.pattern_max_length
            && self
                .pattern_token_stop_features
                .size()
                .first()
                .copied()
                .unwrap_or(0) as usize
                == graph.manifest.pattern_count
            && self
                .pattern_token_segment_features
                .size()
                .first()
                .copied()
                .unwrap_or(0) as usize
                == graph.manifest.pattern_count
    }
}

#[derive(Debug)]
pub struct TchEmbeddings {
    pub station: Tensor,
    pub line: Tensor,
    pub city: Tensor,
}

#[derive(Debug)]
pub struct TchReconstruction {
    pub embeddings: TchEmbeddings,
    pub station_features: Tensor,
    pub line_features: Tensor,
    pub served_by_logits: Tensor,
    pub transfer_logits: Tensor,
}

/// Outputs of the trainable task heads that sit on top of the shared graph
/// encoder.  The vectors returned by the metric heads are L2-normalized so
/// their dot product is a cosine similarity, while `criticality` contains the
/// six normalized regression targets used by the label contract.
#[derive(Debug)]
pub struct TchTaskOutputs {
    pub base: Tensor,
    pub general: Tensor,
    pub role: Tensor,
    pub service: Tensor,
    pub geometry: Tensor,
    pub resilience: Tensor,
    pub criticality: Tensor,
}

#[derive(Debug)]
pub struct TchRelationalAutoencoder {
    pub var_store: nn::VarStore,
    station_temporal_conv1: nn::Conv1D,
    station_temporal_conv2: nn::Conv1D,
    line_temporal_conv1: nn::Conv1D,
    line_temporal_conv2: nn::Conv1D,
    station_input: nn::Linear,
    line_input: nn::Linear,
    pattern_token: nn::Linear,
    pattern_update: nn::Linear,
    layers: Vec<RelationalLayer>,
    station_decoder: nn::Linear,
    line_decoder: nn::Linear,
    task_base: nn::Linear,
    task_role: nn::Linear,
    task_service: nn::Linear,
    task_geometry: nn::Linear,
    task_resilience: nn::Linear,
    task_general: nn::Linear,
    criticality_hidden: nn::Linear,
    criticality_output: nn::Linear,
    station_feature_width: usize,
    line_feature_width: usize,
    hidden_dimension: usize,
    representation_config: RepresentationConfig,
    device: Device,
}

impl TchRelationalAutoencoder {
    pub fn new(device: Device, graph: &GraphTensor, config: &ModelConfig) -> Self {
        Self::new_with_representation_config(
            device,
            graph,
            config,
            &RepresentationConfig::default(),
        )
    }

    /// Construct the encoder and all task-specific heads in one VarStore.
    /// Keeping the heads in the same store is important: one native weights
    /// artifact then contains the complete inference model, and the resumable
    /// Rust-owned optimizer can checkpoint one parameter layout for every
    /// training phase.
    pub fn new_with_representation_config(
        device: Device,
        graph: &GraphTensor,
        config: &ModelConfig,
        representation_config: &RepresentationConfig,
    ) -> Self {
        let var_store = nn::VarStore::new(device);
        let root = var_store.root();
        let temporal_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let temporal_width = config.temporal_dimension as i64;
        let hidden = config.hidden_dimension as i64;
        let station_temporal_conv1 = nn::conv1d(
            &root / "station_temporal_conv1",
            4,
            temporal_width,
            3,
            temporal_config,
        );
        let station_temporal_conv2 = nn::conv1d(
            &root / "station_temporal_conv2",
            temporal_width,
            temporal_width,
            3,
            temporal_config,
        );
        let line_temporal_conv1 = nn::conv1d(
            &root / "line_temporal_conv1",
            4,
            temporal_width,
            3,
            temporal_config,
        );
        let line_temporal_conv2 = nn::conv1d(
            &root / "line_temporal_conv2",
            temporal_width,
            temporal_width,
            3,
            temporal_config,
        );
        let station_input = nn::linear(
            &root / "station_input",
            graph.station_features.cols as i64 + temporal_width * 2,
            hidden,
            Default::default(),
        );
        let line_input = nn::linear(
            &root / "line_input",
            graph.line_features.cols as i64 + temporal_width * 2,
            hidden,
            Default::default(),
        );
        let pattern_token = nn::linear(
            &root / "pattern_token",
            hidden + 3 + EDGE_FEATURES as i64,
            hidden,
            Default::default(),
        );
        let pattern_update = nn::linear(
            &root / "pattern_update",
            hidden * 2,
            hidden,
            Default::default(),
        );
        let layers = (0..config.graph_layers)
            .map(|index| {
                RelationalLayer::new(
                    &root / format!("graph_layer_{index}"),
                    hidden,
                    EDGE_FEATURES as i64,
                )
            })
            .collect();
        let station_decoder = nn::linear(
            &root / "station_decoder",
            hidden,
            graph.station_features.cols as i64,
            Default::default(),
        );
        let line_decoder = nn::linear(
            &root / "line_decoder",
            hidden,
            graph.line_features.cols as i64,
            Default::default(),
        );
        let line_city_width = hidden * 2;
        let task_context_width = line_city_width + graph.line_features.cols as i64;
        let task_base = nn::linear(
            &root / "task_base",
            line_city_width,
            representation_config.base_dimension as i64,
            Default::default(),
        );
        let task_role = nn::linear(
            &root / "task_role",
            task_context_width,
            representation_config.role_dimension as i64,
            Default::default(),
        );
        let task_service = nn::linear(
            &root / "task_service",
            task_context_width,
            representation_config.service_dimension as i64,
            Default::default(),
        );
        let task_geometry = nn::linear(
            &root / "task_geometry",
            task_context_width,
            representation_config.geometry_dimension as i64,
            Default::default(),
        );
        let task_resilience = nn::linear(
            &root / "task_resilience",
            task_context_width,
            representation_config.resilience_dimension as i64,
            Default::default(),
        );
        let task_general = nn::linear(
            &root / "task_general",
            representation_config.base_dimension as i64
                + representation_config.role_dimension as i64
                + representation_config.service_dimension as i64
                + representation_config.geometry_dimension as i64
                + representation_config.resilience_dimension as i64,
            representation_config.general_dimension as i64,
            Default::default(),
        );
        let criticality_hidden = nn::linear(
            &root / "criticality_hidden",
            representation_config.base_dimension as i64 + hidden + graph.line_features.cols as i64,
            hidden,
            Default::default(),
        );
        let criticality_output = nn::linear(
            &root / "criticality_output",
            hidden,
            CRITICALITY_OUTPUTS as i64,
            Default::default(),
        );
        Self {
            var_store,
            station_temporal_conv1,
            station_temporal_conv2,
            line_temporal_conv1,
            line_temporal_conv2,
            station_input,
            line_input,
            pattern_token,
            pattern_update,
            layers,
            station_decoder,
            line_decoder,
            task_base,
            task_role,
            task_service,
            task_geometry,
            task_resilience,
            task_general,
            criticality_hidden,
            criticality_output,
            station_feature_width: graph.station_features.cols,
            line_feature_width: graph.line_features.cols,
            hidden_dimension: config.hidden_dimension,
            representation_config: representation_config.clone(),
            device,
        }
    }

    /// The representation-head dimensions are part of the model architecture
    /// and are persisted in the resolved experiment configuration. Exposing a
    /// copy lets exporters and callers validate an artifact before inference.
    pub fn representation_config(&self) -> &RepresentationConfig {
        &self.representation_config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn hidden_dimension(&self) -> usize {
        self.hidden_dimension
    }

    pub fn forward(
        &self,
        graph: &GraphTensor,
        mask: &MaskSelection,
        train: bool,
    ) -> Result<TchReconstruction> {
        let prepared = self.prepare_graph(graph)?;
        self.forward_prepared(graph, &prepared, mask, train)
    }

    pub fn prepare_graph(&self, graph: &GraphTensor) -> Result<PreparedGraph> {
        if graph.station_features.cols != self.station_feature_width
            || graph.line_features.cols != self.line_feature_width
        {
            anyhow::bail!(
                "graph feature schema does not match the LibTorch model: station {} vs {}, line {} vs {}",
                graph.station_features.cols,
                self.station_feature_width,
                graph.line_features.cols,
                self.line_feature_width
            );
        }
        PreparedGraph::from_graph(graph, self.device)
    }

    pub fn forward_prepared(
        &self,
        graph: &GraphTensor,
        prepared: &PreparedGraph,
        mask: &MaskSelection,
        train: bool,
    ) -> Result<TchReconstruction> {
        graph.validate()?;
        self.forward_prepared_unchecked(graph, prepared, mask, train)
    }

    /// Forward through a graph whose immutable Rust representation has already
    /// been validated by the caller.  Training uses this after preparing each
    /// graph once; keeping the public checked method above prevents callers
    /// from accidentally bypassing the graph contract.
    pub fn forward_prepared_unchecked(
        &self,
        graph: &GraphTensor,
        prepared: &PreparedGraph,
        mask: &MaskSelection,
        train: bool,
    ) -> Result<TchReconstruction> {
        if graph.station_features.cols != self.station_feature_width
            || graph.line_features.cols != self.line_feature_width
        {
            anyhow::bail!(
                "graph feature schema does not match the LibTorch model: station {} vs {}, line {} vs {}",
                graph.station_features.cols,
                self.station_feature_width,
                graph.line_features.cols,
                self.line_feature_width
            );
        }
        validate_mask(graph, mask)?;
        if !prepared.matches(graph, self.device) {
            anyhow::bail!("prepared graph does not match the LibTorch graph or device");
        }
        let indices = prepared.indices();
        let station_static = prepared.station_features.shallow_clone()
            * visible_rows(&mask.station_rows, self.device);
        let line_static =
            prepared.line_features.shallow_clone() * visible_rows(&mask.line_rows, self.device);
        let station_temporal = prepared.station_temporal.shallow_clone()
            * visible_temporal(
                &mask.station_temporal_blocks,
                graph.manifest.station_count,
                self.device,
            );
        let line_temporal = prepared.line_temporal.shallow_clone()
            * visible_temporal(
                &mask.line_temporal_blocks,
                graph.manifest.line_count,
                self.device,
            );
        let station_temporal = station_temporal
            .apply_t(&self.station_temporal_conv1, train)
            .gelu("none")
            .apply_t(&self.station_temporal_conv2, train)
            .gelu("none");
        let line_temporal = line_temporal
            .apply_t(&self.line_temporal_conv1, train)
            .gelu("none")
            .apply_t(&self.line_temporal_conv2, train)
            .gelu("none");
        let mut station = Tensor::cat(&[station_static, pool_temporal(&station_temporal)], 1)
            .apply(&self.station_input)
            .gelu("none");
        let mut line = Tensor::cat(&[line_static, pool_temporal(&line_temporal)], 1)
            .apply(&self.line_input)
            .gelu("none");
        let pattern_context = pattern_sequence_context(
            prepared,
            &station,
            &self.pattern_token,
            &self.pattern_update,
            self.device,
        )?;
        line = (line + pattern_context * 0.22).gelu("none");
        let served_mask = edge_visibility(&mask.served_edges, self.device);
        let transfer_mask = edge_visibility(&mask.transfer_edges, self.device);
        let transit_mask =
            Tensor::ones([graph.transit_src.len() as i64], (Kind::Float, self.device));
        let interchange_mask = Tensor::ones(
            [graph.interchange_src.len() as i64],
            (Kind::Float, self.device),
        );

        for layer in &self.layers {
            let station_to_line = mean_aggregate(
                &station,
                &indices.serves_src,
                &indices.serves_dst,
                graph.manifest.line_count,
                &layer.station_to_line,
                Some(&served_mask),
            );
            let line_to_station = mean_aggregate(
                &line,
                &indices.serves_dst,
                &indices.serves_src,
                graph.manifest.station_count,
                &layer.line_to_station,
                Some(&served_mask),
            );
            let transfer = mean_aggregate(
                &station,
                &indices.transfer_src,
                &indices.transfer_dst,
                graph.manifest.station_count,
                &layer.transfer,
                Some(&transfer_mask),
            );
            let transit = transit_aggregate(TransitAggregation {
                station: &station,
                line: &line,
                edge_features: &prepared.transit_features,
                source_indices: &indices.transit_src,
                destination_indices: &indices.transit_dst,
                line_indices: &indices.transit_line,
                destination_count: graph.manifest.station_count,
                station_projection: &layer.transit_station,
                line_projection: &layer.transit_line,
                edge_projection: &layer.transit_edge,
                visibility: Some(&transit_mask),
                device: self.device,
            })?;
            let interchange = mean_aggregate(
                &line,
                &indices.interchange_src,
                &indices.interchange_dst,
                graph.manifest.line_count,
                &layer.interchange,
                Some(&interchange_mask),
            );
            station = (station.apply(&layer.station_self)
                + line_to_station
                + transfer * 0.12
                + transit * 0.16)
                .gelu("none")
                .apply(&layer.station_norm);
            line = (line.apply(&layer.line_self) + station_to_line + interchange * 0.14)
                .gelu("none")
                .apply(&layer.line_norm);
        }
        let city = city_pool(&station, &line);
        let station_features = station.apply(&self.station_decoder);
        let line_features = line.apply(&self.line_decoder);
        let served_by_logits = dot_rows(&station, &line, &indices.serves_src, &indices.serves_dst);
        let transfer_logits = dot_rows(
            &station,
            &station,
            &indices.transfer_src,
            &indices.transfer_dst,
        );
        Ok(TchReconstruction {
            embeddings: TchEmbeddings {
                station,
                line,
                city,
            },
            station_features,
            line_features,
            served_by_logits,
            transfer_logits,
        })
    }

    /// Apply the task heads to a clean encoder output and a prepared line
    /// feature matrix. `line_features` must have one row per encoded line and
    /// must live on the same device as the embeddings.
    pub fn task_outputs(
        &self,
        embeddings: &TchEmbeddings,
        line_features: &Tensor,
        train: bool,
    ) -> Result<TchTaskOutputs> {
        let line_count = embeddings.line.size().first().copied().unwrap_or(0);
        let line_width = line_features.size().get(1).copied().unwrap_or(0);
        if line_count == 0
            || embeddings.city.size().len() != 1
            || line_width != self.line_feature_width as i64
            || line_features.size().first().copied().unwrap_or(0) != line_count
        {
            anyhow::bail!("LibTorch task-head inputs do not match the graph schema");
        }
        if line_features.device() != self.device
            || embeddings.line.device() != self.device
            || embeddings.city.device() != self.device
        {
            anyhow::bail!("LibTorch task-head inputs are on the wrong device");
        }
        let city = embeddings.city.unsqueeze(0).expand([line_count, -1], false);
        let line_city = Tensor::cat(&[embeddings.line.shallow_clone(), city.shallow_clone()], 1);
        let context = Tensor::cat(
            &[line_city.shallow_clone(), line_features.shallow_clone()],
            1,
        );
        let base = normalize_rows(line_city.apply_t(&self.task_base, train).gelu("none"));
        let role = normalize_rows(context.apply_t(&self.task_role, train).gelu("none"));
        let service = normalize_rows(context.apply_t(&self.task_service, train).gelu("none"));
        let geometry = normalize_rows(context.apply_t(&self.task_geometry, train).gelu("none"));
        let resilience = normalize_rows(context.apply_t(&self.task_resilience, train).gelu("none"));
        let general_input = Tensor::cat(
            &[
                base.shallow_clone(),
                role.shallow_clone(),
                service.shallow_clone(),
                geometry.shallow_clone(),
                resilience.shallow_clone(),
            ],
            1,
        );
        let general = normalize_rows(
            general_input
                .apply_t(&self.task_general, train)
                .gelu("none"),
        );
        let criticality_input = Tensor::cat(
            &[base.shallow_clone(), city, line_features.shallow_clone()],
            1,
        );
        let criticality = criticality_input
            .apply_t(&self.criticality_hidden, train)
            .gelu("none")
            .apply_t(&self.criticality_output, train);
        Ok(TchTaskOutputs {
            base,
            general,
            role,
            service,
            geometry,
            resilience,
            criticality,
        })
    }

    /// Save model parameters and buffers.
    ///
    /// This writes a weights-only `VarStore` artifact. It does not save any
    /// optimizer, scheduler, sampler, RNG, or training-cursor state and must
    /// not be advertised or consumed as an exact-resume checkpoint. Use
    /// [`save_optimizer_state`] only to detect the currently unsupported
    /// optimizer portion of such a checkpoint.
    pub fn save_weights(&self, path: &std::path::Path) -> Result<()> {
        self.var_store
            .save(path)
            .with_context(|| format!("saving LibTorch model weights {}", path.display()))?;
        Ok(())
    }

    /// Load model parameters and buffers from a weights-only artifact.
    ///
    /// Loading this file does not restore optimizer state and therefore does
    /// not resume an Adam/AdamW training trajectory exactly.
    pub fn load_weights(&mut self, path: &std::path::Path) -> Result<()> {
        // `load_partial` keeps old pre-task LibTorch weights usable: task
        // heads are newly initialized when an older encoder artifact is
        // promoted to a multi-task run. The resumable checkpoint loader still
        // validates its complete manifest and optimizer parameter names.
        let missing = self
            .var_store
            .load_partial(path)
            .with_context(|| format!("loading LibTorch model weights {}", path.display()))?;
        let unexpected_task_missing = missing.iter().any(|name| {
            !name.starts_with("task_")
                && !name.starts_with("criticality_hidden.")
                && !name.starts_with("criticality_output.")
        });
        if unexpected_task_missing {
            anyhow::bail!(
                "LibTorch model weights {} are missing required encoder parameters: {}",
                path.display(),
                missing.join(", ")
            );
        }
        Ok(())
    }

    /// Compatibility alias for [`Self::save_weights`].
    ///
    /// The alias is retained for existing callers, but the result is still a
    /// weights-only artifact; it is not a complete training checkpoint.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.save_weights(path)
    }

    /// Compatibility alias for [`Self::load_weights`].
    ///
    /// The alias restores model tensors only and never claims exact training
    /// resume.
    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        self.load_weights(path)
    }
}

fn normalize_rows(value: Tensor) -> Tensor {
    let norm = (&value * &value)
        .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
        .sqrt()
        .clamp_min(1.0e-8);
    value / norm
}

fn matrix_tensor(matrix: &FeatureMatrix, device: Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(&matrix.values)
        .to_device(device)
        .reshape([matrix.rows as i64, matrix.cols as i64]))
}

fn pattern_sequence_context(
    prepared: &PreparedGraph,
    station: &Tensor,
    token_projection: &nn::Linear,
    recurrent_update: &nn::Linear,
    device: Device,
) -> Result<Tensor> {
    let hidden = station.size().get(1).copied().unwrap_or(0);
    let pattern_count = prepared.pattern_lines.size().first().copied().unwrap_or(0);
    let max_length = prepared.pattern_max_length;
    if pattern_count == 0 || max_length == 0 {
        return Ok(Tensor::zeros(
            [prepared.line_count as i64, hidden],
            (Kind::Float, device),
        ));
    }

    // Gather every station occurrence in one operation and project all token
    // features in one batched linear call.  The only loop left is over the
    // padded sequence positions, which preserves the recurrent update while
    // eliminating both per-pattern and per-stop tensor construction.
    let flat_stops = prepared.pattern_token_stops.reshape([-1]);
    let station_tokens =
        station
            .index_select(0, &flat_stops)
            .reshape([pattern_count, max_length as i64, hidden]);
    let token_inputs = Tensor::cat(
        &[
            station_tokens,
            prepared.pattern_token_stop_features.shallow_clone(),
            prepared.pattern_token_segment_features.shallow_clone(),
        ],
        2,
    );
    let tokens = token_inputs.apply(token_projection).gelu("none");
    let mut state = Tensor::zeros([pattern_count, hidden], (Kind::Float, device));
    for position in 0..max_length {
        let token = tokens.select(1, position as i64);
        let candidate = Tensor::cat(&[state.shallow_clone(), token], 1)
            .apply(recurrent_update)
            .gelu("none");
        let active = prepared
            .pattern_token_mask
            .select(1, position as i64)
            .unsqueeze(1);
        state = &candidate * &active + &state * (1.0 - &active);
    }

    let weights = prepared.pattern_trip_counts.unsqueeze(1).clamp_min(1.0);
    let output = Tensor::zeros([prepared.line_count as i64, hidden], (Kind::Float, device))
        .index_add(0, &prepared.pattern_lines, &(state * &weights));
    let counts = Tensor::zeros([prepared.line_count as i64, 1], (Kind::Float, device)).index_add(
        0,
        &prepared.pattern_lines,
        &weights,
    );
    Ok(output / counts.clamp_min(1.0))
}

/// Build padded, device-resident pattern tensors once while preparing a graph.
/// The final stop in each pattern receives a zero segment feature, matching
/// the previous scalar implementation. Empty patterns remain masked out.
fn padded_pattern_tensors(
    graph: &GraphTensor,
    device: Device,
) -> (Tensor, Tensor, Tensor, Tensor, usize) {
    let pattern_count = graph.manifest.pattern_count;
    let max_length = graph
        .pattern_offsets
        .windows(2)
        .map(|window| window[1].saturating_sub(window[0]) as usize)
        .max()
        .unwrap_or(0);
    let token_count = pattern_count.saturating_mul(max_length);
    let mut stops = vec![0_i64; token_count];
    let mut stop_features = vec![0.0_f32; token_count.saturating_mul(3)];
    let mut segment_features = vec![0.0_f32; token_count.saturating_mul(EDGE_FEATURES)];
    let mut mask = vec![0.0_f32; token_count];
    let mut segment_offset = 0_usize;
    for pattern in 0..pattern_count {
        let start = graph.pattern_offsets[pattern] as usize;
        let end = graph.pattern_offsets[pattern + 1] as usize;
        let length = end.saturating_sub(start);
        for local_position in 0..length {
            let padded_position = pattern * max_length + local_position;
            stops[padded_position] = i64::from(graph.pattern_stops[start + local_position]);
            stop_features[padded_position * 3..padded_position * 3 + 3]
                .copy_from_slice(graph.pattern_stop_features.row(start + local_position));
            mask[padded_position] = 1.0;
            if local_position + 1 < length {
                let segment_row = segment_offset + local_position;
                segment_features[padded_position * EDGE_FEATURES
                    ..padded_position * EDGE_FEATURES + EDGE_FEATURES]
                    .copy_from_slice(graph.pattern_segment_features.row(segment_row));
            }
        }
        segment_offset = segment_offset.saturating_add(length.saturating_sub(1));
    }

    (
        Tensor::from_slice(&stops)
            .to_device(device)
            .reshape([pattern_count as i64, max_length as i64]),
        Tensor::from_slice(&stop_features)
            .to_device(device)
            .reshape([pattern_count as i64, max_length as i64, 3]),
        Tensor::from_slice(&segment_features)
            .to_device(device)
            .reshape([
                pattern_count as i64,
                max_length as i64,
                EDGE_FEATURES as i64,
            ]),
        Tensor::from_slice(&mask)
            .to_device(device)
            .reshape([pattern_count as i64, max_length as i64]),
        max_length,
    )
}

fn temporal_tensor(matrix: &FeatureMatrix, rows: usize, device: Device) -> Result<Tensor> {
    if matrix.cols != SERVICE_DAY_BINS * 4 {
        anyhow::bail!(
            "temporal matrix has {} columns, expected {}",
            matrix.cols,
            SERVICE_DAY_BINS * 4
        );
    }
    Ok(Tensor::from_slice(&matrix.values)
        .to_device(device)
        .reshape([rows as i64, 4, SERVICE_DAY_BINS as i64]))
}

fn pool_temporal(value: &Tensor) -> Tensor {
    let mean = value.mean_dim(&[2_i64][..], false, Kind::Float);
    let max = value.max_dim(2, false).0;
    Tensor::cat(&[mean, max], 1)
}

fn index_tensor(values: &[u32], device: Device) -> Tensor {
    Tensor::from_slice(
        &values
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}

fn visible_rows(rows: &[bool], device: Device) -> Tensor {
    Tensor::from_slice(
        &rows
            .iter()
            .map(|masked| if *masked { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
    .unsqueeze(1)
}

fn visible_temporal(blocks: &[bool], rows: usize, device: Device) -> Tensor {
    Tensor::from_slice(
        &blocks
            .iter()
            .map(|masked| if *masked { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
    .reshape([rows as i64, SERVICE_DAY_BINS as i64])
    .unsqueeze(1)
}

fn edge_visibility(mask: &[bool], device: Device) -> Tensor {
    Tensor::from_slice(
        &mask
            .iter()
            .map(|masked| if *masked { 0.0_f32 } else { 1.0 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}

fn mean_aggregate(
    source: &Tensor,
    source_indices: &Tensor,
    destination_indices: &Tensor,
    destination_count: usize,
    projection: &nn::Linear,
    visibility: Option<&Tensor>,
) -> Tensor {
    let mut messages = source.index_select(0, source_indices).apply(projection);
    let mut weights = Tensor::ones(
        [destination_indices.size()[0]],
        (Kind::Float, source.device()),
    );
    if let Some(visibility) = visibility {
        messages *= visibility.unsqueeze(1);
        weights = visibility.shallow_clone();
    }
    let hidden = messages.size()[1];
    let sums = Tensor::zeros(
        [destination_count as i64, hidden],
        (Kind::Float, source.device()),
    )
    .index_add(0, destination_indices, &messages);
    let counts = Tensor::zeros(
        [destination_count as i64, 1],
        (Kind::Float, source.device()),
    )
    .index_add(0, destination_indices, &weights.unsqueeze(1))
    .clamp_min(1.0);
    sums / counts
}

struct TransitAggregation<'a> {
    station: &'a Tensor,
    line: &'a Tensor,
    edge_features: &'a Tensor,
    source_indices: &'a Tensor,
    destination_indices: &'a Tensor,
    line_indices: &'a Tensor,
    destination_count: usize,
    station_projection: &'a nn::Linear,
    line_projection: &'a nn::Linear,
    edge_projection: &'a nn::Linear,
    visibility: Option<&'a Tensor>,
    device: Device,
}

fn transit_aggregate(args: TransitAggregation<'_>) -> Result<Tensor> {
    let mut messages = args
        .station
        .index_select(0, args.source_indices)
        .apply(args.station_projection)
        + args
            .line
            .index_select(0, args.line_indices)
            .apply(args.line_projection)
        + args.edge_features.apply(args.edge_projection);
    let mut weights = Tensor::ones(
        [args.destination_indices.size()[0]],
        (Kind::Float, args.device),
    );
    if let Some(visibility) = args.visibility {
        messages *= visibility.unsqueeze(1);
        weights = visibility.shallow_clone();
    }
    let hidden = messages.size()[1];
    let sums = Tensor::zeros(
        [args.destination_count as i64, hidden],
        (Kind::Float, args.device),
    )
    .index_add(0, args.destination_indices, &messages);
    let counts = Tensor::zeros(
        [args.destination_count as i64, 1],
        (Kind::Float, args.device),
    )
    .index_add(0, args.destination_indices, &weights.unsqueeze(1))
    .clamp_min(1.0);
    Ok(sums / counts)
}

fn dot_rows(
    left: &Tensor,
    right: &Tensor,
    left_indices: &Tensor,
    right_indices: &Tensor,
) -> Tensor {
    (left.index_select(0, left_indices) * right.index_select(0, right_indices)).sum_dim_intlist(
        &[1_i64][..],
        false,
        Kind::Float,
    )
}

fn city_pool(station: &Tensor, line: &Tensor) -> Tensor {
    let station_mean = station.mean_dim(&[0_i64][..], false, Kind::Float);
    let station_max = station.max_dim(0, false).0;
    let line_mean = line.mean_dim(&[0_i64][..], false, Kind::Float);
    let line_max = line.max_dim(0, false).0;
    (station_mean + station_max + line_mean + line_max) / 4.0
}

fn validate_mask(graph: &GraphTensor, mask: &MaskSelection) -> Result<()> {
    if mask.station_rows.len() != graph.manifest.station_count
        || mask.line_rows.len() != graph.manifest.line_count
        || mask.station_temporal_blocks.len() != graph.manifest.station_count * SERVICE_DAY_BINS
        || mask.line_temporal_blocks.len() != graph.manifest.line_count * SERVICE_DAY_BINS
        || mask.served_edges.len() != graph.serves_src.len()
        || mask.transfer_edges.len() != graph.transfer_src.len()
    {
        anyhow::bail!("mask shape does not match graph shape");
    }
    Ok(())
}

#[cfg(test)]
mod checkpoint_contract_tests {
    use super::*;

    #[test]
    fn checkpoint_capabilities_are_weights_only() {
        assert_eq!(
            checkpoint_capabilities(),
            TchCheckpointCapabilities::WEIGHTS_ONLY
        );
        assert!(TCH_CHECKPOINT_CAPABILITIES.model_weights);
        assert!(!TCH_CHECKPOINT_CAPABILITIES.optimizer_state);
        assert!(!TCH_CHECKPOINT_CAPABILITIES.exact_split_resume);
    }

    #[test]
    fn native_weights_path_validation_is_cross_platform() {
        for path in ["model.weights.ot", "nested/model.weights.ot"] {
            assert!(validate_safe_relative_path(path).is_ok(), "{path}");
        }
        for path in [
            "../model.weights.ot",
            "nested/../model.weights.ot",
            "nested\\..\\model.weights.ot",
            "C:\\models\\model.weights.ot",
            "/tmp/model.weights.ot",
            "\\\\server\\share\\model.weights.ot",
            "nested/./model.weights.ot",
            "nested//model.weights.ot",
            "model.weights.ot:stream",
        ] {
            assert!(validate_safe_relative_path(path).is_err(), "{path}");
        }
    }

    #[cfg(not(feature = "tch-doc-only"))]
    mod real_libtorch {
        use super::*;
        use std::io::Cursor;
        use std::path::Path;
        use tch::nn::OptimizerConfig;
        use tch::{nn, Device, Kind, Tensor};

        fn scalar_adam() -> (nn::VarStore, Tensor, nn::Optimizer) {
            let var_store = nn::VarStore::new(Device::Cpu);
            let weight = var_store.root().var("weight", &[1], nn::Init::Const(0.0));
            let optimizer = nn::Adam::default()
                .build(&var_store, 0.1)
                .expect("building test Adam optimizer");
            (var_store, weight, optimizer)
        }

        fn adam_step(weight: &Tensor, optimizer: &mut nn::Optimizer, target: f32) {
            let target = Tensor::from_slice(&[target]);
            let error = weight - target;
            let loss = (&error * &error).mean(Kind::Float);
            optimizer.backward_step(&loss);
        }

        fn assert_optimizer_checkpoint_error(error: anyhow::Error) {
            assert_eq!(
                error.downcast_ref::<TchOptimizerCheckpointError>(),
                Some(&TchOptimizerCheckpointError::StateSerializationUnavailable)
            );
        }

        #[test]
        fn optimizer_state_operations_fail_without_a_silent_fallback() {
            let (_, _, mut optimizer) = scalar_adam();
            let save_error = save_optimizer_state(&optimizer, Path::new("unused"))
                .expect_err("optimizer state save must be rejected");
            assert_optimizer_checkpoint_error(save_error);

            let load_error = load_optimizer_state(&mut optimizer, Path::new("unused"))
                .expect_err("optimizer state load must be rejected");
            assert_optimizer_checkpoint_error(load_error);
        }

        #[test]
        fn weights_only_checkpoint_is_not_exact_adam_split_resume() {
            let targets = [1.0_f32, -2.0, 3.0, -4.0];

            let (continuous_var_store, continuous_weight, mut continuous_optimizer) = scalar_adam();
            let mut continuous_prefix = 0.0;
            for (step, target) in targets.iter().copied().enumerate() {
                adam_step(&continuous_weight, &mut continuous_optimizer, target);
                if step == 1 {
                    continuous_prefix = continuous_weight.double_value(&[]);
                }
            }

            let (split_var_store, split_weight, mut split_optimizer) = scalar_adam();
            for target in targets.iter().copied().take(2) {
                adam_step(&split_weight, &mut split_optimizer, target);
            }
            assert!(
                (continuous_prefix - split_weight.double_value(&[])).abs() < 1e-7,
                "the continuous and split runs must agree before checkpointing"
            );

            let mut weights = Vec::new();
            split_var_store
                .save_to_stream(&mut weights)
                .expect("saving the weights-only checkpoint");

            let (mut resumed_var_store, resumed_weight, mut resumed_optimizer) = scalar_adam();
            resumed_var_store
                .load_from_stream(Cursor::new(weights))
                .expect("loading the weights-only checkpoint");
            for target in targets.iter().copied().skip(2) {
                adam_step(&resumed_weight, &mut resumed_optimizer, target);
            }

            let difference = (&continuous_weight - &resumed_weight)
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                difference > 1e-6,
                "resetting Adam state must diverge from continuous training; max difference={difference}"
            );

            // Keep this negative assertion until optimizer serialization is
            // implemented. A future bridge must change the capability contract
            // and this test together, then assert equality instead.
            let _ = continuous_var_store;
        }
    }
}
