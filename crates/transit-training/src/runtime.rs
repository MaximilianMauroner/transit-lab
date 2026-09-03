use crate::PretrainingConfig;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use transit_graph::GraphTensor;
use transit_model::{MaskSelection, ReferenceRelationalAutoencoder};

/// Runtime settings are part of the resolved experiment contract.  They are
/// deliberately independent from a particular tensor backend so the same
/// experiment can be inspected and benchmarked on a machine without
/// LibTorch installed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Cpu,
    Cuda { index: usize },
}

impl Default for DeviceKind {
    fn default() -> Self {
        Self::Cpu
    }
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda { index } => write!(formatter, "cuda:{index}"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DTypeKind {
    F32,
    F16,
    BF16,
}

impl Default for DTypeKind {
    fn default() -> Self {
        Self::F32
    }
}

impl std::fmt::Display for DTypeKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub device: DeviceKind,
    #[serde(default)]
    pub dtype: DTypeKind,
    #[serde(default = "default_intraop_threads")]
    pub intraop_threads: usize,
    #[serde(default = "default_interop_threads")]
    pub interop_threads: usize,
    #[serde(default = "default_rayon_threads")]
    pub rayon_threads: usize,
    #[serde(default = "default_concurrent_training_jobs")]
    pub concurrent_training_jobs: usize,
    #[serde(default = "default_gradient_accumulation")]
    pub gradient_accumulation: usize,
}

fn default_intraop_threads() -> usize {
    4
}

fn default_interop_threads() -> usize {
    1
}

fn default_rayon_threads() -> usize {
    4
}

fn default_concurrent_training_jobs() -> usize {
    1
}

fn default_gradient_accumulation() -> usize {
    1
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceKind::Cpu,
            dtype: DTypeKind::F32,
            intraop_threads: default_intraop_threads(),
            interop_threads: default_interop_threads(),
            rayon_threads: default_rayon_threads(),
            concurrent_training_jobs: default_concurrent_training_jobs(),
            gradient_accumulation: default_gradient_accumulation(),
        }
    }
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("intraop_threads", self.intraop_threads),
            ("interop_threads", self.interop_threads),
            ("rayon_threads", self.rayon_threads),
            ("concurrent_training_jobs", self.concurrent_training_jobs),
            ("gradient_accumulation", self.gradient_accumulation),
        ] {
            if value == 0 {
                bail!("runtime {name} must be positive");
            }
            if value > 100_000 {
                bail!("runtime {name} is unreasonably large");
            }
        }
        if !matches!(self.dtype, DTypeKind::F32) && matches!(self.device, DeviceKind::Cpu) {
            bail!("CPU runtime only supports f32; use f32 on this machine");
        }
        Ok(())
    }

    pub fn manifest(&self) -> serde_json::Value {
        serde_json::json!({
            "device": self.device.to_string(),
            "dtype": self.dtype.to_string(),
            "libtorchIntraopThreads": self.intraop_threads,
            "libtorchInteropThreads": self.interop_threads,
            "rayonThreads": self.rayon_threads,
            "concurrentTrainingJobs": self.concurrent_training_jobs,
            "gradientAccumulation": self.gradient_accumulation,
        })
    }
}

/// Best-effort resident set size for benchmark manifests.  Linux exposes this
/// without requiring a native allocator or a profiling dependency; other
/// platforms return `None` and keep the rest of the benchmark useful.
pub fn peak_resident_memory_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    let value = contents
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?;
    Some(value.saturating_mul(1024))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepBenchmarkReport {
    pub warmup_steps: usize,
    pub measured_steps: usize,
    pub median_milliseconds: f64,
    pub p95_milliseconds: f64,
    pub steps_per_second: f64,
    pub peak_resident_memory_bytes: Option<u64>,
    pub graph_counts: GraphBenchmarkCounts,
    pub runtime: RuntimeConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphBenchmarkCounts {
    pub stations: usize,
    pub lines: usize,
    pub patterns: usize,
    pub transit_edges: usize,
    pub transfer_edges: usize,
}

/// Benchmark the real reference training step, including mask generation and
/// graph encoding.  The warm-up is intentionally part of the API: first-use
/// allocation and CPU kernel setup should not distort the ETA shown by
/// Studio.
pub fn benchmark_reference_train_step(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    warmup_steps: usize,
    measured_steps: usize,
) -> Result<StepBenchmarkReport> {
    if measured_steps == 0 {
        bail!("measured_steps must be positive");
    }
    config.runtime.validate()?;
    graph.validate()?;
    let mut model = ReferenceRelationalAutoencoder::new(config.model.clone());
    for step in 0..warmup_steps {
        let mask =
            MaskSelection::sample(graph, &config.mask, config.seed.wrapping_add(step as u64));
        let _ = model.train_decoder_step(graph, &mask, config.learning_rate)?;
    }
    let mut durations = Vec::with_capacity(measured_steps);
    for step in 0..measured_steps {
        let mask = MaskSelection::sample(
            graph,
            &config.mask,
            config.seed.wrapping_add((warmup_steps + step) as u64),
        );
        let started = Instant::now();
        let _ = model.train_decoder_step(graph, &mask, config.learning_rate)?;
        durations.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    durations.sort_by(f64::total_cmp);
    let total_seconds = durations.iter().sum::<f64>() / 1_000.0;
    let percentile = |fraction: f64| {
        let index = ((durations.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(durations.len() - 1);
        durations[index]
    };
    Ok(StepBenchmarkReport {
        warmup_steps,
        measured_steps,
        median_milliseconds: percentile(0.50),
        p95_milliseconds: percentile(0.95),
        steps_per_second: if total_seconds > 0.0 {
            measured_steps as f64 / total_seconds
        } else {
            0.0
        },
        peak_resident_memory_bytes: peak_resident_memory_bytes(),
        graph_counts: GraphBenchmarkCounts {
            stations: graph.manifest.station_count,
            lines: graph.manifest.line_count,
            patterns: graph.manifest.pattern_count,
            transit_edges: graph.manifest.transit_edge_count,
            transfer_edges: graph.manifest.transfer_edge_count,
        },
        runtime: config.runtime.clone(),
    })
}
