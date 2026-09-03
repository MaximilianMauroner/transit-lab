use anyhow::{bail, Context, Result};
use chrono::{NaiveDate, SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand};
use gtfs_compile::{
    compile, load_snapshot, save_snapshot, CompileOptions, LineGroupingPolicy, ScopeDefinition,
};
use gtfs_ingest::{CalendarRecord, GtfsFeed, RouteRecord, StopRecord, StopTimeRecord, TripRecord};
use gtfs_source::{feed_by_id, fetch_feed, load_source_metadata, raw_feed_directory};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use transit_dataset::{
    create_dataset_manifest, load_dataset_collection, load_dataset_split, save_dataset_manifest,
    DatasetExample, DatasetPart, DatasetSplit,
};
use transit_domain::{hex_digest, parse_departure_time, sha256_bytes, ValidationReport};
use transit_graph::GraphTensor;
#[cfg(feature = "tch-backend")]
use transit_inference::{
    add_metric_percentiles, CriticalityPrediction, LinePrediction, PredictionFile,
};
use transit_inference::{
    load_predictions, predict_reference, rank_by_accessibility, save_predictions,
};
use transit_labels::{
    build_routing_baseline, generate_line_removal_labels, generate_line_removal_labels_resumable,
    generate_selected_line_removal_labels, load_jsonl, load_routing_baseline, sample_origins,
    save_jsonl, save_label_manifest_with_metadata, save_routing_baseline, spearman_rank,
    LabelGenerationConfig, OriginCandidate, LABEL_BATCH_SCHEMA_VERSION,
    LABEL_MANIFEST_SCHEMA_VERSION,
};
#[cfg(feature = "tch-backend")]
use transit_model::{denormalize_criticality_targets, LineEmbedding};
use transit_model::{
    MaskConfig, MaskSelection, ModelConfig, ReferenceLineRepresentationEncoder,
    ReferenceRelationalAutoencoder,
};
use transit_router::{Router, RouterConfig, ROUTER_ALGORITHM_VERSION};
use transit_search::{rank_similar_lines, SimilarityProfile};
use transit_training::{
    benchmark_reference_train_step, build_embedding_cache, list_training_checkpoints,
    load_checkpoint, load_config, load_embedding_cache, load_latest_training_checkpoint,
    load_training_checkpoint, max_wall_time, peak_resident_memory_bytes,
    run_reference_pretraining_with_policy, run_reference_pretraining_with_policy_options,
    save_checkpoint, save_embedding_cache, save_training_checkpoint,
    train_criticality_head_cached_multi_with_observer, train_criticality_head_with_observer,
    train_reference_autoencoder_with_observer, train_reference_multitask_with_observer,
    BestMetricState, CheckpointMetadata, CheckpointPolicy, CriticalityTrainingConfig, DTypeKind,
    DeviceKind, MultiTaskTrainingConfig, OptimizerState, PretrainingConfig, ReferenceCheckpoint,
    ReferenceTrainingOutcome, RngState, RuntimeConfig, SamplerState, ScalerState, SchedulerState,
    TrainingCheckpointV1, TrainingControl, TrainingCursor, TrainingObserver,
};

#[derive(Debug, Parser)]
#[command(name = "transit", version, about = "Rust GTFS graph-learning pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Fetch(FetchArgs),
    Validate(InputArgs),
    Compile(CompileArgs),
    Graph {
        #[command(subcommand)]
        command: GraphCommand,
    },
    Labels {
        #[command(subcommand)]
        command: LabelsCommand,
    },
    Train {
        #[command(subcommand)]
        command: TrainCommand,
    },
    Infer {
        #[command(subcommand)]
        command: InferCommand,
    },
    Verify {
        #[command(subcommand)]
        command: VerifyCommand,
    },
    SimilarLines(SimilarLinesArgs),
    Demo(DemoArgs),
    #[command(name = "build-dataset")]
    BuildDataset(BuildDatasetArgs),
    Evaluate(EvaluateArgs),
    EncodeDataset(EncodeDatasetArgs),
    TrainHeads(TrainHeadsArgs),
    FineTune(FineTuneArgs),
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
}

#[derive(Debug, Subcommand)]
enum GraphCommand {
    Build(GraphBuildArgs),
}

#[derive(Debug, Subcommand)]
enum LabelsCommand {
    LineRemoval(LabelsArgs),
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    Routing(BenchRoutingArgs),
    TrainStep(BenchTrainStepArgs),
    Threads(BenchThreadsArgs),
}

#[derive(Debug, Subcommand)]
enum InferCommand {
    Criticality(InferArgs),
}

#[derive(Debug, Args)]
struct FetchArgs {
    feed: String,
    #[arg(long)]
    url: Option<String>,
    #[arg(long, default_value = "data/raw")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct InputArgs {
    #[arg(long)]
    input: PathBuf,
}

#[derive(Debug, Args)]
struct CompileArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    service_date: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "unspecified feed scope")]
    scope: String,
    /// Optional YAML/JSON scope definition with a reproducible boundary and
    /// mode filter. The file is applied before compilation.
    #[arg(long)]
    scope_file: Option<PathBuf>,
    #[arg(long)]
    line_policy: Option<PathBuf>,
    #[arg(long, default_value = "unknown GTFS feed")]
    source_name: String,
}

#[derive(Debug, Args)]
struct GraphBuildArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct LabelsArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 256)]
    origins: usize,
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        "07:30".to_owned(),
        "08:30".to_owned(),
        "12:00".to_owned(),
        "16:30".to_owned(),
        "17:30".to_owned(),
        "22:00".to_owned()
    ])]
    departure_times: Vec<String>,
    #[arg(long, default_value_t = 4)]
    maximum_transfers: u8,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Reusable intact-network baseline. Defaults beside the label output.
    #[arg(long)]
    baseline: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct BenchRoutingArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, default_value_t = 32)]
    origins: usize,
    #[arg(long, value_delimiter = ',', default_values_t = vec!["07:30".to_owned(), "17:30".to_owned()])]
    departure_times: Vec<String>,
    #[arg(long, default_value_t = 4)]
    maximum_transfers: u8,
    #[arg(long, default_value_t = 4)]
    warmup_queries: usize,
    #[arg(long, default_value_t = 30)]
    measured_queries: usize,
    #[arg(long)]
    disabled_line: Option<u32>,
    #[arg(long, default_value_t = 4)]
    rayon_threads: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct BenchTrainStepArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 30)]
    measured_steps: usize,
    #[arg(long)]
    cpu_threads: Option<usize>,
    #[arg(long)]
    rayon_threads: Option<usize>,
    #[arg(long, default_value_t = 1)]
    gradient_accumulation: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct BenchThreadsArgs {
    #[arg(long)]
    snapshot: Option<PathBuf>,
    #[arg(long)]
    graph: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![2, 4, 6, 8])]
    threads: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    measured: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum TrainCommand {
    Pretrain(PretrainArgs),
    Criticality(CriticalityArgs),
    #[command(name = "multitask", alias = "representation")]
    MultiTask(MultiTaskArgs),
}

#[derive(Debug, Args)]
struct PretrainArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    seed: Option<u64>,
    #[command(flatten)]
    resumable: ResumableArgs,
}

#[derive(Debug, Args, Clone)]
struct ResumableArgs {
    /// Directory containing committed step-XXXXXXXXXXXX checkpoint folders.
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    /// A committed checkpoint directory or `latest`.
    #[arg(long)]
    resume: Option<String>,
    /// JSON control file written by the worker.
    #[arg(long)]
    control_file: Option<PathBuf>,
    #[arg(long)]
    checkpoint_every_steps: Option<usize>,
    #[arg(long)]
    checkpoint_every_seconds: Option<u64>,
    #[arg(long)]
    max_wall_time_seconds: Option<u64>,
    #[arg(long)]
    checkpoint_grace_seconds: Option<u64>,
    #[arg(long)]
    run_id: Option<String>,
    /// Training backend. `reference` is the dependency-free default;
    /// `libtorch` requires the CLI's optional `tch-backend` feature.
    #[arg(long)]
    backend: Option<String>,
    /// Treat the resume checkpoint as a source model for a new logical run.
    /// Ordinary resume remains strict about all experiment fingerprints.
    #[arg(long)]
    fork_from_checkpoint: bool,
    /// Runtime device. The dependency-free reference backend currently uses
    /// CPU; the value is still recorded so the same resolved contract can be
    /// handed to the optional LibTorch backend.
    #[arg(long)]
    device: Option<String>,
    /// Runtime floating-point type (`f32`, `f16`, or `bf16`).
    #[arg(long)]
    dtype: Option<String>,
    #[arg(long)]
    cpu_threads: Option<usize>,
    #[arg(long)]
    rayon_threads: Option<usize>,
    #[arg(long)]
    gradient_accumulation: Option<usize>,
}

#[derive(Debug, Args)]
struct CriticalityArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    labels: PathBuf,
    #[arg(long)]
    encoder: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Args)]
struct MultiTaskArgs {
    /// A versioned dataset directory. Only the selected split is loaded.
    #[arg(long)]
    dataset: Option<PathBuf>,
    /// Dataset partition to use for this training session.
    #[arg(long, default_value = "train")]
    split: String,
    /// Repeat once per compiled graph for low-level development runs. This
    /// bypasses dataset split validation and therefore requires the explicit
    /// --allow-unpartitioned-input acknowledgement.
    #[arg(long = "graph")]
    graphs: Vec<PathBuf>,
    /// Optional labels in the same order as --graph. Missing entries are
    /// treated as unsupervised snapshots.
    #[arg(long = "labels")]
    labels: Vec<PathBuf>,
    /// Explicitly acknowledge that repeated --graph inputs are not partitioned
    /// by a dataset manifest. Intended for local development only.
    #[arg(long)]
    allow_unpartitioned_input: bool,
    /// Explicitly acknowledge training on a validation or test partition.
    /// This is for diagnostics only; the worker always uses the train split.
    #[arg(long)]
    allow_nontrain_training_split: bool,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    seed: Option<u64>,
    #[command(flatten)]
    resumable: ResumableArgs,
}

#[derive(Debug, Args)]
struct InferArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "unknown-model")]
    model_id: String,
}

#[derive(Debug, Subcommand)]
enum VerifyCommand {
    TopLines(VerifyTopLinesArgs),
}

#[derive(Debug, Args)]
struct VerifyTopLinesArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    predictions: PathBuf,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, value_delimiter = ',', default_values_t = vec!["07:30".to_owned(), "08:30".to_owned()])]
    departure_times: Vec<String>,
}

#[derive(Debug, Args)]
struct SimilarLinesArgs {
    #[arg(long)]
    query_graph: PathBuf,
    #[arg(long)]
    query_line: String,
    #[arg(long)]
    candidate_graph: PathBuf,
    #[arg(long, default_value = "network-role")]
    profile: String,
    #[arg(long, help = "Override the role weight in a weighted profile")]
    role_weight: Option<f32>,
    #[arg(long)]
    service_weight: Option<f32>,
    #[arg(long)]
    geometry_weight: Option<f32>,
    #[arg(long)]
    resilience_weight: Option<f32>,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    encoder: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DemoArgs {
    #[arg(long, default_value = "data/demo")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct BuildDatasetArgs {
    /// Repeat once per snapshot graph. Labels are supplied in the same order.
    #[arg(long = "graph", required = true)]
    graphs: Vec<PathBuf>,
    #[arg(long = "labels")]
    labels: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "system-level")]
    split_strategy: String,
    #[arg(long = "validation-snapshot")]
    validation_snapshots: Vec<String>,
    #[arg(long = "test-snapshot")]
    test_snapshots: Vec<String>,
    #[arg(long = "validation-network")]
    validation_networks: Vec<String>,
    #[arg(long = "test-network")]
    test_networks: Vec<String>,
    /// Optional JSON split object supplied by the worker or an automation job.
    #[arg(long)]
    split_json: Option<String>,
}

#[derive(Debug, Args)]
struct EvaluateArgs {
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "test")]
    split: String,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 73)]
    seed: u64,
}

#[derive(Debug, Args)]
struct EncodeDatasetArgs {
    /// Repeat once per graph to encode into the immutable cache.
    #[arg(long = "graph", required = true)]
    graphs: Vec<PathBuf>,
    #[arg(long)]
    encoder: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct TrainHeadsArgs {
    #[arg(long = "graph", required = true)]
    graphs: Vec<PathBuf>,
    /// Optional labels in graph order. Missing entries are allowed only for
    /// graphs that are not used by this head experiment.
    #[arg(long = "labels")]
    labels: Vec<PathBuf>,
    #[arg(long)]
    encoder: PathBuf,
    #[arg(long)]
    embeddings: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Args)]
struct FineTuneArgs {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    steps: Option<usize>,
    #[command(flatten)]
    resumable: ResumableArgs,
}

#[derive(Debug)]
struct CliExit {
    code: i32,
    message: String,
}

impl std::fmt::Display for CliExit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CliExit {}

const EXIT_PAUSED: i32 = 75;
const EXIT_TIME_SLICED: i32 = 76;
const EXIT_CANCELLED: i32 = 77;

fn main() {
    let command = Cli::parse().command;
    let _ = emit_runtime_event("run.started", json!({}));
    let result = run(command);
    match result {
        Ok(()) => {
            let _ = emit_runtime_event("run.completed", json!({}));
        }
        Err(error) => {
            let exit = error.downcast_ref::<CliExit>();
            if exit.is_none() {
                let message = error.to_string();
                let _ = emit_runtime_event(
                    "run.failed",
                    json!({"code": "rust-command-failed", "message": message}),
                );
            }
            eprintln!("{error:#}");
            std::process::exit(exit.map(|value| value.code).unwrap_or(1));
        }
    }
}

fn run(command: Command) -> Result<()> {
    match command {
        Command::Fetch(args) => command_fetch(args),
        Command::Validate(args) => command_validate(args),
        Command::Compile(args) => command_compile(args),
        Command::Graph { command } => match command {
            GraphCommand::Build(args) => command_graph_build(args),
        },
        Command::Labels { command } => match command {
            LabelsCommand::LineRemoval(args) => command_labels(args),
        },
        Command::Train { command } => match command {
            TrainCommand::Pretrain(args) => command_pretrain(args),
            TrainCommand::Criticality(args) => command_criticality(args),
            TrainCommand::MultiTask(args) => command_multitask(args),
        },
        Command::Infer { command } => match command {
            InferCommand::Criticality(args) => command_infer(args),
        },
        Command::Verify { command } => match command {
            VerifyCommand::TopLines(args) => command_verify_top_lines(args),
        },
        Command::SimilarLines(args) => command_similar_lines(args),
        Command::Demo(args) => command_demo(args),
        Command::BuildDataset(args) => command_build_dataset(args),
        Command::Evaluate(args) => command_evaluate(args),
        Command::EncodeDataset(args) => command_encode_dataset(args),
        Command::TrainHeads(args) => command_train_heads(args),
        Command::FineTune(args) => command_fine_tune(args),
        Command::Bench { command } => match command {
            BenchCommand::Routing(args) => command_bench_routing(args),
            BenchCommand::TrainStep(args) => command_bench_train_step(args),
            BenchCommand::Threads(args) => command_bench_threads(args),
        },
    }
}

fn command_fetch(args: FetchArgs) -> Result<()> {
    let spec = feed_by_id(&args.feed).with_context(|| format!("unknown feed {}", args.feed))?;
    let url = args.url.or(spec.download_url.clone()).ok_or_else(|| {
        anyhow::anyhow!(
            "{} has no stable direct URL in the registry; pass --url with the current official ZIP URL",
            spec.id
        )
    })?;
    let staging = args.output.join(&spec.id).join("download");
    let metadata = fetch_feed(&spec, &url, &staging)?;
    let final_directory = raw_feed_directory(&args.output, &spec.id, &metadata.sha256);
    if final_directory.exists() {
        bail!(
            "raw feed snapshot already exists at {}; refusing to overwrite it",
            final_directory.display()
        );
    }
    std::fs::create_dir_all(&final_directory)?;
    std::fs::rename(staging.join("gtfs.zip"), final_directory.join("gtfs.zip"))
        .context("moving immutable GTFS ZIP")?;
    std::fs::rename(
        staging.join("source.json"),
        final_directory.join("source.json"),
    )
    .context("moving immutable source metadata")?;
    std::fs::remove_dir_all(args.output.join(&spec.id).join("download"))?;
    write_artifact_manifest(&final_directory, "raw-gtfs-feed")?;
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

fn command_validate(args: InputArgs) -> Result<()> {
    let feed = GtfsFeed::from_path(&args.input)?;
    println!("{}", serde_json::to_string_pretty(&feed.validation)?);
    if !feed.validation.is_valid() {
        bail!(
            "GTFS validation found {} errors",
            feed.validation.errors.len()
        );
    }
    Ok(())
}

fn command_compile(args: CompileArgs) -> Result<()> {
    let service_date = NaiveDate::parse_from_str(&args.service_date, "%Y-%m-%d")
        .with_context(|| format!("invalid service date {}", args.service_date))?;
    let raw_feed = GtfsFeed::from_path(&args.input)?;
    let scope = args
        .scope_file
        .as_deref()
        .map(ScopeDefinition::from_path)
        .transpose()?;
    let feed = if let Some(scope) = &scope {
        scope.apply(&raw_feed)?
    } else {
        raw_feed
    };
    if !feed.validation.is_valid() {
        bail!(
            "GTFS validation found {} errors; run transit validate for details",
            feed.validation.errors.len()
        );
    }
    let mut options = CompileOptions::for_date(service_date)
        .with_scope(args.scope)
        .with_source_name(args.source_name);
    if let Some(metadata_path) = source_metadata_path(&args.input) {
        let metadata = load_source_metadata(&metadata_path)?;
        if options.source_name == "unknown GTFS feed" {
            options.source_name = metadata.display_name;
        }
        if options.geographical_scope == "unspecified feed scope" {
            options = options.with_scope(metadata.geographical_scope);
        }
        options.licence = metadata.licence;
        options.downloaded_at = Some(metadata.downloaded_at);
    }
    if let Some(path) = args.line_policy {
        options.line_grouping_policy = LineGroupingPolicy::from_path(&path)?;
    }
    if let Some(scope) = &scope {
        options = options.with_scope_definition(scope);
    }
    let network = compile(&feed, &options)?;
    save_snapshot(&network, &args.output)?;
    write_artifact_manifest(&args.output, "compiled-snapshot")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "snapshot_id": network.snapshot_id,
            "stations": network.stations.len(),
            "lines": network.lines.len(),
            "patterns": network.patterns.len(),
            "transit_edges": network.transit_edges.len(),
            "transfers": network.transfers.len(),
        }))?
    );
    Ok(())
}

fn command_graph_build(args: GraphBuildArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let graph = GraphTensor::from_network(&network)?;
    graph.save(&args.output, &network)?;
    write_artifact_manifest(&args.output, "compiled-graph")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "snapshot_id": graph.manifest.snapshot_id,
            "stations": graph.manifest.station_count,
            "lines": graph.manifest.line_count,
            "transit_edges": graph.manifest.transit_edge_count,
        }))?
    );
    Ok(())
}

fn command_bench_routing(args: BenchRoutingArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let router = Router::from_network(
        &network,
        RouterConfig {
            maximum_transfers: args.maximum_transfers,
            ..RouterConfig::default()
        },
    )?;
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(&candidates, args.origins, &Default::default());
    let departures = args
        .departure_times
        .iter()
        .map(|value| parse_departure_time(value))
        .collect::<Result<Vec<_>>>()?;
    let report = router.benchmark_with_threads(
        &origins,
        &departures,
        args.disabled_line.map(transit_domain::LineIndex),
        args.warmup_queries,
        args.measured_queries,
        args.rayon_threads,
    )?;
    let result = json!({
        "schemaVersion": 1,
        "benchmark": "routing",
        "workload": "routing",
        "snapshotId": network.snapshot_id,
        "warmupUnits": report.warmup_queries,
        "measuredUnits": report.measured_queries,
        "estimatedWorkUnits": origins.len() * departures.len(),
        "medianMilliseconds": report.median_milliseconds,
        "p95Milliseconds": report.p95_milliseconds,
        "throughput": report.queries_per_second,
        "throughputUnit": "queries_per_second",
        "peakResidentMemoryBytes": peak_resident_memory_bytes(),
        "graphCounts": {
            "stations": report.station_count,
            "lines": report.line_count,
            "patterns": report.pattern_count
        },
        "runtime": {"device": "cpu", "dtype": "f32", "rayonThreads": args.rayon_threads},
        "threadConfiguration": {"rayonThreads": args.rayon_threads},
        "createdAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    });
    if let Some(output) = args.output.as_deref() {
        write_benchmark_result(output, &result)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "result": result,
            "output": args.output
        }))?
    );
    Ok(())
}

fn command_bench_train_step(args: BenchTrainStepArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let mut config = args
        .config
        .as_deref()
        .map(load_config::<PretrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    if let Some(threads) = args.cpu_threads {
        config.runtime.intraop_threads = threads;
    }
    if let Some(threads) = args.rayon_threads {
        config.runtime.rayon_threads = threads;
    }
    config.runtime.gradient_accumulation = args.gradient_accumulation;
    config.runtime.validate()?;
    let report =
        benchmark_reference_train_step(&graph, &config, args.warmup_steps, args.measured_steps)?;
    let result = json!({
        "schemaVersion": 1,
        "benchmark": "train-step",
        "workload": "train-step",
        "graphId": graph.manifest.snapshot_id,
        "warmupUnits": report.warmup_steps,
        "measuredUnits": report.measured_steps,
        "medianMilliseconds": report.median_milliseconds,
        "p95Milliseconds": report.p95_milliseconds,
        "throughput": report.steps_per_second,
        "throughputUnit": "steps_per_second",
        "peakResidentMemoryBytes": report.peak_resident_memory_bytes,
        "graphCounts": {
            "stations": report.graph_counts.stations,
            "lines": report.graph_counts.lines,
            "patterns": report.graph_counts.patterns,
            "transitEdges": report.graph_counts.transit_edges,
            "transferEdges": report.graph_counts.transfer_edges
        },
        "runtime": report.runtime,
        "threadConfiguration": {
            "intraopThreads": report.runtime.intraop_threads,
            "interopThreads": report.runtime.interop_threads,
            "rayonThreads": report.runtime.rayon_threads
        },
        "createdAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    });
    if let Some(output) = args.output.as_deref() {
        write_benchmark_result(output, &result)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "result": result,
            "output": args.output
        }))?
    );
    Ok(())
}

fn command_bench_threads(args: BenchThreadsArgs) -> Result<()> {
    if args.snapshot.is_none() && args.graph.is_none() {
        bail!("bench threads requires --snapshot, --graph, or both");
    }
    let mut reports = Vec::new();
    if let Some(snapshot_path) = args.snapshot {
        let network = load_snapshot(&snapshot_path)?;
        let router = Router::from_network(&network, RouterConfig::default())?;
        let candidates = network
            .stations
            .iter()
            .map(|station| OriginCandidate {
                index: station.index,
                latitude: station.latitude,
                longitude: station.longitude,
                transfer_degree: station.transfer_degree,
            })
            .collect::<Vec<_>>();
        let origins = sample_origins(&candidates, 32, &Default::default());
        let departures = vec![parse_departure_time("08:00")?];
        for thread_count in &args.threads {
            let report = router.benchmark_with_threads(
                &origins,
                &departures,
                None,
                args.warmup,
                args.measured,
                *thread_count,
            )?;
            reports.push(json!({
                "workload": "routing",
                "benchmark": "threads",
                "threadCount": thread_count,
                "snapshotId": network.snapshot_id,
                "warmupUnits": report.warmup_queries,
                "measuredUnits": report.measured_queries,
                "medianMilliseconds": report.median_milliseconds,
                "p95Milliseconds": report.p95_milliseconds,
                "throughput": report.queries_per_second,
                "throughputUnit": "queries_per_second",
                "graphCounts": {
                    "stations": report.station_count,
                    "lines": report.line_count,
                    "patterns": report.pattern_count
                },
                "threadConfiguration": {"rayonThreads": thread_count}
            }));
        }
    }
    if let Some(graph_path) = args.graph {
        let graph = GraphTensor::load(&graph_path)?;
        for thread_count in &args.threads {
            let mut config = PretrainingConfig::default();
            config.runtime.intraop_threads = *thread_count;
            config.runtime.rayon_threads = *thread_count;
            let report =
                benchmark_reference_train_step(&graph, &config, args.warmup, args.measured)?;
            reports.push(json!({
                "workload": "train-step",
                "benchmark": "threads",
                "threadCount": thread_count,
                "graphId": graph.manifest.snapshot_id,
                "warmupUnits": report.warmup_steps,
                "measuredUnits": report.measured_steps,
                "medianMilliseconds": report.median_milliseconds,
                "p95Milliseconds": report.p95_milliseconds,
                "throughput": report.steps_per_second,
                "throughputUnit": "steps_per_second",
                "peakResidentMemoryBytes": report.peak_resident_memory_bytes,
                "graphCounts": {
                    "stations": report.graph_counts.stations,
                    "lines": report.graph_counts.lines,
                    "patterns": report.graph_counts.patterns,
                    "transitEdges": report.graph_counts.transit_edges,
                    "transferEdges": report.graph_counts.transfer_edges
                },
                "runtime": report.runtime,
                "threadConfiguration": {
                    "intraopThreads": report.runtime.intraop_threads,
                    "interopThreads": report.runtime.interop_threads,
                    "rayonThreads": report.runtime.rayon_threads
                }
            }));
        }
    }
    let result = json!({
        "schemaVersion": 1,
        "benchmark": "threads",
        "workload": if reports.iter().any(|report| report["workload"] == "routing") &&
            reports.iter().any(|report| report["workload"] == "train-step") { "mixed" } else {
            reports.first().and_then(|report| report["workload"].as_str()).unwrap_or("mixed")
        },
        "reports": reports,
        "createdAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    });
    if let Some(output) = args.output.as_deref() {
        write_benchmark_result(output, &result)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"result": result, "output": args.output}))?
    );
    Ok(())
}

fn command_labels(args: LabelsArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let router = Router::from_network(
        &network,
        RouterConfig {
            maximum_transfers: args.maximum_transfers,
            ..RouterConfig::default()
        },
    )?;
    let departures = args
        .departure_times
        .iter()
        .map(|value| parse_departure_time(value))
        .collect::<Result<Vec<_>>>()?;
    let mut label_config = LabelGenerationConfig {
        maximum_origins: args.origins,
        ..LabelGenerationConfig::default()
    };
    label_config.origin_sampling.seed = args.seed;
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(&candidates, args.origins, &label_config.origin_sampling);
    let baseline_path = args.baseline.unwrap_or_else(|| {
        args.output
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("routing-baseline.json")
    });
    let baseline = if baseline_path.exists() {
        load_routing_baseline(&baseline_path, &router)
            .with_context(|| format!("loading routing baseline {}", baseline_path.display()))?
    } else {
        let baseline =
            build_routing_baseline(&router, network.snapshot_id.clone(), &origins, &departures);
        save_routing_baseline(&baseline_path, &baseline)?;
        baseline
    };
    let selected_lines = (0..network.lines.len())
        .map(|line| transit_domain::LineIndex(line as u32))
        .collect::<Vec<_>>();
    let labels = generate_line_removal_labels_resumable(
        &router,
        network.snapshot_id.clone(),
        &label_config,
        &selected_lines,
        &baseline,
        &args.output,
    )?;
    save_label_manifest_with_metadata(
        &args.output,
        &label_config,
        baseline.origins.len(),
        network.snapshot_id.clone(),
        &departures,
    )?;
    write_artifact_manifest(&baseline_path, "routing-baseline")?;
    write_artifact_manifest(&args.output, "criticality-labels")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"labels": labels.len(), "output": args.output}))?
    );
    Ok(())
}

fn copy_directory(source: &Path, target: &Path) -> Result<()> {
    if !source.is_dir() {
        bail!("graph input {} is not a directory", source.display());
    }
    fs::create_dir_all(target).with_context(|| format!("creating {}", target.display()))?;
    for entry in fs::read_dir(source).with_context(|| format!("reading {}", source.display()))? {
        let entry = entry?;
        let source_child = entry.path();
        let target_child = target.join(entry.file_name());
        if source_child.is_dir() {
            copy_directory(&source_child, &target_child)?;
        } else if source_child.is_file() {
            fs::copy(&source_child, &target_child).with_context(|| {
                format!(
                    "copying {} to {}",
                    source_child.display(),
                    target_child.display()
                )
            })?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct ExplicitSplit {
    train: Vec<String>,
    validation: Vec<String>,
    test: Vec<String>,
}

fn parse_explicit_split(raw: &str) -> Result<ExplicitSplit> {
    let value: Value = serde_json::from_str(raw).context("decoding --split-json")?;
    let object = value
        .as_object()
        .context("--split-json must contain an object")?;
    let parse_list = |name: &str| -> Result<Vec<String>> {
        let Some(value) = object.get(name) else {
            return Ok(Vec::new());
        };
        let values = value
            .as_array()
            .with_context(|| format!("--split-json {name} must be an array"))?;
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let value = value
                    .as_str()
                    .with_context(|| format!("--split-json {name}[{index}] must be a string"))?;
                if value.trim().is_empty() {
                    bail!("--split-json {name}[{index}] cannot be blank");
                }
                Ok(value.to_owned())
            })
            .collect()
    };
    let split = ExplicitSplit {
        train: parse_list("train")?,
        validation: parse_list("validation")?,
        test: parse_list("test")?,
    };
    let mut owners = BTreeMap::<String, &str>::new();
    for (name, values) in [
        ("train", &split.train),
        ("validation", &split.validation),
        ("test", &split.test),
    ] {
        for value in values {
            if let Some(previous) = owners.insert(value.clone(), name) {
                bail!("--split-json value {value:?} appears in both {previous} and {name}");
            }
        }
    }
    if split.train.is_empty() && split.validation.is_empty() && split.test.is_empty() {
        bail!("--split-json must contain at least one train, validation, or test value");
    }
    Ok(split)
}

fn split_contains(values: &[String], graph: &GraphTensor) -> bool {
    values.iter().any(|value| {
        value == &graph.manifest.snapshot_id || value == &graph.manifest.network_system_id
    })
}

fn dataset_split(
    args: &BuildDatasetArgs,
    graph: &GraphTensor,
    explicit: Option<&ExplicitSplit>,
) -> Result<String> {
    let snapshot = &graph.manifest.snapshot_id;
    let network = &graph.manifest.network_system_id;
    if let Some(explicit) = explicit {
        let matched = [
            ("test", split_contains(&explicit.test, graph)),
            ("validation", split_contains(&explicit.validation, graph)),
            ("train", split_contains(&explicit.train, graph)),
        ]
        .into_iter()
        .filter_map(|(split, matches)| matches.then_some(split))
        .collect::<Vec<_>>();
        if matched.len() > 1 {
            bail!(
                "graph {} matches multiple --split-json partitions",
                graph.manifest.snapshot_id
            );
        }
        if let Some(split) = matched.first() {
            return Ok((*split).into());
        }
        // A split object that omits `train` is commonly used to name only
        // held-out snapshots; those unmatched graphs remain training data.
        if explicit.train.is_empty() {
            return Ok("train".into());
        }
        bail!(
            "graph {} ({}) is absent from the explicit dataset split",
            snapshot,
            network
        );
    }
    let matches_test = args.test_snapshots.iter().any(|value| value == snapshot)
        || args.test_networks.iter().any(|value| value == network);
    let matches_validation = args
        .validation_snapshots
        .iter()
        .any(|value| value == snapshot)
        || args
            .validation_networks
            .iter()
            .any(|value| value == network);
    if matches_test && matches_validation {
        bail!(
            "graph {} matches both validation and test dataset partitions",
            graph.manifest.snapshot_id
        );
    }
    if matches_test {
        Ok("test".into())
    } else if matches_validation {
        Ok("validation".into())
    } else {
        Ok("train".into())
    }
}

fn line_target_map(label: &transit_labels::LineImpactLabel) -> BTreeMap<String, f32> {
    BTreeMap::from([
        (
            "accessibility_auc_loss".into(),
            label.accessibility_auc_loss,
        ),
        ("unreachable_share".into(), label.unreachable_share),
        (
            "mean_delay_reachable_seconds".into(),
            label.mean_delay_reachable_seconds,
        ),
        (
            "p95_delay_reachable_seconds".into(),
            label.p95_delay_reachable_seconds,
        ),
        ("mean_extra_transfers".into(), label.mean_extra_transfers),
        (
            "stations_losing_all_service_share".into(),
            label.stations_losing_all_service_share,
        ),
    ])
}

fn write_json_file(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value).context("encoding JSON artifact")?;
    if path.is_file() {
        let existing =
            fs::read(path).with_context(|| format!("reading existing {}", path.display()))?;
        if existing == bytes || existing == [bytes.as_slice(), b"\n"].concat() {
            return Ok(());
        }
        bail!(
            "refusing to overwrite immutable JSON artifact {}",
            path.display()
        );
    }
    let temporary = path.with_file_name(format!(
        ".{}-tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .with_context(|| format!("creating {}", temporary.display()))?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)
        .with_context(|| format!("committing JSON artifact {}", path.display()))?;
    Ok(())
}

fn write_benchmark_result(path: &Path, result: &Value) -> Result<()> {
    write_json_file(path, result)?;
    write_artifact_manifest(path, "benchmark-result")?;
    Ok(())
}

fn command_build_dataset(args: BuildDatasetArgs) -> Result<()> {
    if args.graphs.is_empty() {
        bail!("dataset build needs at least one graph");
    }
    if args.labels.len() > args.graphs.len() {
        bail!("provide at most one --labels file for each --graph");
    }
    if args.output.exists() {
        if args.output.join("dataset-manifest.json").is_file() {
            let existing = load_dataset_collection(&args.output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "already-committed",
                    "datasetId": existing.manifest.dataset_id,
                    "fingerprint": existing.manifest.fingerprint,
                    "output": args.output
                }))?
            );
            return Ok(());
        }
        if fs::read_dir(&args.output)?.next().is_some() {
            bail!(
                "refusing to overwrite non-empty dataset output {}",
                args.output.display()
            );
        }
    }
    let graphs = args
        .graphs
        .iter()
        .map(|path| {
            GraphTensor::load(path).with_context(|| format!("loading graph {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut labels = Vec::with_capacity(graphs.len());
    for (index, graph) in graphs.iter().enumerate() {
        let rows = args
            .labels
            .get(index)
            .map(|path| {
                load_jsonl(path).with_context(|| format!("loading labels {}", path.display()))
            })
            .transpose()?
            .unwrap_or_default();
        for label in &rows {
            if label.snapshot != graph.manifest.snapshot_id {
                bail!(
                    "label {} belongs to {}, expected {}",
                    label.line,
                    label.snapshot,
                    graph.manifest.snapshot_id
                );
            }
            if label.line.0 as usize >= graph.manifest.line_count {
                bail!(
                    "label line {} is outside graph {}",
                    label.line,
                    graph.manifest.snapshot_id
                );
            }
        }
        labels.push(rows);
    }
    let explicit_split = args
        .split_json
        .as_deref()
        .map(parse_explicit_split)
        .transpose()?;
    let splits = graphs
        .iter()
        .map(|graph| dataset_split(&args, graph, explicit_split.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    let train = graphs
        .iter()
        .zip(&splits)
        .filter(|(_, split)| split.as_str() == "train")
        .map(|(graph, _)| graph.manifest.snapshot_id.clone())
        .collect::<Vec<_>>();
    let validation = graphs
        .iter()
        .zip(&splits)
        .filter(|(_, split)| split.as_str() == "validation")
        .map(|(graph, _)| graph.manifest.snapshot_id.clone())
        .collect::<Vec<_>>();
    let test = graphs
        .iter()
        .zip(&splits)
        .filter(|(_, split)| split.as_str() == "test")
        .map(|(graph, _)| graph.manifest.snapshot_id.clone())
        .collect::<Vec<_>>();
    let split = if let Some(raw) = args.split_json.as_deref() {
        let value: Value = serde_json::from_str(raw).context("decoding --split-json")?;
        if !value.is_object() {
            bail!("--split-json must contain an object");
        }
        value
    } else {
        json!({
            "strategy": args.split_strategy,
            "train": train,
            "validation": validation,
            "test": test
        })
    };

    let mut examples = Vec::new();
    let mut parts = Vec::with_capacity(graphs.len());
    let mut input_artifacts = Vec::with_capacity(graphs.len());
    for (index, ((graph, rows), split_name)) in graphs.iter().zip(&labels).zip(&splits).enumerate()
    {
        let graph_relative = format!("graphs/{index:04}");
        let label_relative = format!("labels/{index:04}.jsonl");
        for label in rows {
            examples.push(DatasetExample {
                snapshot_id: graph.manifest.snapshot_id.clone(),
                line_index: label.line.0 as usize,
                split: split_name.clone(),
                line_identity: graph
                    .line_identities
                    .get(label.line.0 as usize)
                    .cloned()
                    .filter(|identity| !identity.trim().is_empty()),
                targets: line_target_map(label),
            });
        }
        input_artifacts.push(json!({
            "snapshotId": graph.manifest.snapshot_id,
            "graphSchema": graph.manifest.schema_version,
            "graphFingerprint": json_fingerprint(&graph.manifest)?
        }));
        parts.push(DatasetPart {
            graph,
            labels: rows,
            graph_directory: graph_relative,
            label_file: label_relative,
            split: split_name.clone(),
        });
    }
    let manifest = create_dataset_manifest(
        &parts,
        split,
        Some("examples.json".into()),
        examples.len(),
        input_artifacts,
        std::env::var("TRANSIT_RUN_ID").ok(),
    )?;

    fs::create_dir_all(&args.output)?;
    for (index, rows) in labels.iter().enumerate() {
        let graph_relative = format!("graphs/{index:04}");
        let label_relative = format!("labels/{index:04}.jsonl");
        copy_directory(&args.graphs[index], &args.output.join(&graph_relative))?;
        save_jsonl(&args.output.join(&label_relative), rows)?;
    }
    let examples_path = args.output.join("examples.json");
    write_json_file(&examples_path, &serde_json::to_value(&examples)?)?;
    save_dataset_manifest(&args.output.join("dataset-manifest.json"), &manifest)?;
    write_artifact_manifest(&args.output, "dataset-manifest")?;
    emit_runtime_event(
        "progress",
        json!({"step": "build-dataset", "completed": examples.len(), "total": examples.len(), "unit": "examples"}),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "completed",
            "datasetId": manifest.dataset_id,
            "fingerprint": manifest.fingerprint,
            "graphs": graphs.len(),
            "examples": examples.len(),
            "output": args.output
        }))?
    );
    Ok(())
}

#[derive(Clone, Debug)]
struct EvaluationRow {
    snapshot: String,
    line: usize,
    target: f32,
    features: Vec<f32>,
    frequency: f32,
    gnn: f32,
}

#[derive(Clone, Debug)]
struct LinearBaseline {
    mean: Vec<f32>,
    scale: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

impl LinearBaseline {
    fn score(&self, features: &[f32]) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .enumerate()
                .map(|(index, (weight, value))| {
                    weight * ((*value - self.mean[index]) / self.scale[index])
                })
                .sum::<f32>()
    }
}

fn fit_linear_baseline(rows: &[EvaluationRow], pairwise: bool) -> Option<LinearBaseline> {
    let width = rows.first()?.features.len();
    if width == 0 || rows.len() < 2 {
        return None;
    }
    let mut mean = vec![0.0; width];
    for row in rows {
        for (index, value) in row.features.iter().enumerate() {
            mean[index] += *value;
        }
    }
    for value in &mut mean {
        *value /= rows.len() as f32;
    }
    let mut scale = vec![0.0; width];
    for row in rows {
        for (index, value) in row.features.iter().enumerate() {
            let delta = *value - mean[index];
            scale[index] += delta * delta;
        }
    }
    for value in &mut scale {
        *value = (*value / rows.len() as f32).sqrt().max(1.0e-6);
    }
    let normalized = |features: &[f32]| {
        features
            .iter()
            .enumerate()
            .map(|(index, value)| (*value - mean[index]) / scale[index])
            .collect::<Vec<_>>()
    };
    let values = rows
        .iter()
        .map(|row| normalized(&row.features))
        .collect::<Vec<_>>();
    let mut weights = vec![0.0; width];
    let mut bias = rows.iter().map(|row| row.target).sum::<f32>() / rows.len() as f32;
    if pairwise {
        for _ in 0..250 {
            for left in 0..rows.len() {
                for right in (left + 1)..rows.len() {
                    let direction = (rows[left].target - rows[right].target).signum();
                    if direction == 0.0 {
                        continue;
                    }
                    let score_difference = (0..width)
                        .map(|index| (values[left][index] - values[right][index]) * weights[index])
                        .sum::<f32>();
                    if direction * score_difference < 0.25 {
                        for index in 0..width {
                            weights[index] +=
                                0.002 * direction * (values[left][index] - values[right][index]);
                        }
                    }
                }
            }
        }
    } else {
        for _ in 0..250 {
            let mut gradient = vec![0.0; width];
            let mut bias_gradient = 0.0;
            for (row, value) in rows.iter().zip(&values) {
                let prediction = bias + value.iter().zip(&weights).map(|(x, w)| x * w).sum::<f32>();
                let error = prediction - row.target;
                bias_gradient += error;
                for index in 0..width {
                    gradient[index] += error * value[index];
                }
            }
            let rate = 0.01 / rows.len() as f32;
            bias -= rate * bias_gradient;
            for index in 0..width {
                weights[index] -= rate * gradient[index];
            }
        }
    }
    Some(LinearBaseline {
        mean,
        scale,
        weights,
        bias,
    })
}

fn deterministic_score(seed: u64, snapshot: &str, line: usize) -> f32 {
    let mut value = seed ^ (line as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for byte in snapshot.as_bytes() {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(13);
    }
    value ^= value >> 29;
    (value as f64 / u64::MAX as f64) as f32
}

fn cosine_score(features: &[f32], centroid: &[f32]) -> f32 {
    let dot = features
        .iter()
        .zip(centroid)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left = features
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let right = centroid
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if left <= 1.0e-8 || right <= 1.0e-8 {
        0.0
    } else {
        dot / (left * right)
    }
}

fn ranking_metrics<F>(rows: &[EvaluationRow], top_k: usize, score: F) -> Value
where
    F: Fn(&EvaluationRow) -> f32,
{
    let mut by_snapshot = BTreeMap::<&str, Vec<&EvaluationRow>>::new();
    for row in rows {
        by_snapshot.entry(&row.snapshot).or_default().push(row);
    }
    let mut correlations = Vec::new();
    let mut pairwise_correct = 0usize;
    let mut pairwise_total = 0usize;
    let mut topk_overlap = Vec::new();
    for snapshot_rows in by_snapshot.values() {
        if snapshot_rows.len() < 2 {
            continue;
        }
        let targets = snapshot_rows
            .iter()
            .map(|row| row.target)
            .collect::<Vec<_>>();
        let predictions = snapshot_rows
            .iter()
            .map(|row| score(row))
            .collect::<Vec<_>>();
        if let Some(value) = spearman_rank(&predictions, &targets) {
            correlations.push(value);
        }
        for left in 0..snapshot_rows.len() {
            for right in (left + 1)..snapshot_rows.len() {
                let target_order = (targets[left] - targets[right]).signum();
                if target_order == 0.0 {
                    continue;
                }
                pairwise_total += 1;
                if target_order == (predictions[left] - predictions[right]).signum() {
                    pairwise_correct += 1;
                }
            }
        }
        let take = top_k.max(1).min(snapshot_rows.len());
        let mut target_order = (0..snapshot_rows.len()).collect::<Vec<_>>();
        target_order.sort_by(|left, right| targets[*right].total_cmp(&targets[*left]));
        let mut prediction_order = (0..snapshot_rows.len()).collect::<Vec<_>>();
        prediction_order.sort_by(|left, right| predictions[*right].total_cmp(&predictions[*left]));
        let target_set = target_order
            .into_iter()
            .take(take)
            .collect::<std::collections::BTreeSet<_>>();
        let overlap = prediction_order
            .into_iter()
            .take(take)
            .filter(|index| target_set.contains(index))
            .count();
        topk_overlap.push(overlap as f32 / take as f32);
    }
    let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len().max(1) as f32;
    json!({
        "examples": rows.len(),
        "snapshots": by_snapshot.len(),
        "spearman": if correlations.is_empty() { Value::Null } else { json!(mean(&correlations)) },
        "pairwiseAccuracy": if pairwise_total == 0 { Value::Null } else { json!(pairwise_correct as f32 / pairwise_total as f32) },
        "topKOverlap": if topk_overlap.is_empty() { Value::Null } else { json!(mean(&topk_overlap)) }
    })
}

fn unavailable_ranking_metrics(rows: &[EvaluationRow]) -> Value {
    let snapshots = rows
        .iter()
        .map(|row| row.snapshot.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    json!({
        "examples": rows.len(),
        "snapshots": snapshots.len(),
        "spearman": Value::Null,
        "pairwiseAccuracy": Value::Null,
        "topKOverlap": Value::Null,
        "status": "unavailable",
        "reason": "at least two labelled training examples are required to fit this baseline"
    })
}

fn command_evaluate(args: EvaluateArgs) -> Result<()> {
    let dataset = load_dataset_collection(&args.dataset)
        .with_context(|| format!("loading dataset {}", args.dataset.display()))?;
    #[cfg(feature = "tch-backend")]
    let native_model = is_native_libtorch_model(&args.model)?;
    #[cfg(not(feature = "tch-backend"))]
    let native_model = false;
    let checkpoint = if native_model {
        None
    } else {
        Some(
            load_checkpoint(&args.model)
                .with_context(|| format!("loading model {}", args.model.display()))?,
        )
    };
    let mut all_rows = Vec::new();
    let mut feature_width = None;
    for entry in &dataset.entries {
        let predictions = if native_model {
            #[cfg(feature = "tch-backend")]
            {
                native_prediction_file(
                    &entry.graph,
                    &transit_training::predict_tch_model(&args.model, &entry.graph)?,
                )?
            }
            #[cfg(not(feature = "tch-backend"))]
            {
                unreachable!("native LibTorch models require the tch-backend feature")
            }
        } else {
            predict_reference(
                checkpoint
                    .as_ref()
                    .expect("reference checkpoint is loaded when native model is false"),
                &entry.graph,
            )?
        };
        let mut predicted = BTreeMap::new();
        for prediction in predictions.predictions {
            predicted.insert(
                prediction.line as usize,
                prediction.metrics.first().copied().unwrap_or(0.0),
            );
        }
        let frequency_index = entry
            .graph
            .manifest
            .line_feature_names
            .iter()
            .position(|name| name == "daily_trip_count_log1p");
        if let Some(width) = feature_width {
            if width != entry.graph.line_features.cols {
                bail!("dataset line feature widths are inconsistent");
            }
        } else {
            feature_width = Some(entry.graph.line_features.cols);
        }
        for label in &entry.labels {
            let line = label.line.0 as usize;
            let Some(features) = entry.graph.line_features.values.get(
                line * entry.graph.line_features.cols..(line + 1) * entry.graph.line_features.cols,
            ) else {
                continue;
            };
            let frequency = frequency_index
                .map(|index| features[index])
                .unwrap_or_else(|| features.iter().sum());
            all_rows.push(EvaluationRow {
                snapshot: entry.graph.manifest.snapshot_id.clone(),
                line,
                target: label.accessibility_auc_loss,
                features: features.to_vec(),
                frequency,
                gnn: *predicted.get(&line).unwrap_or(&0.0),
            });
        }
    }
    let evaluation_rows = all_rows
        .iter()
        .filter(|row| {
            args.split == "all"
                || dataset
                    .entries
                    .iter()
                    .find(|entry| entry.graph.manifest.snapshot_id == row.snapshot)
                    .map(|entry| entry.split == args.split)
                    .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    if evaluation_rows.is_empty() {
        bail!("dataset has no labelled examples in split {}", args.split);
    }
    let training_rows = all_rows
        .iter()
        .filter(|row| {
            dataset
                .entries
                .iter()
                .find(|entry| entry.graph.manifest.snapshot_id == row.snapshot)
                .map(|entry| entry.split == "train")
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    // Never fit a baseline on evaluation or test rows. With too little
    // training data, report the learned baselines as unavailable rather than
    // leaking held-out labels into model selection.
    let fit_rows = (training_rows.len() >= 2).then_some(&training_rows);
    let linear = fit_rows.and_then(|rows| fit_linear_baseline(rows, false));
    let pairwise = fit_rows.and_then(|rows| fit_linear_baseline(rows, true));
    let centroid = fit_rows.map(|rows| {
        let mut centroid = vec![0.0; feature_width.unwrap_or(0)];
        for row in rows {
            for (index, value) in row.features.iter().enumerate() {
                centroid[index] += *value;
            }
        }
        for value in &mut centroid {
            *value /= rows.len() as f32;
        }
        centroid
    });
    let learned_or_unavailable = |model: Option<&LinearBaseline>| {
        model
            .map(|model| {
                ranking_metrics(&evaluation_rows, args.top_k, |row| {
                    model.score(&row.features)
                })
            })
            .unwrap_or_else(|| unavailable_ranking_metrics(&evaluation_rows))
    };
    let metrics = vec![
        (
            "gnn",
            ranking_metrics(&evaluation_rows, args.top_k, |row| row.gnn),
        ),
        ("engineered-linear", learned_or_unavailable(linear.as_ref())),
        ("pairwise-linear", learned_or_unavailable(pairwise.as_ref())),
        (
            "frequency",
            ranking_metrics(&evaluation_rows, args.top_k, |row| row.frequency),
        ),
        (
            "handcrafted-cosine",
            centroid
                .as_ref()
                .map(|centroid| {
                    ranking_metrics(&evaluation_rows, args.top_k, |row| {
                        cosine_score(&row.features, centroid)
                    })
                })
                .unwrap_or_else(|| unavailable_ranking_metrics(&evaluation_rows)),
        ),
        (
            "random",
            ranking_metrics(&evaluation_rows, args.top_k, |row| {
                deterministic_score(args.seed, &row.snapshot, row.line)
            }),
        ),
    ];
    let report = json!({
        "schemaVersion": 1,
        "datasetId": dataset.manifest.dataset_id,
        "datasetFingerprint": dataset.manifest.fingerprint,
        "modelId": args.model_id,
        "modelPath": args.model,
        "split": args.split,
        "topK": args.top_k,
        "trainingExamples": training_rows.len(),
        "fitExamples": fit_rows.map_or(0, Vec::len),
        "metrics": metrics.into_iter().map(|(baseline, values)| json!({"baseline": baseline, "values": values})).collect::<Vec<_>>(),
        "createdAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    });
    write_json_file(&args.output, &report)?;
    write_artifact_manifest(&args.output, "evaluation-result")?;
    emit_runtime_event(
        "metric",
        json!({"phase": "evaluation", "step": 1, "epoch": 1, "name": "evaluated_examples", "value": evaluation_rows.len()}),
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn checkpoint_root(output: &Path, requested: Option<&Path>) -> PathBuf {
    requested
        .map(Path::to_path_buf)
        .unwrap_or_else(|| output.with_extension("checkpoints"))
}

fn resolve_resume_checkpoint(root: &Path, requested: Option<&str>) -> Result<Option<PathBuf>> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    if requested.eq_ignore_ascii_case("latest") {
        let checkpoints = list_training_checkpoints(root)?;
        return checkpoints
            .last()
            .cloned()
            .map(Some)
            .with_context(|| format!("no committed checkpoints found under {}", root.display()));
    }
    let path = PathBuf::from(requested);
    if !path.join("manifest.json").is_file() {
        bail!(
            "resume checkpoint {} is not a committed checkpoint directory",
            path.display()
        );
    }
    let _ = load_latest_training_checkpoint(&path)?;
    Ok(Some(path))
}

#[cfg(feature = "tch-backend")]
fn resolve_tch_resume_checkpoint(root: &Path, requested: Option<&str>) -> Result<Option<PathBuf>> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    if requested.eq_ignore_ascii_case("latest") {
        return transit_training::list_tch_training_checkpoints(root)?
            .last()
            .cloned()
            .map(Some)
            .with_context(|| {
                format!(
                    "no committed LibTorch checkpoints found under {}",
                    root.display()
                )
            });
    }
    let path = PathBuf::from(requested);
    if !path.join("manifest.json").is_file() {
        bail!(
            "resume checkpoint {} is not a committed LibTorch checkpoint directory",
            path.display()
        );
    }
    let _ = transit_training::load_tch_checkpoint(&path)?;
    Ok(Some(path))
}

fn training_config_fingerprint(config: &PretrainingConfig) -> Result<String> {
    let encoded = serde_json::to_vec(config).context("encoding resolved training configuration")?;
    Ok(hex_digest(&sha256_bytes(&encoded)))
}

fn parse_device(value: &str) -> Result<DeviceKind> {
    let value = value.trim().to_ascii_lowercase();
    if value == "cpu" {
        return Ok(DeviceKind::Cpu);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        return Ok(DeviceKind::Cuda {
            index: index
                .parse()
                .with_context(|| format!("invalid CUDA device {value}"))?,
        });
    }
    bail!("device must be cpu or cuda:<index>");
}

fn parse_dtype(value: &str) -> Result<DTypeKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Ok(DTypeKind::F32),
        "f16" | "fp16" => Ok(DTypeKind::F16),
        "bf16" => Ok(DTypeKind::BF16),
        _ => bail!("dtype must be f32, f16, or bf16"),
    }
}

fn use_libtorch_backend(args: &ResumableArgs, runtime: &RuntimeConfig) -> Result<bool> {
    let explicit = args
        .backend
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    let requested = match explicit.as_deref() {
        None => false,
        Some("reference") | Some("reference-cpu") => false,
        Some("tch") | Some("libtorch") => true,
        Some(value) => bail!("backend must be reference or libtorch, got {value}"),
    };
    let cuda = matches!(runtime.device, DeviceKind::Cuda { .. });
    if cuda && !requested && args.backend.is_some() {
        bail!("CUDA cannot use the reference backend; select --backend libtorch");
    }
    Ok(requested || cuda)
}

fn apply_runtime_overrides(runtime: &mut RuntimeConfig, args: &ResumableArgs) -> Result<()> {
    if let Some(device) = args.device.as_deref() {
        runtime.device = parse_device(device)?;
    }
    if let Some(dtype) = args.dtype.as_deref() {
        runtime.dtype = parse_dtype(dtype)?;
    }
    if let Some(threads) = args.cpu_threads {
        runtime.intraop_threads = threads;
    }
    if let Some(threads) = args.rayon_threads {
        runtime.rayon_threads = threads;
    }
    if let Some(accumulation) = args.gradient_accumulation {
        runtime.gradient_accumulation = accumulation;
    }
    runtime.validate()
}

fn resolved_runtime(value: &Value, runtime: &mut RuntimeConfig) -> Result<()> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    if let Some(value) = object.get("device").and_then(Value::as_str) {
        runtime.device = parse_device(value)?;
    }
    if let Some(value) = object
        .get("dtype")
        .or_else(|| object.get("precision"))
        .and_then(Value::as_str)
    {
        runtime.dtype = parse_dtype(value)?;
    }
    let number = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| object.get(*name).and_then(Value::as_u64))
            .map(|value| value as usize)
    };
    if let Some(value) = number(&[
        "intraopThreads",
        "intraop_threads",
        "cpuThreads",
        "workerThreads",
    ]) {
        runtime.intraop_threads = value;
    }
    if let Some(value) = number(&["interopThreads", "interop_threads"]) {
        runtime.interop_threads = value;
    }
    if let Some(value) = number(&["rayonThreads", "rayon_threads"]) {
        runtime.rayon_threads = value;
    }
    if let Some(value) = number(&["gradientAccumulation", "gradient_accumulation"]) {
        runtime.gradient_accumulation = value;
    }
    runtime.validate()
}

fn apply_multitask_runtime(
    config: &mut MultiTaskTrainingConfig,
    args: &ResumableArgs,
) -> Result<()> {
    // `runtime` is the experiment-wide contract while the nested
    // pretraining runtime is retained for backwards-compatible YAML files.
    // Treat a non-default top-level value as authoritative and otherwise keep
    // explicitly configured pretraining settings; then mirror the resolved
    // value into both locations before handing it to a backend.
    let defaults = RuntimeConfig::default();
    let mut runtime = if config.runtime != defaults {
        config.runtime.clone()
    } else {
        config.pretraining.runtime.clone()
    };
    apply_runtime_overrides(&mut runtime, args)?;
    config.runtime = runtime.clone();
    config.pretraining.runtime = runtime;
    Ok(())
}

fn training_metadata(
    graph: &GraphTensor,
    config: &PretrainingConfig,
    resumable: &ResumableArgs,
) -> Result<CheckpointMetadata> {
    Ok(CheckpointMetadata {
        run_id: resumable
            .run_id
            .clone()
            .or_else(|| std::env::var("TRANSIT_RUN_ID").ok())
            .unwrap_or_else(|| format!("local-{}", graph.manifest.snapshot_id)),
        attempt_id: std::env::var("TRANSIT_ATTEMPT_ID").ok(),
        dataset_fingerprint: std::env::var("TRANSIT_DATASET_FINGERPRINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| format!("graph:{}", graph.manifest.snapshot_id)),
        config_fingerprint: std::env::var("TRANSIT_CONFIG_FINGERPRINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(training_config_fingerprint(config)?),
        code_commit: std::env::var("TRANSIT_LAB_GIT_COMMIT")
            .unwrap_or_else(|_| "working-tree".into()),
        backend_version: env!("CARGO_PKG_VERSION").into(),
        device_type: config.runtime.device.to_string(),
    })
}

fn finish_resumable_outcome(
    output: &Path,
    checkpoint_root: &Path,
    config: &PretrainingConfig,
    session: transit_training::ReferenceTrainingSession,
    outcome: ReferenceTrainingOutcome,
    metadata: &CheckpointMetadata,
) -> Result<()> {
    match outcome {
        ReferenceTrainingOutcome::Completed { checkpoint_path } => {
            save_checkpoint(
                output,
                &ReferenceCheckpoint {
                    encoder: session.model,
                    head: None,
                    report: Some(session.report.clone()),
                    representation: None,
                    config_fingerprint: Some(metadata.config_fingerprint.clone()),
                    seed: Some(config.seed),
                    training_run_id: Some(metadata.run_id.clone()),
                    dataset_fingerprint: Some(metadata.dataset_fingerprint.clone()),
                    model_id: None,
                },
            )?;
            write_artifact_manifest(output, "model-checkpoint")?;
            emit_checkpoint_created(&checkpoint_path, "pretraining", config.steps, config.steps)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "completed",
                    "output": output,
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "report": session.report
                }))?
            );
            Ok(())
        }
        ReferenceTrainingOutcome::Paused { checkpoint_path } => {
            emit_runtime_event(
                "run.paused",
                json!({"path": checkpoint_path, "reason": "cooperative-pause"}),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "paused",
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "globalStep": session.cursor.global_step
                }))?
            );
            Err(anyhow::Error::new(CliExit {
                code: EXIT_PAUSED,
                message: "training paused after committing a checkpoint".into(),
            }))
        }
        ReferenceTrainingOutcome::TimeSliceExpired { checkpoint_path } => {
            emit_runtime_event(
                "run.time-sliced",
                json!({"path": checkpoint_path, "reason": "attempt-deadline"}),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "time-sliced",
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "globalStep": session.cursor.global_step
                }))?
            );
            Err(anyhow::Error::new(CliExit {
                code: EXIT_TIME_SLICED,
                message: "training attempt ended after committing a deadline checkpoint".into(),
            }))
        }
        ReferenceTrainingOutcome::Cancelled => {
            emit_runtime_event("run.cancelled", json!({"reason": "cooperative-cancel"}))?;
            Err(anyhow::Error::new(CliExit {
                code: EXIT_CANCELLED,
                message: "training cancelled".into(),
            }))
        }
    }
}

fn finish_multitask_resumable_outcome(
    output: &Path,
    checkpoint_root: &Path,
    result: transit_training::ResumableMultiTaskResult,
    outcome: ReferenceTrainingOutcome,
) -> Result<()> {
    match outcome {
        ReferenceTrainingOutcome::Completed { checkpoint_path } => {
            let (_, manifest) = load_training_checkpoint(&checkpoint_path).with_context(|| {
                format!("loading committed checkpoint {}", checkpoint_path.display())
            })?;
            save_checkpoint(output, &result.checkpoint)?;
            write_artifact_manifest(output, "model-checkpoint")?;
            emit_checkpoint_created(
                &checkpoint_path,
                &manifest.phase,
                manifest.global_step as usize,
                manifest.global_step as usize,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "completed",
                    "output": output,
                    "checkpointRoot": checkpoint_root,
                    "checkpoint": checkpoint_path,
                    "report": result.report
                }))?
            );
            Ok(())
        }
        ReferenceTrainingOutcome::Paused { checkpoint_path } => {
            emit_runtime_event(
                "run.paused",
                json!({"path": checkpoint_path, "reason": "cooperative-pause"}),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "paused",
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "report": result.report
                }))?
            );
            Err(anyhow::Error::new(CliExit {
                code: EXIT_PAUSED,
                message: "training paused after committing a checkpoint".into(),
            }))
        }
        ReferenceTrainingOutcome::TimeSliceExpired { checkpoint_path } => {
            emit_runtime_event(
                "run.time-sliced",
                json!({"path": checkpoint_path, "reason": "attempt-deadline"}),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "time-sliced",
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "report": result.report
                }))?
            );
            Err(anyhow::Error::new(CliExit {
                code: EXIT_TIME_SLICED,
                message: "training attempt ended after committing a deadline checkpoint".into(),
            }))
        }
        ReferenceTrainingOutcome::Cancelled => {
            emit_runtime_event("run.cancelled", json!({"reason": "cooperative-cancel"}))?;
            Err(anyhow::Error::new(CliExit {
                code: EXIT_CANCELLED,
                message: "training cancelled".into(),
            }))
        }
    }
}

#[cfg(feature = "tch-backend")]
fn command_pretrain_libtorch(args: PretrainArgs, config: PretrainingConfig) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)
        .with_context(|| format!("loading graph {}", args.graph.display()))?;
    let labels = Vec::new();
    let datasets = vec![(&graph, labels.as_slice())];
    let multitask_config = MultiTaskTrainingConfig {
        pretraining: config.clone(),
        representation: transit_model::RepresentationConfig::default(),
        metric_epochs: 0,
        metric_learning_rate: 0.0,
        metric_margin: 0.25,
        metric_weight_decay: 0.0,
        max_triplets: 0,
        criticality: CriticalityTrainingConfig {
            epochs: 0,
            ..CriticalityTrainingConfig::default()
        },
        runtime: config.runtime.clone(),
    };
    command_libtorch_multitask(
        MultiTaskArgs {
            dataset: None,
            split: "train".into(),
            graphs: vec![args.graph],
            labels: Vec::new(),
            allow_unpartitioned_input: true,
            allow_nontrain_training_split: false,
            config: args.config,
            output: args.output,
            seed: args.seed,
            resumable: args.resumable,
        },
        multitask_config,
        None,
        datasets,
    )
}

#[cfg(not(feature = "tch-backend"))]
fn command_pretrain_libtorch(_args: PretrainArgs, _config: PretrainingConfig) -> Result<()> {
    bail!("the LibTorch backend was requested; rebuild transit-cli with --features tch-backend")
}

fn command_pretrain(args: PretrainArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let mut config = args
        .config
        .as_deref()
        .map(load_config::<PretrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    if let Some(seed) = args.seed {
        config.seed = seed;
    }
    apply_runtime_overrides(&mut config.runtime, &args.resumable)?;
    config.runtime.validate()?;
    if use_libtorch_backend(&args.resumable, &config.runtime)? {
        return command_pretrain_libtorch(args, config);
    }
    let root = checkpoint_root(&args.output, args.resumable.checkpoint_dir.as_deref());
    let resume = resolve_resume_checkpoint(&root, args.resumable.resume.as_deref())?;
    let control = TrainingControl::with_policy(
        args.resumable.control_file.clone(),
        args.resumable
            .max_wall_time_seconds
            .map(std::time::Duration::from_secs),
        args.resumable
            .checkpoint_grace_seconds
            .map(std::time::Duration::from_secs),
    );
    let metadata = training_metadata(&graph, &config, &args.resumable)?;
    let mut observer = JsonlTrainingObserver;
    let (session, outcome) = run_reference_pretraining_with_policy_options(
        &graph,
        &config,
        &root,
        resume.as_deref(),
        &control,
        CheckpointPolicy {
            every_steps: args.resumable.checkpoint_every_steps,
            every_seconds: args.resumable.checkpoint_every_seconds,
        },
        &metadata,
        args.resumable.fork_from_checkpoint,
        &mut observer,
    )?;
    finish_resumable_outcome(&args.output, &root, &config, session, outcome, &metadata)
}

fn command_criticality(args: CriticalityArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let labels = transit_labels::load_jsonl(&args.labels)?;
    let checkpoint = load_checkpoint(&args.encoder)?;
    let mut config = args
        .config
        .as_deref()
        .map(load_config::<CriticalityTrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    if let Some(seed) = args.seed {
        config.seed = seed;
    }
    let mut observer = JsonlTrainingObserver;
    let (head, report) = train_criticality_head_with_observer(
        &checkpoint.encoder,
        &graph,
        &labels,
        &config,
        &mut observer,
    )?;
    let training_run_id = checkpoint.training_run_id.clone();
    let dataset_fingerprint = checkpoint.dataset_fingerprint.clone();
    let model_id = checkpoint.model_id.clone();
    save_checkpoint(
        &args.output,
        &ReferenceCheckpoint {
            encoder: checkpoint.encoder,
            head: Some(head),
            report: Some(report.clone()),
            representation: checkpoint.representation,
            config_fingerprint: None,
            seed: Some(config.seed),
            training_run_id,
            dataset_fingerprint,
            model_id,
        },
    )?;
    write_artifact_manifest(&args.output, "model-checkpoint")?;
    emit_checkpoint_created(&args.output, "criticality", config.epochs, config.epochs)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn load_multitask_config(
    path: Option<&Path>,
) -> Result<(MultiTaskTrainingConfig, Option<String>, Option<u64>)> {
    let Some(path) = path else {
        return Ok((MultiTaskTrainingConfig::default(), None, None));
    };
    if matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("yaml") | Some("yml")
    ) {
        let config: MultiTaskTrainingConfig = load_config(path)?;
        config.runtime.validate()?;
        config.pretraining.runtime.validate()?;
        return Ok((config, None, None));
    }
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding JSON config {}", path.display()))?;
    let Some(model_config) = value.get("modelConfig") else {
        let mut config: MultiTaskTrainingConfig = load_config(path)?;
        if let Some(runtime) = value.get("runtime") {
            resolved_runtime(runtime, &mut config.runtime)?;
            config.pretraining.runtime = config.runtime.clone();
        }
        return Ok((config, None, None));
    };
    let fingerprint = value
        .get("configFingerprint")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let seed = value.get("seed").and_then(Value::as_u64);
    let mut config: MultiTaskTrainingConfig = serde_json::from_value(model_config.clone())
        .context("decoding resolved multitask model config")?;
    if let Some(runtime) = value.get("runtime") {
        resolved_runtime(runtime, &mut config.runtime)?;
        config.pretraining.runtime = config.runtime.clone();
    }
    Ok((config, fingerprint, seed))
}

fn set_multitask_seed(config: &mut MultiTaskTrainingConfig, seed: u64) {
    config.pretraining.seed = seed;
    config.representation.seed = seed;
    config.criticality.seed = seed;
}

fn load_multitask_datasets(
    args: &MultiTaskArgs,
) -> Result<(Vec<GraphTensor>, Vec<Vec<transit_labels::LineImpactLabel>>)> {
    if let Some(dataset) = args.dataset.as_deref() {
        if !args.graphs.is_empty() {
            bail!("--dataset cannot be combined with --graph");
        }
        if !args.labels.is_empty() {
            bail!("--dataset cannot be combined with --labels");
        }
        if args.allow_unpartitioned_input {
            bail!("--allow-unpartitioned-input is only valid with --graph");
        }
        let split = DatasetSplit::parse(&args.split)?;
        if split != DatasetSplit::Train && !args.allow_nontrain_training_split {
            bail!(
                "training on the {} split requires --allow-nontrain-training-split",
                split.as_str()
            );
        }
        let collection = load_dataset_split(dataset, split).with_context(|| {
            format!(
                "loading {} split from dataset {}",
                split.as_str(),
                dataset.display()
            )
        })?;
        let mut graphs = Vec::with_capacity(collection.entries.len());
        let mut labels = Vec::with_capacity(collection.entries.len());
        for entry in collection.entries {
            graphs.push(entry.graph);
            labels.push(entry.labels);
        }
        return Ok((graphs, labels));
    }

    if args.graphs.is_empty() {
        bail!("provide --dataset or at least one --graph");
    }
    if !args.allow_unpartitioned_input {
        bail!(
            "repeated --graph training inputs are unpartitioned; provide --dataset or pass --allow-unpartitioned-input for local development"
        );
    }
    if args.allow_nontrain_training_split {
        bail!("--allow-nontrain-training-split requires --dataset");
    }
    if DatasetSplit::parse(&args.split)? != DatasetSplit::Train {
        bail!("--split is only available with --dataset");
    }
    if args.labels.len() > args.graphs.len() {
        bail!("provide at most one --labels file for each --graph");
    }
    let graphs = args
        .graphs
        .iter()
        .map(|path| {
            GraphTensor::load(path).with_context(|| format!("loading graph {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut labels = Vec::with_capacity(graphs.len());
    for index in 0..graphs.len() {
        labels.push(if let Some(path) = args.labels.get(index) {
            load_jsonl(path).with_context(|| format!("loading labels {}", path.display()))?
        } else {
            Vec::new()
        });
    }
    Ok((graphs, labels))
}

fn command_multitask(args: MultiTaskArgs) -> Result<()> {
    let (mut config, config_fingerprint, declared_seed) =
        load_multitask_config(args.config.as_deref())?;
    if let (Some(declared), Some(requested)) = (declared_seed, args.seed) {
        if declared != requested {
            bail!("--seed {requested} does not match the resolved config seed {declared}");
        }
    }
    if let Some(seed) = args.seed {
        set_multitask_seed(&mut config, seed);
    }
    apply_multitask_runtime(&mut config, &args.resumable)?;
    let (graphs, labels) = load_multitask_datasets(&args)?;
    let datasets: Vec<(&GraphTensor, &[transit_labels::LineImpactLabel])> = graphs
        .iter()
        .zip(&labels)
        .map(|(graph, labels)| (graph, labels.as_slice()))
        .collect();
    if use_libtorch_backend(&args.resumable, &config.runtime)? {
        return command_libtorch_multitask(args, config, config_fingerprint, datasets);
    }
    if resumable_requested(&args.resumable) {
        return command_multitask_resumable(args, config, config_fingerprint, datasets);
    }
    let mut observer = JsonlTrainingObserver;
    let (checkpoint, report) =
        train_reference_multitask_with_observer(&datasets, &config, &mut observer)?;
    let checkpoint = ReferenceCheckpoint {
        config_fingerprint,
        seed: Some(args.seed.unwrap_or(config.pretraining.seed)),
        training_run_id: std::env::var("TRANSIT_RUN_ID").ok(),
        dataset_fingerprint: std::env::var("TRANSIT_DATASET_FINGERPRINT").ok(),
        model_id: None,
        ..checkpoint
    };
    save_checkpoint(&args.output, &checkpoint)?;
    write_artifact_manifest(&args.output, "model-checkpoint")?;
    emit_checkpoint_created(
        &args.output,
        "criticality",
        config.criticality.epochs,
        config.criticality.epochs,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "output": args.output,
            "report": report,
        }))?
    );
    Ok(())
}

#[cfg(feature = "tch-backend")]
fn command_libtorch_multitask(
    args: MultiTaskArgs,
    config: MultiTaskTrainingConfig,
    config_fingerprint: Option<String>,
    datasets: Vec<(&GraphTensor, &[transit_labels::LineImpactLabel])>,
) -> Result<()> {
    if args.resumable.fork_from_checkpoint && args.resumable.resume.is_none() {
        bail!("--fork-from-checkpoint requires --resume");
    }
    let Some((first_graph, _)) = datasets.first() else {
        bail!("no graph datasets were provided");
    };
    let checkpoint_root = checkpoint_root(&args.output, args.resumable.checkpoint_dir.as_deref());
    let metadata = CheckpointMetadata {
        run_id: args
            .resumable
            .run_id
            .clone()
            .or_else(|| std::env::var("TRANSIT_RUN_ID").ok())
            .unwrap_or_else(|| format!("local-{}", first_graph.manifest.snapshot_id)),
        attempt_id: std::env::var("TRANSIT_ATTEMPT_ID").ok(),
        dataset_fingerprint: std::env::var("TRANSIT_DATASET_FINGERPRINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                datasets
                    .iter()
                    .map(|(graph, _)| graph.manifest.snapshot_id.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            }),
        config_fingerprint: std::env::var("TRANSIT_CONFIG_FINGERPRINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or(config_fingerprint)
            .unwrap_or(json_fingerprint(&config)?),
        code_commit: std::env::var("TRANSIT_LAB_GIT_COMMIT")
            .unwrap_or_else(|_| "working-tree".into()),
        backend_version: env!("CARGO_PKG_VERSION").into(),
        device_type: config.runtime.device.to_string(),
    };
    let resume = resolve_tch_resume_checkpoint(&checkpoint_root, args.resumable.resume.as_deref())?;
    let control = TrainingControl::with_policy(
        args.resumable.control_file.clone(),
        args.resumable
            .max_wall_time_seconds
            .map(std::time::Duration::from_secs),
        args.resumable
            .checkpoint_grace_seconds
            .map(std::time::Duration::from_secs),
    );
    let mut observer = JsonlTrainingObserver;
    let (session, outcome, report) = transit_training::run_tch_multitask_with_policy_options(
        &datasets,
        &config,
        &checkpoint_root,
        resume.as_deref(),
        &control,
        CheckpointPolicy {
            every_steps: args.resumable.checkpoint_every_steps,
            every_seconds: args.resumable.checkpoint_every_seconds,
        },
        &metadata,
        args.resumable.fork_from_checkpoint,
        &mut observer,
    )?;
    match outcome {
        transit_training::TchTrainingOutcome::Completed { checkpoint_path } => {
            let weights_path = native_model_weights_path(&args.output)?;
            let graph_refs = datasets.iter().map(|(graph, _)| *graph).collect::<Vec<_>>();
            transit_training::save_tch_model_artifact(
                &args.output,
                &weights_path,
                &session,
                &graph_refs,
                &config,
                &metadata,
            )?;
            write_artifact_manifest_for_paths(
                &args.output,
                "model-checkpoint",
                &[&args.output, &weights_path],
            )?;
            emit_checkpoint_created(
                &checkpoint_path,
                &session.cursor.phase,
                session.cursor.global_step as usize,
                session.cursor.global_step as usize,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "completed",
                    "backend": "libtorch",
                    "output": args.output,
                    "weights": weights_path,
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "report": report
                }))?
            );
            Ok(())
        }
        transit_training::TchTrainingOutcome::Paused { checkpoint_path } => {
            emit_runtime_event(
                "run.paused",
                json!({"path": checkpoint_path, "reason": "cooperative-pause"}),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "paused",
                    "backend": "libtorch",
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "globalStep": session.cursor.global_step,
                    "report": report
                }))?
            );
            Err(anyhow::Error::new(CliExit {
                code: EXIT_PAUSED,
                message: "LibTorch multi-task training paused after committing a checkpoint".into(),
            }))
        }
        transit_training::TchTrainingOutcome::TimeSliceExpired { checkpoint_path } => {
            emit_runtime_event(
                "run.time-sliced",
                json!({"path": checkpoint_path, "reason": "attempt-deadline"}),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "time-sliced",
                    "backend": "libtorch",
                    "checkpoint": checkpoint_path,
                    "checkpointRoot": checkpoint_root,
                    "globalStep": session.cursor.global_step,
                    "report": report
                }))?
            );
            Err(anyhow::Error::new(CliExit {
                code: EXIT_TIME_SLICED,
                message: "LibTorch multi-task training attempt ended after committing a deadline checkpoint".into(),
            }))
        }
        transit_training::TchTrainingOutcome::Cancelled => {
            emit_runtime_event("run.cancelled", json!({"reason": "cooperative-cancel"}))?;
            Err(anyhow::Error::new(CliExit {
                code: EXIT_CANCELLED,
                message: "LibTorch multi-task training cancelled".into(),
            }))
        }
    }
}

#[cfg(not(feature = "tch-backend"))]
fn command_libtorch_multitask(
    _args: MultiTaskArgs,
    _config: MultiTaskTrainingConfig,
    _config_fingerprint: Option<String>,
    _datasets: Vec<(&GraphTensor, &[transit_labels::LineImpactLabel])>,
) -> Result<()> {
    bail!("the LibTorch backend was requested; rebuild transit-cli with --features tch-backend")
}

#[cfg(feature = "tch-backend")]
fn native_model_weights_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .context("native LibTorch model output has a non-UTF-8 filename")?;
    Ok(output.with_file_name(format!("{name}.weights.ot")))
}

fn resumable_requested(args: &ResumableArgs) -> bool {
    args.checkpoint_dir.is_some()
        || args.resume.is_some()
        || args.control_file.is_some()
        || args.checkpoint_every_steps.is_some()
        || args.checkpoint_every_seconds.is_some()
        || args.max_wall_time_seconds.is_some()
        || args.checkpoint_grace_seconds.is_some()
        || args.run_id.is_some()
        || args.backend.is_some()
        || args.fork_from_checkpoint
        || args.device.is_some()
        || args.dtype.is_some()
        || args.cpu_threads.is_some()
        || args.rayon_threads.is_some()
        || args.gradient_accumulation.is_some()
}

fn command_multitask_resumable(
    args: MultiTaskArgs,
    config: MultiTaskTrainingConfig,
    config_fingerprint: Option<String>,
    datasets: Vec<(&GraphTensor, &[transit_labels::LineImpactLabel])>,
) -> Result<()> {
    let Some((first_graph, _)) = datasets.first() else {
        bail!("no graph datasets were provided");
    };
    let pretraining = &config.pretraining;
    pretraining.runtime.validate()?;
    let checkpoint_root = checkpoint_root(&args.output, args.resumable.checkpoint_dir.as_deref());
    let metadata = CheckpointMetadata {
        run_id: args
            .resumable
            .run_id
            .clone()
            .or_else(|| std::env::var("TRANSIT_RUN_ID").ok())
            .unwrap_or_else(|| format!("local-{}", first_graph.manifest.snapshot_id)),
        attempt_id: std::env::var("TRANSIT_ATTEMPT_ID").ok(),
        dataset_fingerprint: std::env::var("TRANSIT_DATASET_FINGERPRINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                datasets
                    .iter()
                    .map(|(graph, _)| graph.manifest.snapshot_id.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            }),
        config_fingerprint: std::env::var("TRANSIT_CONFIG_FINGERPRINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or(config_fingerprint)
            .unwrap_or(json_fingerprint(&config)?),
        code_commit: std::env::var("TRANSIT_LAB_GIT_COMMIT")
            .unwrap_or_else(|_| "working-tree".into()),
        backend_version: env!("CARGO_PKG_VERSION").into(),
        device_type: pretraining.runtime.device.to_string(),
    };
    let resume = resolve_resume_checkpoint(&checkpoint_root, args.resumable.resume.as_deref())?;
    let control = TrainingControl::with_policy(
        args.resumable.control_file.clone(),
        args.resumable
            .max_wall_time_seconds
            .map(std::time::Duration::from_secs),
        args.resumable
            .checkpoint_grace_seconds
            .map(std::time::Duration::from_secs),
    );
    let mut observer = JsonlTrainingObserver;
    let (result, outcome) = transit_training::run_reference_multitask_with_policy_options(
        &datasets,
        &config,
        &checkpoint_root,
        resume.as_deref(),
        &control,
        CheckpointPolicy {
            every_steps: args.resumable.checkpoint_every_steps,
            every_seconds: args.resumable.checkpoint_every_seconds,
        },
        &metadata,
        args.resumable.fork_from_checkpoint,
        &mut observer,
    )?;
    finish_multitask_resumable_outcome(&args.output, &checkpoint_root, result, outcome)
}

fn command_encode_dataset(args: EncodeDatasetArgs) -> Result<()> {
    let checkpoint = load_checkpoint(&args.encoder)
        .with_context(|| format!("loading encoder checkpoint {}", args.encoder.display()))?;
    let graphs = args
        .graphs
        .iter()
        .map(|path| {
            GraphTensor::load(path).with_context(|| format!("loading graph {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let graph_refs = graphs.iter().collect::<Vec<_>>();
    let cache = build_embedding_cache(&checkpoint.encoder, &graph_refs)?;
    save_embedding_cache(&args.output, &cache)?;
    write_artifact_manifest(&args.output, "embedding-cache")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "completed",
            "output": args.output,
            "schemaVersion": transit_training::EMBEDDING_CACHE_SCHEMA_VERSION,
            "encoderFingerprint": cache.encoder_fingerprint,
            "graphs": cache.entries.len(),
            "fingerprint": cache.fingerprint
        }))?
    );
    Ok(())
}

fn load_head_config(path: Option<&Path>) -> Result<(CriticalityTrainingConfig, String)> {
    if let Some(path) = path {
        if let Ok(config) = load_config::<MultiTaskTrainingConfig>(path) {
            let fingerprint = json_fingerprint(&config.criticality)?;
            return Ok((config.criticality, fingerprint));
        }
        let config = load_config::<CriticalityTrainingConfig>(path)?;
        let fingerprint = json_fingerprint(&config)?;
        return Ok((config, fingerprint));
    }
    let config = CriticalityTrainingConfig::default();
    let fingerprint = json_fingerprint(&config)?;
    Ok((config, fingerprint))
}

fn command_train_heads(args: TrainHeadsArgs) -> Result<()> {
    if args.labels.len() > args.graphs.len() {
        bail!("provide at most one --labels file for each --graph");
    }
    let encoder_checkpoint = load_checkpoint(&args.encoder)
        .with_context(|| format!("loading encoder checkpoint {}", args.encoder.display()))?;
    let cache = load_embedding_cache(&args.embeddings)
        .with_context(|| format!("loading embedding cache {}", args.embeddings.display()))?;
    let graphs = args
        .graphs
        .iter()
        .map(|path| {
            GraphTensor::load(path).with_context(|| format!("loading graph {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let labels = (0..graphs.len())
        .map(|index| {
            args.labels
                .get(index)
                .map(|path| {
                    load_jsonl(path).with_context(|| format!("loading labels {}", path.display()))
                })
                .transpose()
                .map(|value| value.unwrap_or_default())
        })
        .collect::<Result<Vec<Vec<transit_labels::LineImpactLabel>>>>()?;
    let datasets = graphs
        .iter()
        .zip(&labels)
        .map(|(graph, labels)| (graph, labels.as_slice()))
        .collect::<Vec<_>>();
    let (mut config, config_fingerprint) = load_head_config(args.config.as_deref())?;
    if let Some(seed) = args.seed {
        config.seed = seed;
    }
    let mut observer = JsonlTrainingObserver;
    let (head, report) = train_criticality_head_cached_multi_with_observer(
        &encoder_checkpoint.encoder,
        &datasets,
        &cache,
        &config,
        &mut observer,
    )?;
    let training_run_id = encoder_checkpoint.training_run_id.clone();
    let dataset_fingerprint = encoder_checkpoint.dataset_fingerprint.clone();
    let model_id = encoder_checkpoint.model_id.clone();
    let checkpoint = ReferenceCheckpoint {
        encoder: encoder_checkpoint.encoder,
        head: Some(head),
        report: Some(report.clone()),
        representation: encoder_checkpoint.representation,
        config_fingerprint: Some(config_fingerprint),
        seed: Some(config.seed),
        training_run_id,
        dataset_fingerprint,
        model_id,
    };
    save_checkpoint(&args.output, &checkpoint)?;
    write_artifact_manifest(&args.output, "model-checkpoint")?;
    emit_checkpoint_created(&args.output, "criticality", config.epochs, config.epochs)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "completed",
            "output": args.output,
            "cacheFingerprint": cache.fingerprint,
            "report": report
        }))?
    );
    Ok(())
}

fn command_fine_tune(args: FineTuneArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    let base = load_checkpoint(&args.model)
        .with_context(|| format!("loading base model {}", args.model.display()))?;
    let mut config = args
        .config
        .as_deref()
        .map(load_config::<PretrainingConfig>)
        .transpose()?
        .unwrap_or_default();
    if let Some(steps) = args.steps {
        config.steps = steps;
    }
    apply_runtime_overrides(&mut config.runtime, &args.resumable)?;
    if config.steps == 0 {
        bail!("fine-tune steps must be positive");
    }
    let checkpoint_root = checkpoint_root(&args.output, args.resumable.checkpoint_dir.as_deref());
    let metadata = training_metadata(&graph, &config, &args.resumable)?;
    let base_checkpoint = if let Some(resume) =
        resolve_resume_checkpoint(&checkpoint_root, args.resumable.resume.as_deref())?
    {
        resume
    } else {
        let checkpoint = TrainingCheckpointV1 {
            schema_version: transit_training::TRAINING_CHECKPOINT_SCHEMA_VERSION,
            run_id: metadata.run_id.clone(),
            attempt_id: metadata.attempt_id.clone(),
            model: base,
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
            cursor: TrainingCursor::default(),
            sampler: SamplerState {
                seed: config.seed,
                graph_order: vec![graph.manifest.snapshot_id.clone()],
                ..SamplerState::default()
            },
            best_metrics: BestMetricState::default(),
            dataset_fingerprint: metadata.dataset_fingerprint.clone(),
            config_fingerprint: metadata.config_fingerprint.clone(),
            code_commit: metadata.code_commit.clone(),
            backend: "reference-cpu-decoder".into(),
            backend_version: metadata.backend_version.clone(),
            device_type: metadata.device_type.clone(),
            report: None,
            decoder_gradients: None,
            multi_task_phase: None,
        };
        save_training_checkpoint(&checkpoint_root, &checkpoint)?
    };
    let control = TrainingControl::with_policy(
        args.resumable.control_file.clone(),
        max_wall_time(args.resumable.max_wall_time_seconds),
        args.resumable
            .checkpoint_grace_seconds
            .map(std::time::Duration::from_secs),
    );
    let mut observer = JsonlTrainingObserver;
    let (session, outcome) = run_reference_pretraining_with_policy(
        &graph,
        &config,
        &checkpoint_root,
        Some(base_checkpoint.as_path()),
        &control,
        CheckpointPolicy {
            every_steps: args.resumable.checkpoint_every_steps,
            every_seconds: args.resumable.checkpoint_every_seconds,
        },
        &metadata,
        &mut observer,
    )?;
    finish_resumable_outcome(
        &args.output,
        &checkpoint_root,
        &config,
        session,
        outcome,
        &metadata,
    )
}

fn json_fingerprint<T: serde::Serialize>(value: &T) -> Result<String> {
    Ok(hex_digest(&sha256_bytes(&serde_json::to_vec(value)?)))
}

fn command_infer(args: InferArgs) -> Result<()> {
    let graph = GraphTensor::load(&args.graph)?;
    #[cfg(feature = "tch-backend")]
    if is_native_libtorch_model(&args.model)? {
        let mut predictions = native_prediction_file(
            &graph,
            &transit_training::predict_tch_model(&args.model, &graph)?,
        )?;
        predictions.model_id = Some(args.model_id);
        rank_by_accessibility(&mut predictions);
        save_predictions(&args.output, &predictions)?;
        write_artifact_manifest(&args.output, "inference-result")?;
        println!("{}", serde_json::to_string_pretty(&predictions)?);
        return Ok(());
    }
    let checkpoint = load_checkpoint(&args.model)?;
    let mut predictions = predict_reference(&checkpoint, &graph)?;
    predictions.model_id = Some(args.model_id);
    rank_by_accessibility(&mut predictions);
    save_predictions(&args.output, &predictions)?;
    write_artifact_manifest(&args.output, "inference-result")?;
    println!("{}", serde_json::to_string_pretty(&predictions)?);
    Ok(())
}

#[cfg(feature = "tch-backend")]
fn is_native_libtorch_model(path: &Path) -> Result<bool> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("reading model {}", path.display()))
        }
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return Ok(false),
    };
    Ok(
        value.get("backend").and_then(Value::as_str) == Some("libtorch")
            && value.get("weightsPath").and_then(Value::as_str).is_some(),
    )
}

#[cfg(feature = "tch-backend")]
fn native_prediction_file(
    graph: &GraphTensor,
    rows: &[transit_training::TchLineInference],
) -> Result<PredictionFile> {
    if rows.len() != graph.manifest.line_count {
        bail!(
            "native LibTorch inference returned {} rows for {} lines",
            rows.len(),
            graph.manifest.line_count
        );
    }
    let metric_names = vec![
        "accessibility_auc_loss".into(),
        "unreachable_share".into(),
        "mean_delay_reachable_seconds".into(),
        "p95_delay_reachable_seconds".into(),
        "mean_extra_transfers".into(),
        "stations_losing_all_service_share".into(),
    ];
    let predictions = rows
        .iter()
        .enumerate()
        .map(|(line, row)| {
            let metrics = denormalize_criticality_targets(
                row.criticality
                    .iter()
                    .copied()
                    .take(transit_model::CRITICALITY_OUTPUTS)
                    .collect::<Vec<_>>(),
            )
            .into_iter()
            .map(|value| {
                if value.is_finite() {
                    value.max(0.0)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
            let criticality = CriticalityPrediction {
                accessibility_loss: metrics.first().copied().unwrap_or(0.0),
                unreachable_share: metrics.get(1).copied().unwrap_or(0.0),
                mean_delay_seconds: metrics.get(2).copied().unwrap_or(0.0),
                p95_delay_seconds: metrics.get(3).copied().unwrap_or(0.0),
                extra_transfers: metrics.get(4).copied().unwrap_or(0.0),
                isolated_station_share: metrics.get(5).copied().unwrap_or(0.0),
                uncertainty: 0.0,
            };
            LinePrediction {
                line: line as u32,
                metrics,
                structural_uniqueness: 0.0,
                metric_percentiles: Vec::new(),
                criticality: Some(criticality),
                uncertainty: 0.0,
            }
        })
        .collect::<Vec<_>>();
    let line_names = graph
        .line_names
        .iter()
        .enumerate()
        .map(|(line, name)| (line.to_string(), name.clone()))
        .collect();
    let line_embeddings = rows
        .iter()
        .enumerate()
        .map(|(line, row)| transit_inference::LineEmbeddingRecord {
            line: line as u32,
            line_name: graph
                .line_names
                .get(line)
                .cloned()
                .unwrap_or_else(|| format!("Line {line}")),
            embedding: LineEmbedding {
                base: row.base.clone(),
                general: row.general.clone(),
                role: row.role.clone(),
                service: row.service.clone(),
                geometry: row.geometry.clone(),
                resilience: row.resilience.clone(),
            },
            anomaly_score: 0.0,
        })
        .collect();
    let mut result = PredictionFile {
        schema_version: 1,
        model_id: None,
        snapshot_id: graph.manifest.snapshot_id.clone(),
        metric_names,
        predictions,
        line_names,
        line_embeddings,
    };
    add_metric_percentiles(&mut result);
    Ok(result)
}

fn command_verify_top_lines(args: VerifyTopLinesArgs) -> Result<()> {
    let network = load_snapshot(&args.snapshot)?;
    let prediction_path = args.predictions;
    let predictions = load_predictions(&prediction_path)?;
    let router = Router::from_network(&network, RouterConfig::default())?;
    let label_config = LabelGenerationConfig::default();
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(
        &candidates,
        label_config.maximum_origins,
        &label_config.origin_sampling,
    );
    let departures = args
        .departure_times
        .iter()
        .map(|value| parse_departure_time(value))
        .collect::<Result<Vec<_>>>()?;
    let selected_lines = predictions
        .predictions
        .iter()
        .take(args.top_k)
        .map(|prediction| transit_domain::LineIndex(prediction.line))
        .collect::<Vec<_>>();
    let labels = generate_selected_line_removal_labels(
        &router,
        network.snapshot_id.clone(),
        &origins,
        &departures,
        &label_config,
        &selected_lines,
    );
    let verified: Vec<_> = labels
        .into_iter()
        .map(|label| {
            let display_name = network
                .lines
                .get(label.line.0 as usize)
                .map(|line| line.display_name.clone())
                .unwrap_or_else(|| label.line.to_string());
            json!({
                "line": label.line.0,
                "display_name": display_name,
                "exact_accessibility_auc_loss": label.accessibility_auc_loss,
                "exact_unreachable_share": label.unreachable_share,
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&verified)?);
    Ok(())
}

fn command_similar_lines(args: SimilarLinesArgs) -> Result<()> {
    let query_graph = GraphTensor::load(&args.query_graph)?;
    let candidate_graph = GraphTensor::load(&args.candidate_graph)?;
    let query_line = resolve_line_index(&query_graph, &args.query_line)?;
    let profile: SimilarityProfile = if args.role_weight.is_some()
        || args.service_weight.is_some()
        || args.geometry_weight.is_some()
        || args.resilience_weight.is_some()
    {
        SimilarityProfile::Weighted {
            role: args.role_weight.unwrap_or(0.0),
            service: args.service_weight.unwrap_or(0.0),
            geometry: args.geometry_weight.unwrap_or(0.0),
            resilience: args.resilience_weight.unwrap_or(0.0),
        }
    } else {
        args.profile
            .parse()
            .map_err(|error: String| anyhow::anyhow!(error))?
    };
    let checkpoint = args.encoder.as_deref().map(load_checkpoint).transpose()?;
    let fallback_encoder;
    let encoder = if let Some(checkpoint) = checkpoint.as_ref() {
        &checkpoint.encoder
    } else {
        fallback_encoder = ReferenceRelationalAutoencoder::new(ModelConfig::default());
        &fallback_encoder
    };
    let query_mask = MaskSelection::all_unmasked(&query_graph);
    let query_embeddings = encoder.encode(&query_graph, &query_mask)?;
    let candidate_mask = MaskSelection::all_unmasked(&candidate_graph);
    let candidate_embeddings = encoder.encode(&candidate_graph, &candidate_mask)?;
    let fallback_representation_encoder;
    let (query_representations, candidate_representations) = if let Some(representation) =
        checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.representation.as_ref())
    {
        (
            representation.encode(&query_graph, &query_embeddings)?,
            representation.encode(&candidate_graph, &candidate_embeddings)?,
        )
    } else {
        fallback_representation_encoder = ReferenceLineRepresentationEncoder::default();
        (
            fallback_representation_encoder.encode(&query_graph, &query_embeddings)?,
            fallback_representation_encoder.encode(&candidate_graph, &candidate_embeddings)?,
        )
    };
    let matches = rank_similar_lines(
        &query_graph,
        &query_representations,
        query_line,
        &candidate_graph,
        &candidate_representations,
        &profile,
        args.top_k,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "query": {
                "snapshot": query_graph.manifest.snapshot_id,
                "line": query_line,
                "line_name": query_graph.line_names.get(query_line),
            },
            "profile": profile.to_string(),
            "matches": matches,
        }))?
    );
    Ok(())
}

fn resolve_line_index(graph: &GraphTensor, query: &str) -> Result<usize> {
    if let Ok(index) = query.trim().parse::<usize>() {
        if index < graph.manifest.line_count {
            return Ok(index);
        }
    }
    let needle = query.trim().to_ascii_lowercase();
    if needle.is_empty() {
        bail!("query line cannot be blank");
    }
    let exact: Vec<usize> = graph
        .line_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| (name.to_ascii_lowercase() == needle).then_some(index))
        .collect();
    if exact.len() == 1 {
        return Ok(exact[0]);
    }
    let partial: Vec<usize> = graph
        .line_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| name.to_ascii_lowercase().contains(&needle).then_some(index))
        .collect();
    if partial.len() == 1 {
        return Ok(partial[0]);
    }
    if exact.len() > 1 || partial.len() > 1 {
        bail!("query line {query:?} is ambiguous; use a numeric line index");
    }
    bail!("could not find query line {query:?}")
}

fn command_demo(args: DemoArgs) -> Result<()> {
    let feed = demo_feed();
    let service_date = NaiveDate::from_ymd_opt(2026, 9, 7).expect("valid demo date");
    let options = CompileOptions::for_date(service_date)
        .with_scope("synthetic demo")
        .with_source_name("synthetic-demo");
    let network = compile(&feed, &options)?;
    let snapshot_dir = args.output.join("snapshot");
    save_snapshot(&network, &snapshot_dir)?;
    let graph = GraphTensor::from_network(&network)?;
    let graph_dir = args.output.join("graph");
    graph.save(&graph_dir, &network)?;
    let router = Router::from_network(&network, RouterConfig::default())?;
    let mut label_config = LabelGenerationConfig {
        maximum_origins: network.stations.len(),
        ..LabelGenerationConfig::default()
    };
    label_config.origin_sampling.seed = 7;
    let candidates = network
        .stations
        .iter()
        .map(|station| OriginCandidate {
            index: station.index,
            latitude: station.latitude,
            longitude: station.longitude,
            transfer_degree: station.transfer_degree,
        })
        .collect::<Vec<_>>();
    let origins = sample_origins(
        &candidates,
        label_config.maximum_origins,
        &label_config.origin_sampling,
    );
    let departures = vec![
        parse_departure_time("07:30")?,
        parse_departure_time("08:30")?,
    ];
    let labels = generate_line_removal_labels(
        &router,
        network.snapshot_id.clone(),
        &origins,
        &departures,
        &label_config,
    );
    let labels_path = args.output.join("labels.jsonl");
    save_jsonl(&labels_path, &labels)?;
    save_label_manifest_with_metadata(
        &labels_path,
        &label_config,
        origins.len(),
        network.snapshot_id.clone(),
        &departures,
    )?;
    write_artifact_manifest(&labels_path, "criticality-labels")?;
    let pretraining = PretrainingConfig {
        model: ModelConfig {
            hidden_dimension: 32,
            temporal_dimension: 16,
            graph_layers: 2,
            dropout: 0.0,
        },
        mask: MaskConfig {
            station_feature_probability: 0.5,
            line_feature_probability: 0.5,
            ..MaskConfig::default()
        },
        steps: 25,
        ..PretrainingConfig::default()
    };
    let mut observer = JsonlTrainingObserver;
    let (encoder, pretrain_report) =
        train_reference_autoencoder_with_observer(&graph, &pretraining, &mut observer)?;
    let (head, head_report) = transit_training::train_criticality_head_with_observer(
        &encoder,
        &graph,
        &labels,
        &CriticalityTrainingConfig {
            epochs: 20,
            ..CriticalityTrainingConfig::default()
        },
        &mut observer,
    )?;
    let checkpoint_path = args.output.join("model.json");
    save_checkpoint(
        &checkpoint_path,
        &ReferenceCheckpoint {
            encoder,
            head: Some(head),
            report: Some(head_report.clone()),
            representation: None,
            config_fingerprint: None,
            seed: Some(pretraining.seed),
            training_run_id: None,
            dataset_fingerprint: None,
            model_id: Some("model-demo-reference".into()),
        },
    )?;
    write_artifact_manifest(&checkpoint_path, "model-checkpoint")?;
    emit_checkpoint_created(&checkpoint_path, "criticality", 20, 20)?;
    let mut predictions = predict_reference(&load_checkpoint(&checkpoint_path)?, &graph)?;
    predictions.model_id = Some("model-demo-reference".into());
    rank_by_accessibility(&mut predictions);
    let predictions_path = args.output.join("predictions.json");
    save_predictions(&predictions_path, &predictions)?;
    write_artifact_manifest(&predictions_path, "inference-result")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "snapshot_id": network.snapshot_id,
            "stations": network.stations.len(),
            "lines": network.lines.len(),
            "labels": labels.len(),
            "pretraining": pretrain_report,
            "criticality": head_report,
            "output": args.output,
        }))?
    );
    Ok(())
}

fn source_metadata_path(input: &std::path::Path) -> Option<PathBuf> {
    let candidate = if input.is_dir() {
        input.join("source.json")
    } else {
        input.parent()?.join("source.json")
    };
    candidate.is_file().then_some(candidate)
}

static RUNTIME_EVENT_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static ARTIFACT_MANIFEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct JsonlTrainingObserver;

impl TrainingObserver for JsonlTrainingObserver {
    fn phase_started(&mut self, phase: &str, total: Option<usize>) {
        let mut payload = serde_json::Map::new();
        payload.insert("phase".into(), json!(phase));
        if let Some(total) = total {
            payload.insert("total".into(), json!(total));
        }
        let _ = emit_runtime_event("phase.started", Value::Object(payload));
    }

    fn epoch_started(&mut self, phase: &str, epoch: usize, total: usize) {
        let _ = emit_runtime_event(
            "epoch.started",
            json!({"phase": phase, "epoch": epoch, "total": total}),
        );
    }

    fn metric(&mut self, phase: &str, epoch: usize, step: usize, name: &str, value: f32) {
        let _ = emit_runtime_event(
            "metric",
            json!({"phase": phase, "epoch": epoch, "step": step, "name": name, "value": value}),
        );
    }

    fn learning_rate_changed(&mut self, phase: &str, step: usize, value: f32) {
        let _ = emit_runtime_event(
            "learning-rate.changed",
            json!({"phase": phase, "step": step, "value": value}),
        );
    }

    fn heartbeat(&mut self, phase: &str, step: usize) {
        let _ = emit_runtime_event("heartbeat", json!({"phase": phase, "step": step}));
    }

    fn phase_completed(&mut self, phase: &str) {
        let _ = emit_runtime_event("phase.completed", json!({"phase": phase}));
    }

    fn checkpoint_started(&mut self, phase: &str, step: usize) {
        let _ = emit_runtime_event("checkpoint.started", json!({"phase": phase, "step": step}));
    }

    fn checkpoint_committed(&mut self, phase: &str, step: usize, path: &Path) {
        let _ = emit_runtime_event(
            "checkpoint.committed",
            json!({"phase": phase, "step": step, "path": path.to_string_lossy()}),
        );
    }
}

fn emit_checkpoint_created(path: &Path, phase: &str, epoch: usize, step: usize) -> Result<()> {
    emit_runtime_event(
        "checkpoint.created",
        json!({
            "phase": phase,
            "epoch": epoch,
            "step": step,
            "path": path.to_string_lossy()
        }),
    )
}

/// Emit machine-readable events only when the Studio worker opts into the
/// protocol. Human stdout remains a separate diagnostic stream.
fn emit_runtime_event(event_type: &str, payload: Value) -> Result<()> {
    let Some(event_path) = std::env::var_os("TRANSIT_EVENT_FILE") else {
        return Ok(());
    };
    let Some(run_id) = std::env::var_os("TRANSIT_RUN_ID") else {
        return Ok(());
    };
    let run_id = run_id.to_string_lossy().into_owned();
    let mut event = serde_json::Map::new();
    event.insert("schemaVersion".into(), json!(1));
    let sequence = RUNTIME_EVENT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    event.insert("seq".into(), json!(sequence));
    event.insert("runId".into(), json!(run_id));
    event.insert(
        "timestamp".into(),
        json!(Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)),
    );
    event.insert("type".into(), json!(event_type));
    if let Some(attempt_id) = std::env::var_os("TRANSIT_ATTEMPT_ID") {
        event.insert("attemptId".into(), json!(attempt_id.to_string_lossy()));
    }
    event.insert("attemptSeq".into(), json!(sequence));
    if let Value::Object(fields) = payload {
        event.extend(fields);
    }
    let path = PathBuf::from(event_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    serde_json::to_writer(&mut file, &Value::Object(event))?;
    file.write_all(b"\n")?;
    Ok(())
}

fn artifact_manifest_path(output: &Path) -> PathBuf {
    if output.is_dir() {
        output.join("artifact-manifest.json")
    } else {
        let name = output
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("output");
        output
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{name}.artifact-manifest.json"))
    }
}

fn artifact_file_entries(path: &Path, root: &Path, entries: &mut Vec<Value>) -> Result<()> {
    let path = fs::canonicalize(path)
        .with_context(|| format!("resolving artifact output {}", path.display()))?;
    if path.is_file() {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        if name == "events.jsonl"
            || name == "artifact-manifest.json"
            || name.ends_with(".artifact-manifest.json")
        {
            return Ok(());
        }
        let bytes = fs::read(&path)?;
        let relative = path
            .strip_prefix(root)
            .with_context(|| {
                format!(
                    "artifact output {} is outside repository root {}",
                    path.display(),
                    root.display()
                )
            })?
            .to_string_lossy()
            .replace('\\', "/");
        entries.push(json!({
            "path": relative,
            "sizeBytes": bytes.len(),
            "sha256": hex_digest(&sha256_bytes(&bytes)),
        }));
        return Ok(());
    }
    if path.is_dir() {
        for child in fs::read_dir(&path)? {
            artifact_file_entries(&child?.path(), root, entries)?;
        }
    }
    Ok(())
}

/// Write the shared artifact-manifest.v1 contract next to a Rust output.
/// Existing manifests are immutable and must describe the same output.
fn write_artifact_manifest(output: &Path, kind: &str) -> Result<()> {
    write_artifact_manifest_for_paths(output, kind, &[output])
}

fn artifact_json(path: &Path, description: &str) -> Result<Value> {
    let bytes =
        fs::read(path).with_context(|| format!("reading {description} {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding {description} {}", path.display()))
}

fn artifact_metadata(output: &Path, kind: &str) -> Result<Value> {
    match kind {
        "routing-baseline" => {
            let baseline = artifact_json(output, "routing baseline")?;
            let version = baseline
                .get("router_algorithm_version")
                .and_then(Value::as_str)
                .context("routing baseline is missing router_algorithm_version")?;
            if version != ROUTER_ALGORITHM_VERSION {
                bail!(
                    "routing baseline uses router {}; expected {}",
                    version,
                    ROUTER_ALGORITHM_VERSION
                );
            }
            let snapshot = baseline
                .get("snapshot_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .context("routing baseline is missing snapshot_id")?;
            Ok(json!({
                "snapshotId": snapshot,
                "routerAlgorithmVersion": ROUTER_ALGORITHM_VERSION,
            }))
        }
        "criticality-labels" => {
            let manifest_path = output.with_extension("manifest.json");
            let manifest = artifact_json(&manifest_path, "label manifest")?;
            let schema = manifest
                .get("schema_version")
                .and_then(Value::as_str)
                .context("label manifest is missing schema_version")?;
            if schema != LABEL_MANIFEST_SCHEMA_VERSION {
                bail!(
                    "unsupported label manifest schema {}; expected {}",
                    schema,
                    LABEL_MANIFEST_SCHEMA_VERSION
                );
            }
            let version = manifest
                .get("router_algorithm_version")
                .and_then(Value::as_str)
                .context("label manifest is missing router_algorithm_version")?;
            if version != ROUTER_ALGORITHM_VERSION {
                bail!(
                    "label manifest uses router {}; expected {}",
                    version,
                    ROUTER_ALGORITHM_VERSION
                );
            }
            let snapshot = manifest
                .get("snapshot_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .context("label manifest is missing snapshot_id")?;
            let policy = manifest
                .get("policy_fingerprint")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .context("label manifest is missing policy_fingerprint")?;

            let batch_path = output.with_extension("batch.json");
            if batch_path.is_file() {
                let batch = artifact_json(&batch_path, "label batch manifest")?;
                let batch_schema = batch
                    .get("schema_version")
                    .and_then(Value::as_str)
                    .context("label batch manifest is missing schema_version")?;
                if batch_schema != LABEL_BATCH_SCHEMA_VERSION {
                    bail!(
                        "unsupported label batch schema {}; expected {}",
                        batch_schema,
                        LABEL_BATCH_SCHEMA_VERSION
                    );
                }
                let batch_version = batch
                    .get("router_algorithm_version")
                    .and_then(Value::as_str)
                    .context("label batch manifest is missing router_algorithm_version")?;
                if batch_version != ROUTER_ALGORITHM_VERSION {
                    bail!(
                        "label batch manifest uses router {}; expected {}",
                        batch_version,
                        ROUTER_ALGORITHM_VERSION
                    );
                }
                if batch.get("snapshot_id").and_then(Value::as_str) != Some(snapshot)
                    || batch.get("policy_fingerprint").and_then(Value::as_str) != Some(policy)
                {
                    bail!("label and batch manifests do not describe the same inputs");
                }
            }

            Ok(json!({
                "snapshotId": snapshot,
                "routerAlgorithmVersion": ROUTER_ALGORITHM_VERSION,
                "labelSchemaVersion": LABEL_MANIFEST_SCHEMA_VERSION,
                "policyFingerprint": policy,
            }))
        }
        _ => Ok(json!({})),
    }
}

fn write_artifact_manifest_for_paths(output: &Path, kind: &str, paths: &[&Path]) -> Result<()> {
    if !output.exists() {
        bail!("cannot manifest missing Rust output {}", output.display());
    }
    let artifact_root = std::env::var_os("TRANSIT_LAB_ARTIFACT_ROOT")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    let root = fs::canonicalize(&artifact_root)
        .with_context(|| format!("resolving artifact root {}", artifact_root.display()))?;
    let mut files = Vec::new();
    for path in paths {
        artifact_file_entries(path, &root, &mut files)?;
    }
    files.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    files.dedup_by(|left, right| left["path"] == right["path"]);
    if files.is_empty() {
        bail!("cannot manifest empty Rust output {}", output.display());
    }
    let metadata = artifact_metadata(output, kind)?;
    let file_bytes = serde_json::to_vec(&files)?;
    let digest = hex_digest(&sha256_bytes(&file_bytes));
    let manifest = json!({
        "schemaVersion": 1,
        "artifactId": format!("artifact-{kind}-{}", &digest[..24]),
        "kind": kind,
        "fingerprint": digest,
        "sha256": if files.len() == 1 { files[0]["sha256"].clone() } else { json!(hex_digest(&sha256_bytes(&file_bytes))) },
        "createdAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        "producingRunId": std::env::var("TRANSIT_RUN_ID").ok(),
        "inputs": [],
        "gitCommit": std::env::var("TRANSIT_LAB_GIT_COMMIT").unwrap_or_else(|_| "working-tree".into()),
        "configuration": {
            "producer": "transit-cli",
            "kind": kind,
            "configFingerprint": std::env::var("TRANSIT_CONFIG_FINGERPRINT").ok()
        },
        "files": files,
        "metadata": metadata
    });
    let manifest_path = artifact_manifest_path(output);
    if let Ok(existing) = fs::read(&manifest_path) {
        let existing: Value =
            serde_json::from_slice(&existing).context("decoding existing artifact manifest")?;
        if existing["fingerprint"] != manifest["fingerprint"]
            || existing["files"] != manifest["files"]
            || existing["kind"] != manifest["kind"]
            || existing["metadata"] != manifest["metadata"]
        {
            bail!(
                "refusing to overwrite immutable artifact manifest {}",
                manifest_path.display()
            );
        }
        return Ok(());
    }
    let encoded = serde_json::to_vec_pretty(&manifest)?;
    let mut expected_bytes = encoded.clone();
    expected_bytes.push(b'\n');
    let temporary = manifest_path.with_file_name(format!(
        ".{}-tmp-{}-{}",
        manifest_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact-manifest.json"),
        std::process::id(),
        ARTIFACT_MANIFEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        file.write_all(&expected_bytes)?;
        file.sync_all()
            .with_context(|| format!("syncing {}", temporary.display()))?;
        drop(file);

        match fs::hard_link(&temporary, &manifest_path) {
            Ok(()) => {
                fs::remove_file(&temporary).with_context(|| {
                    format!(
                        "removing temporary artifact manifest {}",
                        temporary.display()
                    )
                })?;
                sync_directory(manifest_path.parent().unwrap_or_else(|| Path::new(".")))?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing = fs::read(&manifest_path).with_context(|| {
                    format!(
                        "reading competing artifact manifest {}",
                        manifest_path.display()
                    )
                })?;
                let existing: Value = serde_json::from_slice(&existing).with_context(|| {
                    format!(
                        "decoding competing artifact manifest {}",
                        manifest_path.display()
                    )
                })?;
                if existing["fingerprint"] == manifest["fingerprint"]
                    && existing["files"] == manifest["files"]
                    && existing["kind"] == manifest["kind"]
                    && existing["metadata"] == manifest["metadata"]
                {
                    Ok(())
                } else {
                    bail!(
                        "refusing to overwrite immutable artifact manifest {}",
                        manifest_path.display()
                    )
                }
            }
            Err(error) => Err(error).with_context(|| {
                format!("publishing artifact manifest {}", manifest_path.display())
            }),
        }
    })();
    let _ = fs::remove_file(&temporary);
    result?;

    let uri = fs::canonicalize(&manifest_path)
        .ok()
        .and_then(|path| {
            path.strip_prefix(&root)
                .ok()
                .map(|value| value.to_string_lossy().replace('\\', "/"))
        })
        .unwrap_or_else(|| manifest_path.to_string_lossy().replace('\\', "/"));
    emit_runtime_event(
        "artifact.created",
        json!({
            "artifactId": manifest["artifactId"],
            "artifactKind": kind,
            "uri": uri,
            "sha256": manifest["sha256"],
        }),
    )?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)
            .with_context(|| format!("opening directory {} for sync", path.display()))?
            .sync_all()
            .with_context(|| format!("syncing directory {}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn demo_feed() -> GtfsFeed {
    let stops: Vec<StopRecord> = [
        ("a", "Alpha", 48.1000, 16.1000),
        ("b", "Bravo", 48.1010, 16.1010),
        ("c", "Central", 48.1020, 16.1020),
        ("d", "Delta", 48.1030, 16.1030),
        ("e", "Echo", 48.1040, 16.1040),
    ]
    .into_iter()
    .map(|(id, name, lat, lon)| StopRecord {
        stop_id: id.into(),
        stop_name: Some(name.into()),
        stop_lat: Some(lat.to_string()),
        stop_lon: Some(lon.to_string()),
        location_type: Some("0".into()),
        ..StopRecord::default()
    })
    .collect();
    let routes = vec![
        RouteRecord {
            route_id: "blue".into(),
            agency_id: Some("demo".into()),
            route_short_name: Some("Blue".into()),
            route_type: Some("1".into()),
            ..RouteRecord::default()
        },
        RouteRecord {
            route_id: "red".into(),
            agency_id: Some("demo".into()),
            route_short_name: Some("Red".into()),
            route_type: Some("3".into()),
            ..RouteRecord::default()
        },
        RouteRecord {
            route_id: "green".into(),
            agency_id: Some("demo".into()),
            route_short_name: Some("Green".into()),
            route_type: Some("3".into()),
            ..RouteRecord::default()
        },
    ];
    let mut trips = Vec::new();
    let mut stop_times = Vec::new();
    let mut add_trip = |trip_id: &str, route_id: &str, stops: &[&str], start: u32| {
        trips.push(TripRecord {
            route_id: route_id.into(),
            service_id: "weekday".into(),
            trip_id: trip_id.into(),
            direction_id: Some("0".into()),
            ..TripRecord::default()
        });
        for (position, stop_id) in stops.iter().enumerate() {
            let arrival = start + position as u32 * 300;
            let departure = arrival + if position + 1 == stops.len() { 0 } else { 30 };
            stop_times.push(StopTimeRecord {
                trip_id: trip_id.into(),
                arrival_time: format_gtfs_time(arrival),
                departure_time: format_gtfs_time(departure),
                stop_id: (*stop_id).into(),
                stop_sequence: position.to_string(),
                ..StopTimeRecord::default()
            });
        }
    };
    for (index, start) in [27_000, 30_600, 43_200].into_iter().enumerate() {
        add_trip(&format!("blue-{index}"), "blue", &["a", "b", "c"], start);
        add_trip(
            &format!("green-{index}"),
            "green",
            &["a", "b", "c"],
            start + 600,
        );
        add_trip(
            &format!("red-{index}"),
            "red",
            &["c", "d", "e"],
            start + 240,
        );
    }
    let calendars = vec![CalendarRecord {
        service_id: "weekday".into(),
        monday: "1".into(),
        tuesday: "1".into(),
        wednesday: "1".into(),
        thursday: "1".into(),
        friday: "1".into(),
        saturday: "0".into(),
        sunday: "0".into(),
        start_date: "20260101".into(),
        end_date: "20261231".into(),
    }];
    let mut row_counts = BTreeMap::new();
    row_counts.insert("stops.txt".into(), stops.len());
    row_counts.insert("routes.txt".into(), routes.len());
    row_counts.insert("trips.txt".into(), trips.len());
    row_counts.insert("stop_times.txt".into(), stop_times.len());
    GtfsFeed {
        source_path: PathBuf::from("synthetic://demo"),
        source_hash: sha256_bytes(b"transit-lab-demo-feed-v1"),
        stops,
        routes,
        trips,
        stop_times,
        calendars,
        calendar_dates: Vec::new(),
        transfers: Vec::new(),
        pathways: Vec::new(),
        validation: ValidationReport {
            checked_files: vec![
                "stops.txt".into(),
                "routes.txt".into(),
                "trips.txt".into(),
                "stop_times.txt".into(),
                "calendar.txt".into(),
            ],
            row_counts,
            ..ValidationReport::default()
        },
    }
}

fn format_gtfs_time(seconds: u32) -> String {
    format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use gtfs_compile::{compile, CompileOptions};
    use gtfs_ingest::GtfsFeed;
    use std::fs;
    use tempfile::tempdir;
    use transit_training::NoopTrainingObserver;

    fn fixture() -> (transit_domain::CompiledNetwork, GraphTensor) {
        let feed = GtfsFeed::from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/synthetic-feeds/basic"
        ))
        .expect("synthetic fixture loads");
        let network = compile(
            &feed,
            &CompileOptions::for_date(NaiveDate::from_ymd_opt(2026, 9, 7).unwrap())
                .with_scope("synthetic fixture")
                .with_source_name("fixture"),
        )
        .expect("synthetic fixture compiles");
        let graph = GraphTensor::from_network(&network).expect("synthetic graph builds");
        (network, graph)
    }

    fn graph_variant(graph: &GraphTensor, snapshot: &str, system: &str) -> GraphTensor {
        let mut variant = graph.clone();
        variant.manifest.snapshot_id = snapshot.into();
        variant.manifest.network_system_id = system.into();
        variant
    }

    fn empty_parts<'a>(
        train: &'a GraphTensor,
        validation: &'a GraphTensor,
        test: &'a GraphTensor,
    ) -> Vec<DatasetPart<'a>> {
        vec![
            DatasetPart {
                graph: train,
                labels: &[],
                graph_directory: "graphs/train".into(),
                label_file: "labels/train.jsonl".into(),
                split: "train".into(),
            },
            DatasetPart {
                graph: validation,
                labels: &[],
                graph_directory: "graphs/validation".into(),
                label_file: "labels/validation.jsonl".into(),
                split: "validation".into(),
            },
            DatasetPart {
                graph: test,
                labels: &[],
                graph_directory: "graphs/test".into(),
                label_file: "labels/test.jsonl".into(),
                split: "test".into(),
            },
        ]
    }

    #[test]
    fn dataset_training_checkpoint_contains_only_the_train_graph_order() {
        let (network, graph) = fixture();
        let directory = tempdir().unwrap();
        let train = graph_variant(&graph, "snapshot-train", "system-train");
        let validation = graph_variant(&graph, "snapshot-validation", "system-validation");
        let test = graph_variant(&graph, "snapshot-test", "system-test");

        let mut train_network = network;
        train_network.snapshot_id = "snapshot-train".into();
        train
            .save(&directory.path().join("graphs/train"), &train_network)
            .unwrap();
        fs::create_dir_all(directory.path().join("graphs/validation")).unwrap();
        fs::create_dir_all(directory.path().join("graphs/test")).unwrap();
        fs::create_dir_all(directory.path().join("labels")).unwrap();
        fs::write(directory.path().join("labels/train.jsonl"), b"").unwrap();
        fs::write(
            directory.path().join("labels/validation.jsonl"),
            b"not a label file\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("labels/test.jsonl"),
            b"not a label file\n",
        )
        .unwrap();

        let manifest = create_dataset_manifest(
            &empty_parts(&train, &validation, &test),
            json!({
                "strategy": "system-level",
                "train": ["snapshot-train"],
                "validation": ["snapshot-validation"],
                "test": ["snapshot-test"]
            }),
            None,
            0,
            Vec::new(),
            None,
        )
        .unwrap();
        save_dataset_manifest(&directory.path().join("dataset-manifest.json"), &manifest).unwrap();

        let mut args = MultiTaskArgs {
            dataset: Some(directory.path().to_path_buf()),
            split: "train".into(),
            graphs: Vec::new(),
            labels: Vec::new(),
            allow_unpartitioned_input: false,
            allow_nontrain_training_split: false,
            config: None,
            output: directory.path().join("model.json"),
            seed: None,
            resumable: ResumableArgs {
                checkpoint_dir: None,
                resume: None,
                control_file: None,
                checkpoint_every_steps: None,
                checkpoint_every_seconds: None,
                max_wall_time_seconds: None,
                checkpoint_grace_seconds: None,
                run_id: None,
                backend: None,
                fork_from_checkpoint: false,
                device: None,
                dtype: None,
                cpu_threads: None,
                rayon_threads: None,
                gradient_accumulation: None,
            },
        };
        args.split = "test".into();
        let error = match load_multitask_datasets(&args) {
            Ok(_) => panic!("non-train split should require an explicit acknowledgement"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("--allow-nontrain-training-split"));
        args.split = "train".into();
        let (graphs, labels) = load_multitask_datasets(&args).unwrap();
        assert_eq!(
            graphs
                .iter()
                .map(|graph| graph.manifest.snapshot_id.as_str())
                .collect::<Vec<_>>(),
            vec!["snapshot-train"]
        );
        assert!(labels.iter().all(Vec::is_empty));

        let graph_refs = graphs.iter().collect::<Vec<_>>();
        let control = TrainingControl::new(None, None);
        let mut config = MultiTaskTrainingConfig::default();
        config.pretraining.model.hidden_dimension = 8;
        config.pretraining.model.graph_layers = 1;
        config.pretraining.steps = 1;
        config.metric_epochs = 0;
        config.criticality.epochs = 0;
        let metadata = CheckpointMetadata {
            run_id: "run-dataset-boundary".into(),
            dataset_fingerprint: manifest.fingerprint,
            config_fingerprint: "config-dataset-boundary".into(),
            ..CheckpointMetadata::default()
        };
        let (_, outcome) = transit_training::run_reference_pretraining_multi_with_policy_options(
            &graph_refs,
            &config.pretraining,
            &directory.path().join("checkpoints"),
            None,
            &control,
            CheckpointPolicy {
                every_steps: Some(1),
                every_seconds: None,
            },
            &metadata,
            false,
            &mut NoopTrainingObserver,
        )
        .unwrap();
        let ReferenceTrainingOutcome::Completed { checkpoint_path } = outcome else {
            panic!("training fixture did not complete");
        };
        let (checkpoint, _) = load_training_checkpoint(&checkpoint_path).unwrap();
        assert_eq!(
            checkpoint.sampler.graph_order,
            vec!["snapshot-train".to_owned()]
        );
    }
}
